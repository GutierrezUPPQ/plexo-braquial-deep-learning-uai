"""Small CPU-friendly U-Nets for public regional-anesthesia ultrasound.

Only training and validation data are accepted by training functions. Images use
raw RGB float/uint8 [0,255]; masks use [0,1]. Metadata is fitted on TRAIN ONLY.
The saved inference graph consists entirely of standard Keras layers. Loading
with compile=False requires neither custom layers nor this module's loss class.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = 128
SEED = 2026
NUMERIC_COLUMNS = ("age", "height", "BMI")
CATEGORICAL_COLUMNS = ("gender", "left_right", "gain")
MISSING_CATEGORY = "__MISSING__"

def configure_runtime(seed: int = SEED, threads: int = 6) -> dict[str, Any]:
    """Set seeds and bounded CPU threads before creating TensorFlow tensors.

    Deterministic operations are requested. TensorFlow cannot change its global
    thread settings after initialization; this case is reported, not hidden.
    Each training dataset additionally has a private thread pool of size one.
    """
    if threads < 1:
        raise ValueError("threads must be >= 1")
    thread_settings_applied = True
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(threads)
    except RuntimeError:
        thread_settings_applied = False
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return {
        "seed": seed,
        "requested_threads": threads,
        "thread_settings_applied": thread_settings_applied,
        "deterministic_ops_requested": True,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
    }


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _category(value: Any) -> str:
    return MISSING_CATEGORY if pd.isna(value) else "value:" + str(value)


@dataclass
class TabularPreprocessor:
    """Training-fitted median imputation, z scores, and one-hot encoding.

    Numerical means and population standard deviations (ddof=0) are computed
    after training-only median imputation. A constant variable uses scale 1;
    a wholly missing training variable uses median 0. Categorical missingness
    has its own level if observed in training. All unseen levels, including
    missingness first observed outside training, produce an all-zero block.
    """

    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_statistics: dict[str, dict[str, Any]]
    categorical_levels: dict[str, list[str]]
    fit_row_count: int

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        numeric_columns: Sequence[str] = NUMERIC_COLUMNS,
        categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    ) -> "TabularPreprocessor":
        """Fit on TRAINING ROWS ONLY; labels/IDs must not be feature columns."""
        numeric_columns = list(numeric_columns)
        categorical_columns = list(categorical_columns)
        columns = numeric_columns + categorical_columns
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Feature columns must be nonempty and unique")
        if len(train_frame) == 0:
            raise ValueError("Training metadata is empty")
        missing = set(columns) - set(train_frame.columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")
        statistics: dict[str, dict[str, Any]] = {}
        for column in numeric_columns:
            values = _numeric_values(train_frame[column])
            all_missing = bool(np.isnan(values).all())
            median = 0.0 if all_missing else float(np.nanmedian(values))
            imputed = np.where(np.isnan(values), median, values)
            mean = float(imputed.mean())
            raw_scale = float(imputed.std(ddof=0))
            statistics[column] = {
                "median": median,
                "mean": mean,
                "scale": raw_scale if raw_scale > 0.0 else 1.0,
                "all_missing_in_train": all_missing,
            }
        levels = {
            column: sorted({_category(value) for value in train_frame[column]})
            for column in categorical_columns
        }
        return cls(numeric_columns, categorical_columns, statistics, levels,
                   len(train_frame))

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_columns + [
            f"{column}={level}"
            for column in self.categorical_columns
            for level in self.categorical_levels[column]
        ]

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply immutable training statistics; unknown categories are all zero."""
        missing = (set(self.numeric_columns + self.categorical_columns)
                   - set(frame.columns))
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")
        output = np.zeros((len(frame), self.output_dim), dtype=np.float32)
        cursor = 0
        for column in self.numeric_columns:
            values = _numeric_values(frame[column])
            stats = self.numeric_statistics[column]
            values = np.where(np.isnan(values), stats["median"], values)
            output[:, cursor] = (values - stats["mean"]) / stats["scale"]
            cursor += 1
        for column in self.categorical_columns:
            levels = self.categorical_levels[column]
            mapping = {level: index for index, level in enumerate(levels)}
            for row, value in enumerate(frame[column]):
                index = mapping.get(_category(value))
                if index is not None:
                    output[row, cursor + index] = 1.0
            cursor += len(levels)
        if not np.isfinite(output).all():
            raise ValueError("Transformed metadata contains non-finite values")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "training_fitted_tabular_preprocessor",
            "version": 1,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_statistics": self.numeric_statistics,
            "categorical_levels": self.categorical_levels,
            "feature_names": self.feature_names,
            "fit_row_count": self.fit_row_count,
            "unknown_category_policy": "all_zero",
            "missing_category_token": MISSING_CATEGORY,
        }

    def save(self, path: str | Path) -> None:
        """Write plain JSON; no pickle or executable deserialization is used."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2,
                       allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload.get("format") != "training_fitted_tabular_preprocessor"
                or payload.get("version") != 1):
            raise ValueError("Unsupported tabular preprocessor JSON format")
        processor = cls(
            list(payload["numeric_columns"]),
            list(payload["categorical_columns"]),
            payload["numeric_statistics"],
            payload["categorical_levels"],
            int(payload["fit_row_count"]),
        )
        if processor.feature_names != payload["feature_names"]:
            raise ValueError("Preprocessor feature order is inconsistent")
        return processor


@tf.keras.utils.register_keras_serializable(package="regional_anesthesia")
class BCEDiceLoss(tf.keras.losses.Loss):
    """Mean of per-pixel BCE and per-frame soft Dice loss, weights 0.5/0.5.

    Smoothing=1.0 makes the loss finite for empty reference masks. This is an
    optimization loss, not the primary patient-averaged positive-frame metric.
    """
    def __init__(self, smooth: float = 1.0, name: str = "bce_dice_loss",
                 reduction: str = "sum_over_batch_size", **kwargs: Any):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.smooth = float(smooth)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, y_pred.dtype)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce = tf.reduce_mean(bce, axis=[1, 2])
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 0.5 * bce + 0.5 * (1.0 - dice)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "smooth": self.smooth}


@tf.keras.utils.register_keras_serializable(package="regional_anesthesia")
def soft_dice_all_frames(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Training diagnostic only; includes empty frames and uses probabilities."""
    y_true = tf.cast(y_true, y_pred.dtype)
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) + 1.0
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) + 1.0
    return numerator / denominator


def build_unet(
    metadata_dim: int = 0,
    image_size: int = IMAGE_SIZE,
    base_filters: int = 16,
    seed: int = SEED,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Three downsamplings: 16/32/64 encoder, 128-channel bottleneck.

    Each block has two 3x3 ReLU convolutions. Decoder uses bilinear upsampling,
    skip concatenation, and two 3x3 convolutions. Metadata is encoded by Dense32,
    broadcast at the bottleneck, and fused by a 1x1 convolution. The baseline
    uses the same projection and decoder but has no metadata input. No flips,
    augmentation layers, Lambda layers, or target information enter this graph.
    """
    if image_size % 8 or image_size < 32 or metadata_dim < 0 or base_filters < 1:
        raise ValueError("Require size >=32 divisible by8, metadata_dim>=0, filters>=1")
    tf.keras.utils.set_random_seed(seed)
    initializer_index = 0

    def convolution(x: tf.Tensor, filters: int, kernel: int, name: str) -> tf.Tensor:
        nonlocal initializer_index
        initializer_index += 1
        return tf.keras.layers.Conv2D(
            filters, kernel, padding="same", activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(
                seed=seed + initializer_index), name=name)(x)

    def block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        x = convolution(x, filters, 3, name + "_conv1")
        return convolution(x, filters, 3, name + "_conv2")

    image = tf.keras.Input((image_size, image_size, 3), name="image", dtype="float32")
    inputs = {"image": image}
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_to_unit_interval")(image)
    skips = []
    for level, multiplier in enumerate((1, 2, 4), start=1):
        x = block(x, base_filters * multiplier, f"encoder{level}")
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"pool{level}")(x)
    x = block(x, base_filters * 8, "bottleneck")
    x = tf.keras.layers.SpatialDropout2D(
        0.15, seed=seed + 100, name="bottleneck_dropout")(x)
    if metadata_dim:
        metadata = tf.keras.Input((metadata_dim,), name="metadata", dtype="float32")
        inputs["metadata"] = metadata
        encoded = tf.keras.layers.Dense(
            32, activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 101),
            name="metadata_dense32")(metadata)
        encoded = tf.keras.layers.Reshape((1, 1, 32), name="metadata_reshape")(encoded)
        encoded = tf.keras.layers.UpSampling2D(
            size=(image_size // 8, image_size // 8), interpolation="nearest",
            name="metadata_broadcast")(encoded)
        x = tf.keras.layers.Concatenate(name="bottleneck_multimodal_concat")([x, encoded])
    x = convolution(x, base_filters * 8, 1, "bottleneck_projection")
    for level, (skip, multiplier) in enumerate(zip(reversed(skips), (4, 2, 1)), start=1):
        x = tf.keras.layers.UpSampling2D(
            2, interpolation="bilinear", name=f"decoder{level}_upsample")(x)
        x = tf.keras.layers.Concatenate(name=f"decoder{level}_skip")([x, skip])
        x = block(x, base_filters * multiplier, f"decoder{level}")
    prediction = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", name="plexus_probability",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 102))(x)
    model = tf.keras.Model(inputs, prediction,
                           name="fusion_unet" if metadata_dim else "image_only_unet")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=BCEDiceLoss(), metrics=[soft_dice_all_frames])
    return model


def augment_pair(
    image: tf.Tensor, mask: tf.Tensor, random_seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Same small affine map for image/mask; image-only mild contrast/brightness.

    Rotation is uniform in +/-5 degrees; translations +/-4% of each axis.
    Image interpolation is bilinear and mask interpolation nearest. Intensities
    use contrast 0.95..1.05 and brightness +/-5 raw pixel units. No horizontal
    or vertical flips are used, preserving recorded laterality.
    """
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    random_values = tf.random.stateless_uniform((5,), seed=random_seed,
                                               minval=-1.0, maxval=1.0)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    h_float, w_float = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    angle = random_values[0] * (5.0 * np.pi / 180.0)
    cosine, sine = tf.cos(angle), tf.sin(angle)
    cx, cy = (w_float - 1.0) / 2.0, (h_float - 1.0) / 2.0
    tx, ty = random_values[1] * 0.04 * w_float, random_values[2] * 0.04 * h_float
    transform = tf.stack([
        cosine, -sine, cx - cosine * cx + sine * cy - tx,
        sine, cosine, cy - sine * cx - cosine * cy - ty, 0.0, 0.0,
    ])[None, :]

    def warp(values: tf.Tensor, interpolation: str) -> tf.Tensor:
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=values[None, ...], transforms=transform,
            output_shape=tf.stack([height, width]), interpolation=interpolation,
            fill_mode="CONSTANT", fill_value=0.0)[0]

    image, mask = warp(image, "BILINEAR"), warp(mask, "NEAREST")
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = (image - mean) * (1.0 + 0.05 * random_values[3]) + mean
    image = tf.clip_by_value(image + 5.0 * random_values[4], 0.0, 255.0)
    return image, mask


def _model_uses_metadata(model: tf.keras.Model) -> bool:
    return "metadata" in {tensor.name.split(":")[0] for tensor in model.inputs}


def make_dataset(
    images: np.ndarray, masks: np.ndarray, metadata: np.ndarray | None,
    *, batch_size: int = 8, training: bool = False, seed: int = SEED,
) -> tf.data.Dataset:
    """Bounded deterministic tf.data pipeline; augmentation only for training."""
    if len(images) == 0 or len(images) != len(masks) or batch_size < 1:
        raise ValueError("Dataset needs equal nonempty images/masks and positive batch")
    inputs = {"image": images}
    if metadata is not None:
        if len(metadata) != len(images):
            raise ValueError("Metadata row count differs from images")
        inputs["metadata"] = np.asarray(metadata, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, masks))
    if training:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.enumerate()

        def augment(index: tf.Tensor, pair: tuple) -> tuple:
            features, mask = pair
            features = dict(features)
            image, mask = augment_pair(features["image"], mask,
                                       tf.stack([tf.cast(seed, tf.int32),
                                                 tf.cast(index, tf.int32)]))
            features["image"] = image
            return features, mask

        dataset = dataset.map(augment, num_parallel_calls=1, deterministic=True)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def predict_batches(
    model: tf.keras.Model, images: np.ndarray, metadata: np.ndarray | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Infer raw images in supplied row order without dataset worker creation."""
    if len(images) == 0 or batch_size < 1:
        raise ValueError("Require nonempty images and positive batch size")
    use_metadata = _model_uses_metadata(model)
    if use_metadata and (metadata is None or len(metadata) != len(images)):
        raise ValueError("This model requires aligned encoded metadata")
    outputs = []
    for left in range(0, len(images), batch_size):
        features = {"image": np.asarray(images[left:left + batch_size], dtype=np.float32)}
        if use_metadata:
            features["metadata"] = np.asarray(metadata[left:left + batch_size], dtype=np.float32)
        outputs.append(np.asarray(model(features, training=False), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


class _TrainingEvidence(tf.keras.callbacks.Callback):
    def __init__(self, history_path: str | Path | None = None):
        super().__init__()
        self.rows: list[dict[str, float]] = []
        self.history_path = Path(history_path) if history_path else None

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        self.rows.append({"epoch": int(epoch + 1),
                          "epoch_seconds": time.perf_counter() - self.started,
                          **{key: float(value) for key, value in (logs or {}).items()}})
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(self.rows, indent=2, allow_nan=False))


@dataclass
class TrainingResult:
    model: tf.keras.Model
    history: dict[str, list[float]]
    val_probabilities: np.ndarray
    record: dict[str, Any]


def _validate_arrays(images: np.ndarray, masks: np.ndarray, name: str) -> None:
    if (images.ndim != 4 or images.shape[-1] != 3 or masks.ndim != 4
            or masks.shape[-1] != 1 or images.shape[:3] != masks.shape[:3]):
        raise ValueError(f"{name}: expected RGB[N,H,W,3] and mask[N,H,W,1]")
    if not len(images) or not np.isfinite(images).all() or not np.isfinite(masks).all():
        raise ValueError(f"{name}: empty or nonfinite arrays")
    if images.min() < 0 or images.max() > 255:
        raise ValueError(f"{name}: raw image values must be in [0,255]")
    if not np.isin(masks, [0, 1]).all():
        raise ValueError(f"{name}: reference masks must be binary [0,1]")


def train_unet(
    name: str,
    train_images: np.ndarray, train_masks: np.ndarray,
    train_metadata: np.ndarray | None,
    val_images: np.ndarray, val_masks: np.ndarray,
    val_metadata: np.ndarray | None,
    *, seed: int = SEED, batch_size: int = 8, max_epochs: int = 30,
    patience: int = 6, base_filters: int = 16, learning_rate: float = 1e-3,
    history_path: str | Path | None = None, checkpoint_path: str | Path | None = None,
    verbose: int = 2,
) -> TrainingResult:
    """Train with train/validation only; checkpoint minimizes validation loss.

    Pass metadata=None for image-only baseline. No test examples, mask-based
    frame selection, class weighting, threshold search, or outcome optimization
    is performed. Primary Dice is evaluated externally per positive patient at
    the prespecified threshold 0.5, with negative frames reported separately.
    """
    if name not in ("image_only", "fusion"):
        raise ValueError("name must be image_only or fusion")
    if not 1 <= max_epochs <= 35 or patience < 1:
        raise ValueError("Require epochs1..35 and patience>=1")
    _validate_arrays(train_images, train_masks, "train")
    _validate_arrays(val_images, val_masks, "validation")
    if train_images.shape[1:] != val_images.shape[1:]:
        raise ValueError("Training and validation image dimensions differ")
    if (name == "fusion") != (train_metadata is not None):
        raise ValueError("Only fusion must receive metadata")
    metadata_dim = 0
    if train_metadata is not None:
        if (val_metadata is None or np.asarray(train_metadata).ndim != 2
                or np.asarray(val_metadata).ndim != 2
                or train_metadata.shape[1] != val_metadata.shape[1]
                or not np.isfinite(train_metadata).all()
                or not np.isfinite(val_metadata).all()):
            raise ValueError("Require finite compatible encoded metadata")
        metadata_dim = int(train_metadata.shape[1])
    elif val_metadata is not None:
        raise ValueError("Image-only baseline must not receive validation metadata")
    model = build_unet(metadata_dim, train_images.shape[1], base_filters, seed,
                       learning_rate)
    evidence = _TrainingEvidence(history_path)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=patience,
        min_delta=0.0, restore_best_weights=True)
    callbacks = [evidence, early_stopping, tf.keras.callbacks.TerminateOnNaN()]
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", mode="min", save_best_only=True))
    train_data = make_dataset(train_images, train_masks, train_metadata,
                              batch_size=batch_size, training=True, seed=seed)
    validation_data = make_dataset(val_images, val_masks, val_metadata,
                                   batch_size=batch_size, training=False, seed=seed)
    started = time.perf_counter()
    trained = model.fit(train_data, validation_data=validation_data, epochs=max_epochs,
                        callbacks=callbacks, verbose=verbose, shuffle=False)
    elapsed = time.perf_counter() - started
    history = {key: [float(value) for value in values]
               for key, values in trained.history.items()}
    history["epoch_seconds"] = [row["epoch_seconds"] for row in evidence.rows]
    if any(not np.isfinite(values).all() for values in history.values()):
        raise RuntimeError("Training produced non-finite history; no valid model is declared")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    val_probabilities = predict_batches(model, val_images, val_metadata, batch_size)
    record = {
        "model_name": name, "seed": seed, "batch_size": batch_size,
        "max_epochs": max_epochs, "epochs_run": len(history["loss"]),
        "patience": patience, "checkpoint_monitor": "val_loss",
        "checkpoint_mode": "min", "restored_best_weights": True,
        "best_epoch": best_epoch,
        "best_val_loss": history["val_loss"][best_epoch - 1],
        "training_seconds": elapsed, "n_train": len(train_images),
        "n_validation": len(val_images), "image_size": train_images.shape[1],
        "image_channels": 3, "image_range": [0, 255],
        "image_normalization": "divide_by_255_inside_model",
        "metadata_dim": metadata_dim, "base_filters": base_filters,
        "encoder_filters": [base_filters * n for n in [1, 2, 4]],
        "bottleneck_filters": base_filters * 8,
        "metadata_dense_units": 32 if metadata_dim else 0,
        "bottleneck_spatial_dropout": 0.15,
        "total_parameters": model.count_params(),
        "trainable_parameters": int(sum(np.prod(x.shape) for x in model.trainable_weights)),
        "learning_rate": learning_rate, "optimizer": "Adam",
        "loss": "0.5*mean_pixel_BCE + 0.5*(1-per_frame_soft_Dice_smooth1)",
        "training_metric_note": "soft_dice_all_frames is diagnostic; not primary positive-patient Dice",
        "augmentation_train_only": {
            "rotation_degrees": [-5, 5], "translation_fraction": [-0.04, 0.04],
            "contrast": [0.95, 1.05], "brightness_raw_units": [-5, 5],
            "mask_interpolation": "nearest", "horizontal_flip": False,
            "vertical_flip": False,
        },
        "decision_threshold_prespecified": 0.5,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
    }
    return TrainingResult(model, history, val_probabilities, record)


def save_and_verify(
    model: tf.keras.Model, path: str | Path, sample_images: np.ndarray,
    sample_metadata: np.ndarray | None = None,
) -> dict[str, Any]:
    """Save native .keras, reopen compile=False/safe_mode, compare predictions."""
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError("Destination must end with .keras")
    expected = predict_batches(model, sample_images, sample_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    reopened = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    actual = predict_batches(reopened, sample_images, sample_metadata)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    if actual.min() < 0 or actual.max() > 1 or not np.isfinite(actual).all():
        raise RuntimeError("Saved inference model produced invalid probabilities")
    return {
        "saved_path": str(path), "bytes": path.stat().st_size,
        "safe_mode_load_passed": True, "prediction_equivalence_passed": True,
        "max_absolute_difference": float(np.max(np.abs(expected - actual))),
        "sample_count": len(sample_images), "parameters": reopened.count_params(),
    }
