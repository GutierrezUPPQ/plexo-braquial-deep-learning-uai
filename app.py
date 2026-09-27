"""Demostración académica de segmentación multimodal del plexo braquial."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from anesthesia_model import TabularPreprocessor, predict_batches

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'bundle'
st.set_page_config(page_title='Plexo · Ecografía multimodal',page_icon='◉',layout='wide')
st.markdown('''<style>
.stApp {background:#f6f9fa;color:#153440}
.block-container {max-width:1280px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-.03em}
[data-testid="stMetric"] {background:white;border:1px solid #d9e5e9;padding:14px;border-radius:12px}
.stButton button {background:#006c75!important;color:white!important;border:0;padding:.7rem 1.4rem;font-weight:650}
[data-testid="stMetric"] * {color:#153440!important}
[data-testid="stMetricValue"] {font-size:clamp(1rem,2.2vw,1.9rem)!important}
[data-testid="stMetricLabel"] {font-size:.85rem!important}
.caption-label {font-size:12px;letter-spacing:.16em;color:#437381;font-weight:700}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model=tf.keras.models.load_model(BUNDLE/'fusion.keras',compile=False)
    pre=TabularPreprocessor.load(BUNDLE/'preprocessor.json')
    manifest=pd.read_csv(BUNDLE/'test_examples.csv')
    images=np.load(BUNDLE/'test_images.npy')
    masks=np.load(BUNDLE/'test_masks.npy')
    encoded=np.load(BUNDLE/'test_metadata.npy')
    metrics=json.loads((BUNDLE/'results_summary.json').read_text())
    assert len(manifest)==len(images)==len(masks)==len(encoded)
    assert set(manifest.split)=={'test'}
    np.testing.assert_allclose(pre.transform(manifest),encoded,atol=1e-6)
    return model,pre,manifest,images,masks,metrics

model,pre,manifest,images,masks,metrics=load_assets()
st.markdown('<div class="caption-label">DEEP LEARNING · UAI · TAREA 2</div>',unsafe_allow_html=True)
st.title('El plexo braquial, píxel a píxel')
st.markdown('**Ecografía + datos tabulares** · U-Net multimodal entrenada con Regional-US')
st.caption('Demostración educativa con imágenes públicas de prueba. No validada para decisiones clínicas ni para guiar una aguja.')

left,right=st.columns([1.1,2.5],gap='large')
with left:
    st.subheader('1. Elegir una ecografía de test')
    labels=[f"Ejemplo {i+1:02d} · video {r.patient_id} · frame {int(r.frame_idx)}" for i,r in manifest.iterrows()]
    choice=st.selectbox('Casos no usados para entrenar',range(len(labels)),format_func=lambda i:labels[i])
    row=manifest.iloc[[choice]]
    st.caption('Todos los fotogramas del mismo paciente se mantuvieron en una sola partición.')
    st.subheader('2. Contexto asociado a la imagen')
    r=row.iloc[0]
    c1,c2=st.columns(2)
    c1.metric('Edad',f'{float(r.age):.0f} años');c2.metric('IMC',f'{float(r.BMI):.1f}')
    c1.metric('Talla',f'{float(r.height):.0f} cm');c2.metric('Sexo en fuente',str(r.gender))
    st.write('Lateralidad:',str(r.left_right),'· Ganancia:',str(r.gain))
    st.caption('Entradas reales del dataset. El plano anatómico, las máscaras, el número del video y las métricas no entran en el modelo.')
    run=st.button('Segmentar plexo',type='primary',use_container_width=True)

with right:
    st.subheader('3. Imagen y predicción')
    st.image(images[choice],caption='Ecografía de prueba · entrada de la red a 128 × 128 píxeles',width=440)

if run:
    with st.spinner('Ejecutando el modelo multimodal guardado…'):
        meta=pre.transform(row)
        probability=predict_batches(model,images[choice:choice+1],meta,batch_size=1)[0,...,0]
        binary=probability>=.5
        st.session_state['inference']={'choice':choice,'probability':probability,'binary':binary}

result=st.session_state.get('inference')
if result is not None and result['choice']==choice:
    probability=result['probability'];binary=result['binary'];gt=masks[choice,...,0]>0
    base=images[choice].astype(float)
    overlay=base.copy();overlay[binary]=.45*overlay[binary]+.55*np.array([255,157,32])
    truth=base.copy();truth[gt]=.45*truth[gt]+.55*np.array([25,181,113])
    st.divider()
    st.subheader('Inferencia realizada')
    a,b,c=st.columns(3)
    a.image(images[choice],caption='Entrada · ecografía original redimensionada',use_container_width=True)
    b.image(overlay.astype('uint8'),caption='Naranja · predicción de la U-Net multimodal',use_container_width=True)
    c.image(truth.astype('uint8'),caption='Verde · referencia del dataset, sólo para evaluar',use_container_width=True)
    p,q,s=st.columns(3)
    if gt.any():
        intersection=(gt&binary).sum();dice=2*intersection/(gt.sum()+binary.sum());iou=intersection/(gt|binary).sum()
        p.metric('Dice · este fotograma',f'{dice:.3f}');q.metric('IoU · este fotograma',f'{iou:.3f}')
    else:
        p.metric('Referencia','Sin plexo marcado');q.metric('Área falsa positiva',f'{100*binary.mean():.2f}%')
    s.metric('Umbral fijado antes de test','0,50')
    st.caption('Las métricas de un ejemplo no sustituyen la evaluación de todos los pacientes de prueba. Las anotaciones combinan seguimiento, contornos activos y revisión experta.')
else:
    st.info('Pulsa «Segmentar plexo» para ejecutar una inferencia real con la imagen y sus datos tabulares.')

with st.expander('Resultados del experimento y funcionamiento de la fusión'):
    st.write('La rama de imagen obtiene mapas espaciales mediante convoluciones. La rama tabular transforma seis variables con parámetros aprendidos sólo en entrenamiento. Su vector se expande espacialmente y se concatena en el cuello de botella; el decodificador U-Net produce una máscara binaria.')
    st.dataframe(pd.DataFrame(metrics['test_metrics']),hide_index=True,use_container_width=True)
    st.write('La métrica principal es Dice en fotogramas de referencia positiva: primero se promedia dentro de cada paciente y luego entre pacientes. Los fotogramas sin referencia positiva se evalúan por separado.')
    st.write(metrics['conclusion'])
    st.caption('Modelo cargado: fusion.keras · SHA-256 '+hashlib.sha256((BUNDLE/'fusion.keras').read_bytes()).hexdigest()[:16])

st.divider()
st.caption('Claudio Gutiérrez · Trabajo académico individual con apoyo declarado de OpenAI Codex · Regional-US, Tyagi et al., IROS 2024 · Datos para uso no comercial.')
st.markdown('[Fuente de los datos](https://github.com/Regional-US/brachial_plexus) · [Artículo original](https://arxiv.org/abs/2308.03717)')
