# Plexo braquial · Tarea 2

Descomprimir el ZIP. Requiere Python 3.11 o 3.12.

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

La app carga bundle/fusion.keras y sus datos reales de test. No necesita entrenar ni descargar datos para demostrar la inferencia. En Windows activar .venv\Scripts\activate.

Fuentes: Regional-US https://github.com/Regional-US/brachial_plexus y Tyagi et al. https://arxiv.org/abs/2308.03717 . El README del dataset declara uso no comercial; no se atribuye otra licencia. Ejercicio educativo, sin validación clínica. Código desarrollado con apoyo declarado de OpenAI Codex.
