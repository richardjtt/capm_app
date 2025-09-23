import streamlit as st
import pandas as pd

# Cargar datos desde Damodaran
url = 'https://www.stern.nyu.edu/~adamodar/pc/datasets/betas.xls'
df = pd.read_excel(url, sheet_name=1, header=None)
df = df.drop(index=range(0, 9)).reset_index(drop=True)
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

from sentence_transformers import SentenceTransformer, util

st.subheader("🧠 Sugerencia inteligente de industria Damodaran")

# Entrada libre del usuario
descripcion_usuario = st.text_input("Detalla tu industria (en inglés, ej. 'company dedicated to the manufacture of toilet soaps')")

# Cargar modelo SBERT solo una vez
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo = cargar_modelo()

# Lista de industrias Damodaran
industrias = df["Industry Name"].dropna().unique().tolist()
embeddings_industrias = modelo.encode(industrias, convert_to_tensor=True)

# Procesar entrada del usuario
if descripcion_usuario:
    embedding_input = modelo.encode(descripcion_usuario, convert_to_tensor=True)
    similitudes = util.cos_sim(embedding_input, embeddings_industrias)[0]
    top_indices = similitudes.argsort(descending=True)[:3]

    st.subheader("🎯 Industrias sugeridas por similitud semántica:")
    for idx in top_indices:
        industria = industrias[idx]
        beta = df[df["Industry Name"] == industria]["Unlevered beta corrected for cash"].values[0]
        st.write(f"- **{industria}** → β = {float(beta):.4f}")

