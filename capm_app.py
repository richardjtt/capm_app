import streamlit as st
import pandas as pd

# Cargar datos desde Damodaran
url = 'https://www.stern.nyu.edu/~adamodar/pc/datasets/betas.xls'
df = pd.read_excel(url, sheet_name=1, header=None)
df = df.drop(index=range(0, 9)).reset_index(drop=True)
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# URL oficial de FRED (10-Year Treasury Constant Maturity Rate)
url_rf = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"

# Leer CSV desde la web
df_rf = pd.read_csv(url_rf)

# Normalizar nombres de columnas
df_rf.columns = df_rf.columns.str.strip().str.upper()

# Detectar columna de fecha
fecha_col = next((col for col in df_rf.columns if "DATE" in col), df_rf.columns[0])

# Convertir fecha
df_rf[fecha_col] = pd.to_datetime(df_rf[fecha_col], errors="coerce")

# Convertir tasa a numérico
df_rf["DGS10"] = pd.to_numeric(df_rf["DGS10"], errors="coerce")

# Limpiar y ordenar
df_rf = df_rf.dropna(subset=["DGS10"]).sort_values(fecha_col)

# Extraer último valor
rf = df_rf.iloc[-1]["DGS10"]
rf = rf / 100

# URL del archivo Excel de Country Risk Premiums
url = 'https://www.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx'

# Cargar la hoja "ERPs by Country" sin encabezado
erp_data = pd.read_excel(url, sheet_name='ERPs by country', header=None)

# Eliminar las primeras 6 filas (introducción o texto previo a la tabla)
erp_data = erp_data.drop(index=range(0, 6)).reset_index(drop=True)

# Asignar la fila 7 como el nuevo encabezado
erp_data.columns = erp_data.iloc[0]
erp_data = erp_data.drop(index=0).reset_index(drop=True)

# Limpiar nombres de columnas
erp_data.columns = erp_data.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

# Definir país y columna
pais = "Peru"
col_erp = "Total Equity Risk Premium"

# Extraer ERP para Perú
erp_peru = erp_data[erp_data["Country"] == pais][col_erp].values[0]
erp_peru = float(erp_peru)



# Extraer beta desapalancado corregido por caja para Metals & Mining
industria = "Metals & Mining"
col_beta = "Unlevered beta corrected for cash"
beta_desapalancado = df[df["Industry Name"] == industria][col_beta].values[0]
beta_desapalancado = float(beta_desapalancado)


st.title("Modelo CAPM - Conservador | Desarrollado por Richard Jammer"")
st.markdown("Herramienta de análisis de Beta y Costo de Capital para empresas. By **Richard Jammer**.")


st.markdown("Este modelo calcula el **Costo de Capital (Ke)** sin deuda usando la fórmula del **CAPM**:")
st.latex(r"Ke = R_f + \beta \times ERP")


# Inputs
rf = st.number_input("Tasa libre de riesgo (Rf) - FRED 10Y", value=rf, step=0.0001, format="%.4f", disabled=True)
erp = st.number_input("Total Equity Risk Premium (ERP) - Perú", value=erp_peru, step=0.0001, format="%.4f", disabled=True)
beta = st.number_input("Beta (β) de Metals & Mining (Damodaran) ", value=beta_desapalancado, step=0.01, disabled=True)

# Convertir a decimales
rf_dec = rf 
erp_dec = erp

# Calcular Ke
ke = rf_dec + beta * erp_dec

# Mostrar resultado
st.subheader("Resultado")
st.write(f"**Ke = {ke*100:.2f}%**")

# Ejemplo con varios betas (opcional)
st.subheader("Sensibilidad del Ke con distintos β")
betas = [0.5, 1.0, 1.5, 2.0]
for b in betas:
    ke_b = rf_dec + b * erp_dec
    st.write(f"β = {b:.1f} → Ke = {ke_b*100:.2f}%")
    

import streamlit as st
from PIL import Image

# Cargar la imagen desde la carpeta "imagen"
img = Image.open("imagen/VPN.png")

# Mostrar la imagen
st.image(img, caption="VPN Logo", use_container_width=True)




from sentence_transformers import SentenceTransformer, util

st.subheader("🧠 Sugerencia inteligente de industria Damodaran")

# Entrada libre del usuario
descripcion_usuario = st.text_input("Detalla tu industria (se sugiere traducir a inglés, ej. 'company dedicated to the manufacture of toilet soaps')")

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










