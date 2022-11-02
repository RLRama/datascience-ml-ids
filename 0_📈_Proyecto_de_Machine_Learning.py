import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()

@st.cache
def load_data():
    df = pd.read_csv('data.csv')
    df = df.dropna(axis=1)
    df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
    return df

st.set_page_config(
    page_title="Predicción con ML",
    page_icon="🤖",
)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/apple/325/desktop-computer_1f5a5-fe0f.png",
    width=100,
)
st.title("Aplicación web de clasificación binaria")
st.sidebar.title("Parámetros de clasificación binaria")
st.markdown("Detección de tipo de tumor (benigno o maligno)")
st.markdown("**1 (uno)** corresponde a **detecciones malignas** y **0 (cero)** a **detecciones benignas**")
st.sidebar.markdown("Ingeniería en Software - The RAMBros")

df = load_data()

st.header("Estructura de datos fuente")
st.dataframe(df)

fig = plt.figure(figsize=(8,6))
plt.title("Diagnósticos tomados")
plt.xticks(fontsize=12)
sns.countplot(x="diagnosis",data=df)
st.pyplot(fig)

st.markdown("**Correlación** entre las columnas")
st.dataframe(df.corr())

st.markdown("**Mapa de calor** para visualizar la **correlación** entre las columnas")
fig2 = plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.gcf().set_size_inches(40, 20)
st.pyplot(fig2)