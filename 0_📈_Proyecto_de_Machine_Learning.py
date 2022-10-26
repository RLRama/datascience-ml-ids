import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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
st.sidebar.markdown("Ingeniería en Software - The RAMBros")

df = pd.read_csv('data.csv')
df = df.dropna(axis=1)

sns.set_style("darkgrid",{"axes.facecolor": ".9"})
sns.countplot(df['diagnosis'], label="Cantidad")