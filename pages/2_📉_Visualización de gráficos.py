import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Vista de gráficos",
    page_icon="🤖",
)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/abacus_1f9ee.png",
    width=100
)

@st.cache
def load_data():
    df = pd.read_csv('data.csv')
    df = df.dropna(axis=1)
    df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
    return df

df = load_data()

fig3 = sns.pairplot(df, hue="diagnosis")
st.pyplot(fig3)