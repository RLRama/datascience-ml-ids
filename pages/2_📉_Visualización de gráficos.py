import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import cufflinks as cf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

labelencoder_Y = LabelEncoder()

st.set_page_config(
    page_title="Vista de gráficos",
    page_icon="🤖",
)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/abacus_1f9ee.png",
    width=100
)
st.title('Vista de gráficos detallada')

@st.cache
def load_data():
    df = pd.read_csv('data.csv')
    df = df.dropna(axis=1)
    df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
    return df

df = load_data()

st.subheader('Matriz de correlación')
st.markdown('Ayuda a comprender la correlación entre las variables')
fig3 = plt.figure(figsize=(20,20))
df_corr = df.corr()
mask = np.triu(df_corr, k=1)
sns.heatmap(df_corr, cmap= 'YlGnBu', annot=True, fmt=".2f", mask=mask)
st.pyplot(fig3)
with st.expander("Ver explicación"):
    st.markdown(
        """
        Una matriz de correlación es simplemente una tabla que muestra los
        coeficientes de correlación entre diferentes variables. Es una gran
        herramienta para sumarizar grandes conjuntos de datos y visualizar patrones
        en los datos dados.
        """
    )

fig4 = plt.figure(figsize=(8,6))
plt.title("Diagnósticos tomados")
plt.xticks(fontsize=12)
sns.pie(x="diagnosis",data=df)
st.pyplot(fig4)