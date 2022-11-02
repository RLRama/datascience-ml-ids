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

y = df.diagnosis
list = ['id','diagnosis']
x = df.drop(list,axis=1)
df1 = df.describe(include='all').fillna("").astype("str")
st.write(df.describe)

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

st.subheader('Gráficos de relación de pares')
st.markdown('Grafican relaciones de pares, de forma que cada variable es mostrada junto a la otra (**0 [benigno]** y **1 [maligno]**)')
st.caption('Medidas de radio de tumor (media, error estándar y peor caso)')
radius = df[['radius_mean','radius_se','radius_worst','diagnosis']]
fig4 = sns.pairplot(radius, hue='diagnosis',palette="husl", markers=["o", "s"],size=4)
st.pyplot(fig4)

st.caption('Medidas de textura de tumor (media, error estándar y peor caso)')
texture = df[['texture_mean','texture_se','texture_worst','diagnosis']]
fig5 = sns.pairplot(texture, hue='diagnosis', palette="Blues_d",size=4, kind="reg")
st.pyplot(fig5)

st.caption('Medidas de perímetro de tumor (media, error estándar y peor caso)')
perimeter = df[['perimeter_mean','perimeter_se','perimeter_worst','diagnosis']]
fig6 = sns.pairplot(perimeter, hue='diagnosis', size = 4, kind="reg")
st.pyplot(fig6)

st.caption('Medidas de área de tumor (media, error estándar y peor caso)')
area = df[['area_mean','area_se','area_worst','diagnosis']]
fig7 = sns.pairplot(area, hue='diagnosis', size =4)
st.pyplot(fig7)

st.caption('Medidas de lisura de tumor (media, error estándar y peor caso)')
smoothness = df[['smoothness_mean','smoothness_se','smoothness_worst','diagnosis']]
fig8 = sns.pairplot(smoothness, hue='diagnosis')
st.pyplot(fig8)

st.caption('Medidas de compacidad de tumor (media, error estándar y peor caso)')
compactness = df[['compactness_mean','compactness_se','compactness_worst','diagnosis']]
fig9 = sns.pairplot(compactness, hue='diagnosis')
st.pyplot(fig9)

st.caption('Medidas de concavidad de tumor (media, error estándar y peor caso)')
concavity = df[['concavity_mean','concavity_se','concavity_worst','diagnosis']]
fig10 = sns.pairplot(concavity, hue='diagnosis')
st.pyplot(fig10)

st.caption('Medidas de puntos cóncavos de tumor (media, error estándar y peor caso)')
concave_points = df[['concave points_mean','concave points_se','concave points_worst','diagnosis']]
fig11 = sns.pairplot(concave_points, hue='diagnosis')
st.pyplot(fig11)

st.caption('Medidas de simetría de tumor (media, error estándar y peor caso)')
symmetry = df[['symmetry_mean','symmetry_se','symmetry_worst','diagnosis']]
fig12 = sns.pairplot(symmetry, hue='diagnosis')
st.pyplot(fig12)

st.caption('Medidas de dimensión fractal de tumor (media, error estándar y peor caso)')
fractal_dimension = df[['fractal_dimension_mean','fractal_dimension_se','fractal_dimension_worst','diagnosis']]
fig13 = sns.pairplot(fractal_dimension, hue='diagnosis')
st.pyplot(fig13)

with st.expander("Ver explicación"):
    st.markdown(
        """
        Un gráfico de relación de pares es una visualización de datos que imprime
        relaciones de pares entre las variables de un conjunto de datos. Este permite entender
        mejor las relaciones visualmente. Cada variable se imprime en las filas y columnas, mostrando
        la relación entre variables.
        """
    )