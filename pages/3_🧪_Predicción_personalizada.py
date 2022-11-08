import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

labelencoder_Y = LabelEncoder()

st.set_page_config(
    page_title="Predicción en tiempo real",
    page_icon="🤖",
)

def load_data():
    df = pd.read_csv('data.csv')
    df = df.dropna(axis=1)
    df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
    return df

df = load_data()

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/petri-dish_1f9eb.png",
    width=100
)
st.title('Predicción con nueva muestra personalizada')
st.markdown(
    """
    - En este caso, usaremos el modelo de **bosque aleatorio**
    - Cada muestra necesita 30 variables para poder realizar la predicción
    """
)

X = df.iloc[:, 2:32].values
Y = df.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)

st.header("Creación de nueva muestra personalizada")
st.markdown(
    """
    ### Sección de medidas promedio (mean)
    """
)
sp_radius_mean = st.slider('radius_mean (Radio de tejido)', min_value=6.981, max_value=28.11, value=14.127291739894552)
sp_texture_mean = st.slider('texture_mean (Textura de tejido)', min_value=9.71, max_value=39.28, value=19.289648506151142)
sp_perimeter_mean = st.slider('perimeter_mean (Perímetro de tejido)', min_value=43.79, max_value=188.5, value=91.96903339191564)
sp_area_mean = st.slider('area_mean (Área de tejido)', min_value=143.5, max_value=2501.0, value=654.8891036906855)
sp_smoothness_mean = st.slider('smoothness_mean (Lisura de tejido)', min_value=0.05263, max_value=0.1634, value=0.0963602811950791)
sp_compactness_mean = st.slider('compactness_mean (Compacidad de tejido)', min_value=0.01938, max_value=0.3454, value=0.10434098418277679)
sp_concavity_mean = st.slider('concavity_mean (Concavidad de tejido)', min_value=0.0, max_value=0.4268, value=0.0887993158172232)
sp_concave_points_mean = st.slider('concave_points_mean (Puntos cóncavos de tejido)', min_value=0.0, max_value=0.2012, value=0.04891914586994728)
sp_symmetry_mean = st.slider('symmetry_mean (Simetría de tejido)', min_value=0.106, max_value=0.304, value=0.18116186291739894)
sp_fractal_dimension_mean = st.slider('fractal_dimension_mean (Dimensión fractal de tejido)', min_value=0.04996, max_value=0.09744, value=0.06279760984182776)

st.markdown(
    """
    ### Sección de medidas de error estándar (se)
    """
)
sp_radius_se = st.slider('radius_se (Radio de tejido)', min_value=0.1115, max_value=2.873, value=0.40517205623901575)
sp_texture_se = st.slider('texture_se (Textura de tejido)', min_value=0.3602, max_value=4.885, value=1.2168534270650264)
sp_perimeter_se = st.slider('perimeter_se (Perímetro de tejido)', min_value=0.757, max_value=21.98, value=2.8660592267135327)
sp_area_se = st.slider('area_se (Área de tejido)', min_value=6.802, max_value=542.2, value=40.337079086116)
sp_smoothness_se = st.slider('smoothness_se (Lisura de tejido)', min_value=0.001713, max_value=0.03113, value=0.007040978910369069)
sp_compactness_se = st.slider('compactness_se (Compacidad de tejido)', min_value=0.002252, max_value=0.1354, value=0.025478138840070295)
sp_concavity_se = st.slider('concavity_se (Concavidad de tejido)', min_value=0.0, max_value=0.396, value=0.03189371634446397)
sp_concave_points_se = st.slider('concave_points_se (Puntos cóncavos de tejido)', min_value=0.0, max_value=0.05279, value=0.011796137082601054)
sp_symmetry_se = st.slider('symmetry_se (Simetría de tejido)', min_value=0.007882, max_value=0.07895, value=0.02054229876977153)
sp_fractal_dimension_se = st.slider('fractal_dimension_se (Dimensión fractal de tejido)', min_value=0.0008948, max_value=0.02984, value=0.0037949038664323374)

st.markdown(
    """
    ### Sección de medidas de peor caso (worst)
    """
)
sp_radius_worst = st.slider('radius_worst (Radio de tejido)', min_value=7.93, max_value=36.04, value=16.269189806678387)
sp_texture_worst = st.slider('texture_worst (Textura de tejido)', min_value=12.02, max_value=49.54, value=25.677223198594024)
sp_perimeter_worst = st.slider('perimeter_worst (Perímetro de tejido)', min_value=50.41, max_value=251.2, value=107.26121265377857)
sp_area_worst = st.slider('area_worst (Área de tejido)', min_value=185.2, max_value=4254.0, value=880.5831282952548)
sp_smoothness_worst = st.slider('smoothness_worst (Lisura de tejido)', min_value=0.07117, max_value=0.2226, value=0.13236859402460457)
sp_compactness_worst = st.slider('compactness_worst (Compacidad de tejido)', min_value=0.02729, max_value=1.058, value=0.25426504393673116)
sp_concavity_worst = st.slider('concavity_worst (Concavidad de tejido)', min_value=0.0, max_value=1.252, value=0.27218848330404216)
sp_concave_points_worst = st.slider('concave_points_worst (Puntos cóncavos de tejido)', min_value=0.0, max_value=0.291, value=0.11460622319859401)
sp_symmetry_worst = st.slider('symmetry_worst (Simetría de tejido)', min_value=0.1565, max_value=0.6638, value=0.2900755711775044)
sp_fractal_dimension_worst = st.slider('fractal_dimension_worst (Dimensión fractal de tejido)', min_value=0.05504, max_value=0.2075, value=0.0839458172231986)

prediction = forest.predict([[sp_radius_mean,sp_texture_mean,sp_perimeter_mean,sp_area_mean,sp_smoothness_mean,sp_compactness_mean,sp_concavity_mean,sp_concave_points_mean,sp_symmetry_mean,sp_fractal_dimension_mean,sp_radius_se,sp_texture_se,sp_perimeter_se,sp_area_se,sp_smoothness_se,sp_compactness_se,sp_concavity_se,sp_concave_points_se,sp_symmetry_se,sp_fractal_dimension_se,sp_radius_worst,sp_texture_worst,sp_perimeter_worst,sp_area_worst,sp_smoothness_worst,sp_compactness_worst,sp_concavity_worst,sp_concave_points_worst,sp_symmetry_worst,sp_fractal_dimension_worst]])[0]

if st.button('Generar predicción'):
    if prediction == 1:
        st.write("La predicción para la nueva muestra de tejido de tumor es:")
        st.error("TUMOR MALIGNO")
    elif prediction == 0:
        st.write("La predicción para la nueva muestra de tejido de tumor es:")
        st.error("TUMOR BENIGNO")
else:
    st.warning("Esperando presentación de muestra...")