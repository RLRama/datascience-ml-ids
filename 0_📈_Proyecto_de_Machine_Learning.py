import streamlit as st
import pandas as pd
import numpy as np
import os
import sklearn as sk
import matplotlib as mplt
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
rcParams["figure.figsize"] = (10, 10)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

def main():
    st.set_page_config(
        page_title="The RAMBros",
        page_icon="🤖",
    )

    st.title("Aplicación web de clasificación binaria")
    st.sidebar.title("Parámetros de clasificación binaria")
    st.markdown("Detección de tipo de tumor (benigno o maligno)")
    st.sidebar.markdown("Ingeniería en Software - The RAMBros")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('data.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.diagnosis
        x = df.drop(columns =['diagnosis'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'Matriz de confusión' in metrics_list:
            st.subheader("Matriz de confusión") 
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
        
        if 'Curva de característica operativa de receptor' in metrics_list:
            st.subheader("Curva de característica operativa de receptor") 
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Curva de precisión-exhaustividad' in metrics_list:
            st.subheader("Curva de precisión-exhaustividad")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    class_names = ['b', 'm']

    x_train, x_test, y_train, y_test = split(df)
    
    st.sidebar.subheader("Elegir método de clasificación")
    classifier = st.sidebar.selectbox("Clasificador", ("Máquina de vector soporte", "Regresión logística", "Bosque aleatorio", "Perceptrón"))

    if classifier == 'Máquina de vector soporte':
        st.sidebar.subheader("Hiperparámetros de modelo")
        C = st.sidebar.number_input("C (Parámetro de regularización)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel",("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Coeficiente de kernel)", ("scale", "auto"), key = 'gamma')
        metrics = st.sidebar.multiselect("Métricas a imprimir",('Matriz de confusión', 'Curva de característica operativa de receptor', 'Curva de precisión-exhaustividad'))

        if st.sidebar.button("Clasificar", key='classify'):
            st.subheader("Resultados de máquina de vector soporte")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Exactitud: ", accuracy.round(2))
            st.write("Precisión: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Exhaustividad: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Regresión logística':
        st.sidebar.subheader("Hiperparámetros de modelo")
        C = st.sidebar.number_input("C (Parámetro de regularización)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Máximo número de iteraciones", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("Métricas a imprimir",('Matriz de confusión', 'Curva de característica operativa de receptor', 'Curva de precisión-exhaustividad'))

        if st.sidebar.button("Clasificar", key='classify'):
            st.subheader("Resultados de regresión logística")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Exactitud: ", accuracy.round(2))
            st.write("Precisión: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Exhaustividad: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Bosque aleatorio':
        st.sidebar.subheader("Hiperparámetros de modelo")
        n_estimators  = st.sidebar.number_input("Número de árboles en el bosque", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Profundidad máxima de árbol", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Muestras de bootstrap al generar árboles", ('True','False'), key='bootstrap')
        metrics = st.sidebar.multiselect("Métricas a imprimir",('Matriz de confusión', 'Curva de característica operativa de receptor', 'Curva de precisión-exhaustividad'))

        if st.sidebar.button("Clasificar", key='classify'):
            st.subheader("Resultados de bosque aleatorio")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Exactitud: ", accuracy.round(2))
            st.write("Precisión: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Exhaustividad: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Mostrar datos crudos", False):
        st.subheader("Conjunto de datos de diagnósticos de cáncer de mama (clasificación)")
        st.write(df)

if __name__ == '__main__':
    main()