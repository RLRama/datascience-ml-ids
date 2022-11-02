import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def models(X_train,Y_train):
  
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)

  print('[0] Precisión de regresión logística:', log.score(X_train, Y_train))
  print('[1] Precisión de k-vecinos más cercanos:', knn.score(X_train, Y_train))
  print('[2] Precisión de máquina de vector soporte (rbf):', svc_lin.score(X_train, Y_train))
  print('[3] Precisión de máquina de vector soporte (lineal):', svc_rbf.score(X_train, Y_train))
  print('[4] Precisión de Naive Bayes (Gaussiana):', gauss.score(X_train, Y_train))
  print('[5] Precisión de árbol de decisión:', tree.score(X_train, Y_train))
  print('[6] Precisión de bosque aleatorio:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest

model = models(X_train,Y_train)

st.write(model)