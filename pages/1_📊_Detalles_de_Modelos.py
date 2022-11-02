import streamlit as st

st.set_page_config(
    page_title="Información de modelos",
    page_icon="🤖",
)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/open-book_1f4d6.png",
    width=100
)

def wikia():
    st.title("Modelos de aprendizaje utilizados")
    st.header("Máquina de vector soporte")
    st.markdown(
        """
        Las máquinas de vectores se soporte o máquinas de vector soporte son un conjunto de aprendizaje supervisado desarrollados por Vladimir Vapnik y su equipo en los laboratorios de AT&T Bell.
        Están relacionados con problemas de clasificación y regresión.
        Se le da un conjunto de ejemplos de entrenamiento (muestras) y podemos etiquetar las clases y entrenar un SVM para construir un modelo que prediga la clase de una nueva.
        Por lo tanto, dado un conjunto de puntos, subconjunto de un conjunto mayor (espacio), en el que cada uno de ellos pertenece a una de dos posibles categorías, y el propio algoritmo construye un modelo capaz de predecir si un punto nuevo (categoría que desconocemos) pertenece a una categoría o a la otra.
        Este tipo de algoritmo pertenece a la familia de clasificadores lineales.
        Está estrechamente relacionado con las redes neuronales.
        - Al usar una función kernel, resultan un método de entrenamiento alternativo para clasificadores polinomiales, funciones de base radial y perceptrón multicapa.
        """
    )
    st.header("Regresión logística")
    st.markdown(
        """
        En estadística, la regresión es un tipo de análisis de regresión utilizado para predecir utilizado para predecir el resultado de una variable categórica (una variable que puede adoptar un número limitado de categorías) en función de la variables independientes o predictoras.
        Es útil para modelar la probabilidad de un evento ocurriendo en función de otros factores. El análisis de regresión logística se enmarca en el conjunto de Modelos Lineales  Generalizados que usa como función de enlace la función “logit”.
        Las probabilidades que describen la posición resultado de un único ensayo se modelan como una función de variables explicativas, utilizando una función logística.
        La regresión logística es usada extensamente en las ciencias médicas y sociales. Otros nombres para regresión logística usados en varias áreas de aplicación incluyen modelo logístico, modelo logit, y clasificador de máxima entropía.
        """
        )
    st.header("Bosque aleatorio")
    st.markdown(
        """
        Es una combinación de árboles predictores tal que cada árbol depende de los valores de un vector aleatorio probado independientemente y con la misma distribución para cada uno de estos.
        Es una modificación sustancial de bagging que construye una larga colección de árboles no correlacionados y luego los promedia.
        La idea esencial del bagging es promediar muchos modelos ruidosos aproximadamente imparciales, y por tanto reducir la variación. 
        Los árboles son los candidatos ideales para el bagging, dado que ellos pueden registrar estructuras de interacción compleja en los datos, y si crecen suficientemente profundo, tienen relativamente baja parcialidad. Producto de que los árboles son notoriamente ruidosos, ellos se benefician enormemente al promediar.
        Siguen el siguiente algoritmo:
        1. Sea N el número de casos de prueba, M es el número de variables en el clasificador.
        2. Sea m el número de variables de entrada a ser usado para determinar la decisión en un nodo dado; m debe ser mucho menor que M.
        3. Elegir un conjunto de entrenamiento para este árbol y usar el resto de los casos de prueba para estimar el error.
        4. Para cada nodo del árbol, elegir aleatoriamente variables en las cuales basar la decisión. Calcular la mejor partición del conjunto de entrenamiento a partir de las variables.
        """
    )

    st.header("k-Nearest Neighbor")
    st.markdown(
        """
        Es un algoritmo basado en instancia de tipo supervisado de ML. Puede usarse para clasificar nuevas muestras(valores discretos) o para predecir(regresión,valores continuos).
        Sirve para clasificar valores buscando los puntos de datos “similares” aprendidos en la etapa de entrenamiento y haciendo conjeturas de nuevos puntos basado en esa clasificación.
        Busca observaciones más cercanas a la que se está tratando de predecir y clasificar el punto de interés basado en la mayoría de datos que le rodean.
        Se aplica en Sistemas de recomendación, búsqueda semántica y detección de anomalías.
        #### Ventajas
        Es sencillo de aprender e implementar
        #### Desventajas
        Utiliza todo el dataset para entrenar “cada punto” y por eso requiere de uso de mucha memoria y recursos de procesamiento.
        #### ¿Cómo funciona?
        1. Calcular la distancia entre el ítem a clasificar y el resto de ítems del dataset de entrenamiento,
        2. Seleccionar los “k” elementos más cercanos.
        3. Realizar una “votación de mayoría” entre los k puntos: los de una clase/etiqueta que “dominen” decidirán su clasificación final.
        Teniendo en cuenta esto, para decidir la clase de un punto es importante el valor de K, ya que determinará a qué grupo pertenecen los puntos, sobre todo las fronteras entre grupos.
        Para medir la “cercanía” entre puntos están la [distancia Euclidiana](https://es.wikipedia.org/wiki/Distancia_euclidiana) o la [Cosine Similarity](https://es.wikipedia.org/wiki/Similitud_coseno).

        """
    )

    st.header("Clasificador bayesiano ingenuo")
    st.markdown(
        """
        Es uno de los algoritmos más simples y poderosos para la clasificación basado en el Teorema de Bayes con una suposición de independencia entre los predictores. Es fácil de construir y útil para conjuntos de datos muy grandes.
        El clasificador asume que el efecto de una característica particular en una clase es independiente de otras características.
        La fórmula es:
        """)
    st.image("https://live.staticflickr.com/65535/47792141631_5788f52f0c_b.jpg")
    st.markdown(   
        """
        - P(h): es la probabilidad de que la hipótesis h sea cierta (independientemente de los datos). Esto se conoce como la probabilidad previa de h.
        - P(D): probabilidad de los datos (independientemente de la hipótesis). Esto se conoce como probabilidad previa.
        - P(h|D): es la probabilidad de la hipótesis h dada los datos D. Esto se conoce como la probabilidad posterior.
        #### Pasos para que el clasificador Naive Bayes calcule la probabilidad de un evento:
        1. calcular la probabilidad previa para las etiquetas de clase dadas.
        2. determinar la probabilidad de probabilidad con cada atributo para cada clase.
        3. poner estos valores en el teorema de Bayes y calcular la probabilidad posterior.
        4. ver qué clase tiene una probabilidad más alta, dado que la variable de entrada pertenece a la clase de probabilidad más alta.
        #### Ventajas
        - Es fácil y rápido predecir la clase de conjunto de datos de prueba. También funciona bien en la predicción multiclase.
        - Cuando se mantiene la suposición de independencia, un clasificador Naive Bayes funciona mejor en comparación con otros modelos como la Regresión Logística y se necesitan menos datos de entrenamiento.
        - Funciona bien en el caso de variables de entrada categóricas comparada con variables numéricas.
        #### Desventajas
        - Si la variable categórica tiene una categoría en el conjunto de datos de prueba, que no se observó en el conjunto de datos de entrenamiento, el modelo asignará una probabilidad de 0 y no podrá hacer una predicción. Esto se conoce a menudo como frecuencia cero. Para resolver esto, podemos utilizar la técnica de alisamiento.
        - Otra limitación de Naive Bayes es la asunción de predictores independientes. En la vida real, es casi imposible que obtengamos un conjunto de predictores que sean completamente independientes.
        """
    )
    

wikia()