import streamlit as st

st.set_page_config(
    page_title="Informaci贸n de modelos",
    page_icon="馃",
)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/open-book_1f4d6.png",
    width=100
)

st.title("Modelos de aprendizaje utilizados")
st.header("M谩quina de vector soporte")
st.markdown(
    """
    Las m谩quinas de vectores se soporte o m谩quinas de vector soporte son un conjunto de aprendizaje supervisado desarrollados por Vladimir Vapnik y su equipo en los laboratorios de AT&T Bell.
    Est谩n relacionados con problemas de clasificaci贸n y regresi贸n.
    Se le da un conjunto de ejemplos de entrenamiento (muestras) y podemos etiquetar las clases y entrenar un SVM para construir un modelo que prediga la clase de una nueva.
    Por lo tanto, dado un conjunto de puntos, subconjunto de un conjunto mayor (espacio), en el que cada uno de ellos pertenece a una de dos posibles categor铆as, y el propio algoritmo construye un modelo capaz de predecir si un punto nuevo (categor铆a que desconocemos) pertenece a una categor铆a o a la otra.
    Este tipo de algoritmo pertenece a la familia de clasificadores lineales.
    Est谩 estrechamente relacionado con las redes neuronales.
    - Al usar una funci贸n kernel, resultan un m茅todo de entrenamiento alternativo para clasificadores polinomiales, funciones de base radial y perceptr贸n multicapa.
    """
)
st.header("Regresi贸n log铆stica")
st.markdown(
    """
    En estad铆stica, la regresi贸n es un tipo de an谩lisis de regresi贸n utilizado para predecir utilizado para predecir el resultado de una variable categ贸rica (una variable que puede adoptar un n煤mero limitado de categor铆as) en funci贸n de la variables independientes o predictoras.
    Es 煤til para modelar la probabilidad de un evento ocurriendo en funci贸n de otros factores. El an谩lisis de regresi贸n log铆stica se enmarca en el conjunto de Modelos Lineales  Generalizados que usa como funci贸n de enlace la funci贸n 鈥渓ogit鈥?.
    Las probabilidades que describen la posici贸n resultado de un 煤nico ensayo se modelan como una funci贸n de variables explicativas, utilizando una funci贸n log铆stica.
    La regresi贸n log铆stica es usada extensamente en las ciencias m茅dicas y sociales. Otros nombres para regresi贸n log铆stica usados en varias 谩reas de aplicaci贸n incluyen modelo log铆stico, modelo logit, y clasificador de m谩xima entrop铆a.
    """
    )
st.header("Bosque aleatorio")
st.markdown(
    """
    Es una combinaci贸n de 谩rboles predictores tal que cada 谩rbol depende de los valores de un vector aleatorio probado independientemente y con la misma distribuci贸n para cada uno de estos.
    Es una modificaci贸n sustancial de bagging que construye una larga colecci贸n de 谩rboles no correlacionados y luego los promedia.
    La idea esencial del bagging es promediar muchos modelos ruidosos aproximadamente imparciales, y por tanto reducir la variaci贸n. 
    Los 谩rboles son los candidatos ideales para el bagging, dado que ellos pueden registrar estructuras de interacci贸n compleja en los datos, y si crecen suficientemente profundo, tienen relativamente baja parcialidad. Producto de que los 谩rboles son notoriamente ruidosos, ellos se benefician enormemente al promediar.
    Siguen el siguiente algoritmo:
    1. Sea N el n煤mero de casos de prueba, M es el n煤mero de variables en el clasificador.
    2. Sea m el n煤mero de variables de entrada a ser usado para determinar la decisi贸n en un nodo dado; m debe ser mucho menor que M.
    3. Elegir un conjunto de entrenamiento para este 谩rbol y usar el resto de los casos de prueba para estimar el error.
    4. Para cada nodo del 谩rbol, elegir aleatoriamente variables en las cuales basar la decisi贸n. Calcular la mejor partici贸n del conjunto de entrenamiento a partir de las variables.
    """
)

st.header("k-vecinos m谩s cercanos")
st.markdown(
    """
    Es un algoritmo basado en instancia de tipo supervisado de ML. Puede usarse para clasificar nuevas muestras(valores discretos) o para predecir(regresi贸n,valores continuos).
    Sirve para clasificar valores buscando los puntos de datos 鈥渟imilares鈥? aprendidos en la etapa de entrenamiento y haciendo conjeturas de nuevos puntos basado en esa clasificaci贸n.
    Busca observaciones m谩s cercanas a la que se est谩 tratando de predecir y clasificar el punto de inter茅s basado en la mayor铆a de datos que le rodean.
    Se aplica en Sistemas de recomendaci贸n, b煤squeda sem谩ntica y detecci贸n de anomal铆as.
    #### Ventajas
    - Es sencillo de aprender e implementar
    #### Desventajas
    - Utiliza todo el dataset para entrenar 鈥渃ada punto鈥? y por eso requiere de uso de mucha memoria y recursos de procesamiento.
    #### 驴C贸mo funciona?
    1. Calcular la distancia entre el 铆tem a clasificar y el resto de 铆tems del dataset de entrenamiento,
    2. Seleccionar los 鈥渒鈥? elementos m谩s cercanos.
    3. Realizar una 鈥渧otaci贸n de mayor铆a鈥? entre los k puntos: los de una clase/etiqueta que 鈥渄ominen鈥? decidir谩n su clasificaci贸n final.
    Teniendo en cuenta esto, para decidir la clase de un punto es importante el valor de K, ya que determinar谩 a qu茅 grupo pertenecen los puntos, sobre todo las fronteras entre grupos.
    Para medir la 鈥渃ercan铆a鈥? entre puntos est谩n la [distancia Euclidiana](https://es.wikipedia.org/wiki/Distancia_euclidiana) o la [Cosine Similarity](https://es.wikipedia.org/wiki/Similitud_coseno).

    """
)

st.header("Clasificador bayesiano ingenuo")
st.markdown(
    """
    Es uno de los algoritmos m谩s simples y poderosos para la clasificaci贸n basado en el teorema de Bayes con una suposici贸n de independencia entre los predictores. Es f谩cil de construir y 煤til para conjuntos de datos muy grandes.
    El clasificador asume que el efecto de una caracter铆stica particular en una clase es independiente de otras caracter铆sticas.
    La f贸rmula es:
    """)
st.image("https://live.staticflickr.com/65535/47792141631_5788f52f0c_b.jpg")
st.markdown(   
    """
    - P(h): es la probabilidad de que la hip贸tesis h sea cierta (independientemente de los datos). Esto se conoce como la probabilidad previa de h.
    - P(D): probabilidad de los datos (independientemente de la hip贸tesis). Esto se conoce como probabilidad previa.
    - P(h|D): es la probabilidad de la hip贸tesis h dada los datos D. Esto se conoce como la probabilidad posterior.
    #### Pasos para que el clasificador bayesiano ingenuo calcule la probabilidad de un evento:
    1. calcular la probabilidad previa para las etiquetas de clase dadas.
    2. determinar la probabilidad de probabilidad con cada atributo para cada clase.
    3. poner estos valores en el teorema de Bayes y calcular la probabilidad posterior.
    4. ver qu茅 clase tiene una probabilidad m谩s alta, dado que la variable de entrada pertenece a la clase de probabilidad m谩s alta.
    #### Ventajas
    - Es f谩cil y r谩pido predecir la clase de conjunto de datos de prueba. Tambi茅n funciona bien en la predicci贸n multiclase.
    - Cuando se mantiene la suposici贸n de independencia, un clasificador bayesiano ingenuo funciona mejor en comparaci贸n con otros modelos como la Regresi贸n Log铆stica y se necesitan menos datos de entrenamiento.
    - Funciona bien en el caso de variables de entrada categ贸ricas comparada con variables num茅ricas.
    #### Desventajas
    - Si la variable categ贸rica tiene una categor铆a en el conjunto de datos de prueba, que no se observ贸 en el conjunto de datos de entrenamiento, el modelo asignar谩 una probabilidad de 0 y no podr谩 hacer una predicci贸n. Esto se conoce a menudo como frecuencia cero. Para resolver esto, podemos utilizar la t茅cnica de alisamiento.
    - Otra limitaci贸n del clasificador bayesiano ingenuo es la asunci贸n de predictores independientes. En la vida real, es casi imposible que obtengamos un conjunto de predictores que sean completamente independientes.
    """
)