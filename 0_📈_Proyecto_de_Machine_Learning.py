from turtle import width
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import cufflinks as cf
from pylab import rcParams
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.models import HoverTool
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

