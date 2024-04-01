import streamlit as st
import pandas as pd
import os
from aplicacao import train_model, eda_tab, train_model_tab, avaliacao_modelo_tab, producao_tab, serve_model, evaluate_model, data_pre_process
from lists import models_map_class, plots_map_class
from sklearn.metrics import accuracy_score, classification_report
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="darkgrid")
st.set_option('deprecation.showPyplotGlobalUse', False)    

################################# MAIN APP #######################################
def main():
    
    st.title("Auto Machine Learning")
    tabs = ["Processamento de Dados", "Treinamento do Modelo", "Avaliação do Modelo", "Aplicação do Modelo"]
    selected_tab = st.sidebar.radio("Guia:", tabs)

    if selected_tab == "Processamento de Dados":
        with st.spinner('Processando Dataset...'):
            eda_tab()

    if selected_tab == "Treinamento do Modelo":
        with st.spinner('Treinando Modelo...'):
            train_model_tab()

    if selected_tab == "Avaliação do Modelo":
        with st.spinner('Avaliando Modelo...'):
            avaliacao_modelo_tab()
        
    if selected_tab == "Aplicação do Modelo":
        with st.spinner('Aplicando Modelo...'):
            producao_tab()
        
if __name__ == "__main__":
    main()
