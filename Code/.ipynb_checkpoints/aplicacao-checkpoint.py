########################## EDA TAB ##############################################
def eda_tab():
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from plots_func import generate_plot_eda
    from lists import normalize_methods, outliers_methods
    import mlflow

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.title("Análise Exploratória de Dados")

    # Load Dataset
    st.sidebar.subheader("Configurações")
    file_to_load = st.sidebar.file_uploader("Carregar arquivo de dados brutos:", type=["csv", "parquet"])
    if file_to_load is not None:
        df = load_dataframe(file_to_load)

        # Exibir os dados brutos 
        st.subheader("Visualização dos Dados Brutos")
        st.write(f"O Dataset original contém {df.shape[0]} amostras, {df.shape[1]} atributos e {df.isna().sum().sum()} valores Nulos/NA")
        st.write(df)

        # Seleção de colunas para análise
        selected_columns = st.sidebar.multiselect("Selecionar colunas para filtragem:", options=sorted(df.columns.tolist()), default=sorted(df.columns.tolist()))
        target_col = st.sidebar.selectbox("Coluna alvo:", options=selected_columns)


        st.sidebar.subheader("Parâmetros(Dados)")
        # Realizar a limpeza de dados
        if st.sidebar.toggle("Remover dados faltantes(null/NA):", value=True):
            df_clean = df[selected_columns].dropna()
        else:
            df_clean = df[selected_columns]

        # Imbalance
        fix_imbalance = st.sidebar.toggle("Balancear(SMOTE)", value=True)
        
        # Outliers
        remove_outliers = st.sidebar.toggle("Remover 'Outliers'", value=True)  
        if remove_outliers:
            selected_outliers_method = st.sidebar.selectbox("Método de remoção de 'Outliers':", list(outliers_methods.keys()), index=0)
            outliers_method = outliers_methods[selected_outliers_method]
        else:
            outliers_method = "iforest"
        
        # Normalize
        normalize = st.sidebar.toggle("Normalizar", value=True)    
        if normalize:
            selected_normalize_method = st.sidebar.selectbox("Método de normalização:", list(normalize_methods.keys()), index=0)
            normalize_method = normalize_methods[selected_normalize_method]
        else:
            normalize_method = "zscore"         

        try:
            st.session_state.df_clean = df_clean
        except:
            df_clean = df_clean
        try:
            # Análise Estatística Descritiva
            st.subheader("Análise Estatística Descritiva")
            st.write(df_clean.describe())
        except Exception as e:
            st.write(f"Favor escolher Colunas para Análise")

        try:
            # Visualização de dados faltantes
            df_clean_missing = df_clean.isnull().sum().reset_index()
            df_clean_missing = df_clean_missing.rename(columns={"index": "Colunas", 0: "N º N/A"})    
            st.subheader("Visualização de Valores Faltantes")
            st.write(df_clean_missing)
        except Exception as e:
            st.write(f"Favor escolher Colunas para Análise")
        
        st.sidebar.subheader("Gráfico Pré Processamento")
        st.header("Gráfico")
        # Escolha do tipo de gráfico
        plot_type = st.sidebar.selectbox("Escolher gráfico para plotagem:", ["Categorias", "Distribuição", "Heatmap",  "Joinplot",  "Pairplot", "Relacional"], index=None)
        
        # Chamada da função para gerar o plot
        try:
            generate_plot_eda(df_clean, selected_columns, plot_type)
        except Exception as e:
            st.error(f"Ocorreu um erro. Verifque os atributos para plotagem: {e}")     

        st.sidebar.subheader("Parâmetros de Rastreamento")
        experiment_id = st.sidebar.text_input("ID do experimento:", value="Kob" )
        experiment_name = st.sidebar.text_input("Nome do experimento(run):", value="PreparacaoDados")
        mlflow_db = st.sidebar.text_input("BD rastreamento(MLflow):", "mlruns")

        try:
            st.session_state.mlflow_db = mlflow_db
            st.session_state.experiment_id = experiment_id
            st.session_state.experiment_name = experiment_name 
            st.session_state.target_col = target_col 
        except:
            st.write("")
        
        # Save preprocess dataset
        st.sidebar.subheader("Salvar Dados Processados")        
        file_name = st.sidebar.text_input("Nome Dataset Processado(sem extensão):", "data_clean")
        save_format = st.sidebar.selectbox("Formato de arquivo para salvar:", ["CSV", "Parquet"])        
        try:
            if st.sidebar.button("Aplicar e Salvar"):
                try:
                    df_clean = st.session_state.df_clean
                except:
                    df_clean = df_clean
                # Call preprocess function
                df_clean = data_pre_process(df_clean, remove_outliers, outliers_method, fix_imbalance, normalize, normalize_method, mlflow_db, experiment_id, experiment_name, target_col, file_to_load, file_name, save_format)
        except Exception as e:
            st.write(f"Verificar entrada da parâmetros.{e}")

########################### TRAIN MODEL TAB #####################################

def train_model_tab():
    import streamlit as st
    import pandas as pd
    import os
    import time
    from lists import models_map_class, plots_map_class, metrics_list_class, optimize_class, normalize_methods, outliers_methods, fold_strategys

    st.sidebar.title("Treinamento do Modelo")
    # Sidebar Inputs
    st.sidebar.subheader("Configurações")

    # Load session stats
    try:
        target_col = st.session_state.target_col    
        mlflow_db = st.session_state.mlflow_db
        experiment_id = st.session_state.experiment_id
    except:
        target_col, mlflow_db, experiment_id,   = "", "", ""

    # Caminho base onde os datasets ficam salvos para treinamento
    base_file_path = "../Data/processed/"
    
    file_to_load = st.sidebar.file_uploader("Carregar Dataset de Treino:", type=["csv", "parquet"])
    if file_to_load is not None:
        df_loaded = load_dataframe(file_to_load)
            
        # Obter nomes das colunas e armazená-los no estado da sessão
        columns = df_loaded.columns.tolist()
        st.session_state.df_columns = columns
        st.session_state.df_loaded = df_loaded
        
    test_size_percentage = st.sidebar.number_input("Tamanho da base de teste(%): ", min_value=5, max_value=100, value=20, step=5)
    test_size = test_size_percentage / 100.0

    # Populate columns to keep and target column after loading column names
    if 'df_columns' in st.session_state:
        col_to_keep = st.sidebar.multiselect("Colunas a Analisar(Alvo incluso):", options=st.session_state.df_columns, default=st.session_state.df_columns)
        target_col = st.sidebar.selectbox("Coluna alvo:", options=st.session_state.df_columns)

    # Model Selection with Popover
    st.sidebar.subheader("Modelos")
    selected_models = st.sidebar.multiselect("Escolha os modelos:", options=list(models_map_class.keys()), format_func=lambda x: x)
    selected_models = [models_map_class[model] for model in selected_models]
    models = selected_models    
    
    # Parameters 
    st.sidebar.subheader("Parâmetros(Dados)")
    # Outliers
    remove_outliers = st.sidebar.toggle("Remover 'Outliers'", value=False)  
    if remove_outliers:
        selected_outliers_method = st.sidebar.selectbox("Método de Remoção de 'Outliers':", list(outliers_methods.keys()), index=0)
        outliers_method = outliers_methods[selected_outliers_method]
    else:
        outliers_method = "iforest"
    # Imbalance
    fix_imbalance = st.sidebar.toggle("Balancear(SMOTE) ", value=False)
    # Normalize
    normalize = st.sidebar.toggle("Normalizar", value=False)    
    if normalize:
        selected_normalize_method = st.sidebar.selectbox("Método de normalização:", list(normalize_methods.keys()), index=0)
        normalize_method = normalize_methods[selected_normalize_method]
    else:
        normalize_method = "zscore"    
    #
    st.sidebar.subheader("Parâmetros(Setup)")
    folds = st.sidebar.number_input("Nº Folds: ", min_value=1, max_value=50, value=10)
    selected_fold_strategy = st.sidebar.selectbox("Método de Remoção de 'Outliers':", list(fold_strategys.keys()), index=0)
    fold_strategy = fold_strategys[selected_fold_strategy]    
    #
    st.sidebar.subheader("Parâmetros(Tuning)")
    selected_optimize_key = st.sidebar.selectbox("Otimizador:", options=list(optimize_class.keys()), format_func=lambda x: x)
    optimize = optimize_class[selected_optimize_key]   
    n_iter = st.sidebar.number_input("Nº Iterações: ", min_value=1, max_value=50, value=10)
    # 
    st.sidebar.subheader("Experimento")
    mlflow_db = st.sidebar.text_input("BD rastreamento(MLflow):", f"{mlflow_db}")
    experiment_id = st.sidebar.text_input("ID do experimento:", f"{experiment_id}")
    experiment_name = st.sidebar.text_input("Nome do experimento(run):", "Treinamento")

    # Classification Plots with Popover
    st.sidebar.subheader("Plots de Classificação")
    selected_plots = st.sidebar.multiselect("Plots a serem gerados(Artefatos):", options=list(plots_map_class.keys()), format_func=lambda x: x)
    selected_plots = [plots_map_class[plot] for plot in selected_plots]
    classification_plots = selected_plots

    # Load Session variables
    try:
        st.session_state.target_col = target_col
    except:
        st.write("")
    st.session_state.mlflow_db = mlflow_db
    st.session_state.experiment_id = experiment_id
    st.session_state.experiment_name = experiment_name    
        
    # Train Button
    try:
        if st.sidebar.button("Treinar"):
            # Load session dataset
            df_clean = st.session_state.df_loaded
            df_clean = df_clean[col_to_keep]
    
            # Run Parameters
            parameters = {"seed": 10, "experiment_name": experiment_name}
    
            # Training Call
            result, train_loss, test_loss, train_f1, test_f1, base_train_size, base_test_size, leaderboard = train_model(experiment_id, experiment_name, test_size, selected_models, parameters, df_clean, target_col, mlflow_db, selected_plots, optimize, n_iter, normalize, normalize_method, fix_imbalance, remove_outliers, outliers_method, folds, fold_strategy)
    
            col1, col2 = st.columns(2)
            with col1:
            # Iterando sobre o dicionário de métricas e imprimindo os resultados
                for metrica_pt, metrica_en in metrics_list_class.items():
                    st.write(f"{metrica_pt}: {round(result.metrics[metrica_en],4)}")
                st.write(f"Tamanho Dataset Treino: {base_train_size}")
                st.write(f"Otimizador: {optimize}")
    
            with col2:                    
                st.write(f"Log Loss Treino: {round(train_loss,4)}")
                st.write(f"Log Loss Test: {round(test_loss,4)}")
                st.write(f"F1-Score Treino: {round(train_f1,4)}")
                st.write(f"F1-Score Teste: {round(test_f1,4)}")
                st.write(f"Tamanho Dataset Teste: {base_test_size}")
                st.write(f"Nº Iterações: {n_iter}")
                     
            st.write("Métricas dos Modelos Comparados")
            st.write(leaderboard)
    except Exception as e:
        st.write(f"Verificar entrada da parâmetros.")
    
    # Display plots created in the last run
    plot_files = [f for f in os.listdir() if f.endswith(('.png', '.jpg', '.jpeg')) and time.time() - os.path.getmtime(f) < 60]
    for plot_file in plot_files:
        st.image(plot_file, caption=plot_file, use_column_width=True)

############################# EVALUATE MODELO TAB ##################################

def avaliacao_modelo_tab():
    from aplicacao import serve_model, evaluate_model
    from lists import metrics_list_eval
    import pandas as pd
    import streamlit as st
    
    st.sidebar.title("Avaliação do Modelo")    

    # Carrega o dataset de produção usando o file uploader
    file_to_load  = st.sidebar.file_uploader("Carregar dataset de validação:", type=["csv", "parquet"])
    if file_to_load is not None:
        production_data = load_dataframe(file_to_load)                  
        #st.sidebar.write(production_data.columns.tolist())
    else:
        st.empty()

    # Recuperar valores de target_col e mlflow_db da sessão
    try:
        target_col = st.session_state.target_col    
        mlflow_db = st.session_state.mlflow_db
        experiment_id = st.session_state.experiment_id
        experiment_name = st.session_state.experiment_name    
        model_uri = st.session_state.mlflow_model_uri 
    except:
        target_col, mlflow_db, experiment_id, experiment_name, model_uri = "", "", "", "", "",  
        
    # Inputs do usuário
    try:
        target_col = st.sidebar.selectbox("Nome da coluna alvo:", options=production_data.columns.tolist())
    except:
        target_col = ""
    experiment_id = st.sidebar.text_input("ID do experimento avaliação:", value=experiment_id )
    experiment_name = st.sidebar.text_input("Nome do experimento(run) avaliação:", value="")    
    mlflow_db = st.sidebar.text_input("BD rastreamento(MLflow) do MLflow:", value="mlruns")    
    port = st.sidebar.number_input("Porta (port):", min_value=1, max_value=65535, value=5001)
    
    uri_choice = st.sidebar.selectbox("Escolha run ou modelo:", ["Último run", "Modelo registrado"])

    if uri_choice == "Último run":
        formatted_model_uri = format_run_uri(model_uri)
    else:
        formatted_model_uri = format_registered_model_uri()
        
    logged_model = st.sidebar.text_input("Caminho do run/model:", value=formatted_model_uri)
    model_uri = logged_model

    # Botão para aplicar o modelo
    try:
        if st.sidebar.button("Avaliar Modelo") and production_data is not None:
            # Chama a função para servir o modelo em um processo separado
            serve_model(model_uri, port, mlflow_db)
    
            # Espera alguns segundos para garantir que o servidor MLflow esteja iniciado antes de avaliar o modelo
            import time
            time.sleep(2)
    
            # Avalia o modelo
            result, accuracy, classification_report_df, confusion_table, roc_auc, precision, recall, f1, mcc = evaluate_model(experiment_id, experiment_name, logged_model, production_data, target_col, mlflow_db)
    
            col1, col2 = st.columns(2)
            metrics_list_eval_items = list(metrics_list_eval.items())
            
            with col1:
                for metrica_pt, metrica_en in metrics_list_eval_items[:4]:
                    st.write(f"{metrica_pt}: {round(result.metrics[metrica_en], 4)}")
    
            with col2:       
                for metrica_pt, metrica_en in metrics_list_eval_items[-4:]:
                    st.write(f"{metrica_pt}: {round(result.metrics[metrica_en], 4)}")
    
            st.write("Classification Report:")
            st.write(classification_report_df) 
    
            st.write("Matrix de Confusão")
            st.table(confusion_table)
            
            st.write("Métricas")
            st.write(result.metrics)
    except Exception as e:
        st.write(f"Verificar entrada da parâmetros. {e}")

######################## HEALTH MODEL TAB #################################

def saude_model_tab():
    import pandas as pd
    import streamlit as st
    
    st.sidebar.title("Saúde do Modelo")
    
    data_prod_choice = st.sidebar.toggle("Dataset para comparação(opcional):", value=True)
    if data_prod_choice is True:
        file_to_load = st.sidebar.file_uploader("Carregar Dataset:", type=["csv", "parquet"])
        if file_to_load is not None:
            production_df = load_dataframe(file_to_load)     
        else:
            production_df = False    
    else:
        production_df = False 
    
    try:
        columns = production_df.columns.tolist()
        target_col = st.sidebar.selectbox("Coluna alvo:", options=columns)
    except:
        columns = "" 
        target_col = st.sidebar.text_input("Coluna alvo:")

    try:
        target_col = st.session_state.target_col    
        mlflow_db = st.session_state.mlflow_db 
        model_uri = st.session_state.mlflow_model_uri 
        model_uri = format_run_uri(model_uri)
    except:
        mlflow_db, model_uri = "", ""

    mlflow_db = st.sidebar.text_input("BD rastreamento(MLflow):", f"{mlflow_db}")
    model_uri = st.sidebar.text_input("Caminho do Modelo(Uri MLflow):", f"{model_uri}")

    try:
        if st.sidebar.button("Verificar Saúde"):
            train_report, test_report, production_report = evaluate_and_compare_datasets(mlflow_db, model_uri, target_col, production_df)
            
            st.write("Relatório de Classificação(Treino)")
            st.dataframe(train_report)
            st.write("Relatório de Classificação(Teste)")
            st.dataframe(test_report)
            
            if production_report is not None:
                st.write("Relatório de Classificação(Dataset)")
                st.dataframe(production_report)     
    except Exception as e:
        st.write(f"Verificar entrada da parâmetros.")

######################## PRODUCTION MODEL TAB #################################

def producao_tab():
    import pandas as pd
    import streamlit as st

    st.sidebar.title("Aplicação do Modelo")
    file_to_load = st.sidebar.file_uploader("Carregar Dataset de Produção:", type=["csv", "parquet"])
    if file_to_load is not None:
        df_producao = load_dataframe(file_to_load)            
    else:
        df_producao = ""
        st.write("Adiconar base de dados")
    try:
        columns = df_producao.columns.tolist()
    except:
        columns = "" 

    alvo_bool = st.sidebar.toggle("Dataset possue alvo?(Ativo = Sim):", value=True)
    if alvo_bool is True:
        target_col = st.sidebar.selectbox("Coluna alvo:", options=columns)
        
    mlflow_db = st.sidebar.text_input("BD rastreamento(MLflow):", "mlruns")

    try:
        model_uri = st.session_state.mlflow_model_uri 
        model_uri = format_run_uri(model_uri)
        model_uri = st.sidebar.text_input("Caminho do Modelo(Uri MLflow):", f"{model_uri}")
    except:
        model_uri = st.sidebar.text_input("Caminho do Modelo(Uri MLflow):", "")
    try:
        if st.sidebar.button("Aplicar Modelo"):
            if alvo_bool is True:
                report, acuracia = production_model_alvo(df_producao, columns, mlflow_db, target_col, model_uri)
                st.write(f"Acurária: {acuracia}")
                st.write(report)
            else:
                df_result = production_model(df_producao, mlflow_db, model_uri) 
                st.write(df_result)

    except Exception as e:
        st.write(f"Verificar entrada da parâmetros.{e}")

############################################################################################################

########################## DATA PROCESS ########################################

def data_pre_process(df_clean, remove_outliers, outliers_method, fix_imbalance, normalize, normalize_method, mlflow_db, experiment_id, experiment_name, target_col, file_to_load, file_name, save_format):
    import pycaret.classification as pc
    import pandas as pd
    import streamlit as st
    import mlflow

    # Setting experiment Name    
    experiment_id = check_experiment_id(experiment_id)

    mlflow.set_tracking_uri(f"sqlite:///Models/{mlflow_db}.db")   

    with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name):
        data_exp = pc.setup(df_clean, 
                            target=target_col, 
                            remove_outliers=remove_outliers, 
                            outliers_method=outliers_method, 
                            fix_imbalance=fix_imbalance,
                            normalize=normalize, 
                            normalize_method=normalize_method
                           )
        df_clean = data_exp.get_config("dataset_transformed") 

        base_save_path = "../Data/processed/"
        if save_format == "CSV":
            save_path = base_save_path + file_name + ".csv"
            df_clean = df_clean.sort_index(axis=1)
            df_clean.to_csv(save_path, index=False)            
        elif save_format == "Parquet":
            save_path = base_save_path + file_name + ".parquet"
            df_clean = df_clean.sort_index(axis=1)
            df_clean.to_parquet(save_path, index=False)
            
        st.sidebar.success("Dados salvos com sucesso!")
        st.subheader("Visualização processados")
        st.write(f"O Dataset filtrado contém {df_clean.shape[0]} amostras, {df_clean.shape[1]} atributos e {df_clean.isna().sum().sum()} valores Nulos/NA")
        st.write(df_clean)        
        
        #  Logging
        mlflow.log_param("selected_features", df_clean.columns.tolist())      
        mlflow.log_param("remove_outliers", remove_outliers)
        mlflow.log_param("outliers_method", outliers_method)
        mlflow.log_param("fix_imbalance", fix_imbalance)
        mlflow.log_param("normalize", normalize)
        mlflow.log_param("normalize_method", normalize_method)
        mlflow.log_param("raw_dataset", file_to_load.name)
        mlflow.log_param("processed_dataset", f"{file_name}.{save_format.lower()}")
        mlflow.log_metric("processed_dataset_size", df_clean.shape[0])
        mlflow.log_artifact(save_path)
        print(">>>>>>>>>> Dataset Processado Salvo com Sucesso!")
            
    mlflow.end_run()
    return df_clean

########################### TRAIN MODEL #########################################

def train_model(experiment_id, experiment_name, test_size, models, parameters, clean_dataset, target_col, mlflow_db, classification_plots, optimize, n_iter, normalize, normalize_method, fix_imbalance, remove_outliers, outliers_method, folds, fold_strategy) -> None:
    from sklearn.metrics import log_loss, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import mlflow
    import numpy as np
    import pandas as pd
    import pycaret.classification as pc
    from pycaret.classification import get_leaderboard
    from mlflow.models.signature import infer_signature
    from mlflow.tracking import MlflowClient
    import os
    import streamlit as st
    
    # Configure MLflow Tracking
    mlflow.set_tracking_uri(f"sqlite:///Models/{mlflow_db}.db")   

    # Setting experiment Name    
    experiment_id = check_experiment_id(experiment_id)

    #Train, Test Split
    base_train, base_test = train_test_split(clean_dataset, test_size=test_size) 

    # Saving Train Set
    base_train_path = "../Data/processed/base_train.parquet"
    base_train.to_parquet(base_train_path, index=False)
    
    # Saving Test Set
    base_test_path = "../Data/processed/base_test.parquet"
    base_test.to_parquet(base_test_path, index=False)    
    
    # Start Mlflow Run and Model Setup
    with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name) as run:
        experiment = pc.setup(
            session_id=10,
            data=base_train,
            test_data=base_test,
            target=target_col,
            profile=False,
            fold=folds,
            fold_strategy=fold_strategy,
            fix_imbalance=fix_imbalance,
            remove_outliers=remove_outliers,
            outliers_method=outliers_method,
            normalize=normalize,
            normalize_method = normalize_method,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            experiment_name=experiment_name,
            log_experiment=False,
        )

        # Model Training and Tuning
        print(">>>>>>>>>> Comparing Models...")
        model = pc.compare_models(n_select=1, sort="Accuracy", include=models)
        print('(">>>>>>>>>> Selected Model: ', model)

        print(">>>>>>>>>> Tuning Model...")
        tuned_model = pc.tune_model(model, optimize=optimize, search_library="scikit-learn", search_algorithm="random", n_iter=n_iter, choose_better=True,)

        print(">>>>>>>>>> Calibrating Model...")
        calibrated_model = pc.calibrate_model(tuned_model, method="sigmoid", calibrate_fold=5)
        # Best Model
        final_model = pc.finalize_model(calibrated_model)
        # Leaderboard
        leaderboard = get_leaderboard()
        
        # Save Model  
        print(">>>>>>>>>> Saving Model...")
        try:
            pc.save_model(final_model, experiment_name)
            mlflow.log_artifact(f"./{experiment_name}.pkl") 
        except Exception as e:
            print(f"Failed to save model. Error: {e}")

        # Load Full Pipeline
        model_pipe = pc.load_model(f'./{experiment_name}')
        
        # Model Signature Inferred by MLFlow
        model_features = list(clean_dataset.drop(target_col, axis=1).columns)
        inf_signature = infer_signature(clean_dataset[model_features], model_pipe.predict(clean_dataset.drop(target_col, axis=1)))
        
        # Example input for MLmodel
        nexamples = 6
        input_example = {x: clean_dataset[x].values[:nexamples] for x in model_features}
        
        # Log do pipeline de modelagem do sklearn e registrar como uma nova versao
        mlflow.sklearn.log_model(
            sk_model=model_pipe,
            artifact_path="pycaret-model",
            registered_model_name=experiment_name,
            signature = inf_signature,
            input_example = input_example
        )

        # Evaluate the model in MLflow
        model_uri = mlflow.get_artifact_uri("pycaret-model")
        eval_data = clean_dataset  
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets=target_col,
            model_type="classifier",
            evaluators=["default"],
        )

        # Get the MLflow model URI and store it in the session
        st.session_state.mlflow_model_uri = model_uri        
                
        # Calculating log loss and F1-score
        train_pred, test_pred = final_model.predict(base_train.drop(target_col, axis=1)), final_model.predict(base_test.drop(target_col, axis=1))
        train_loss, test_loss = log_loss(base_train[target_col], train_pred), log_loss(base_test[target_col], test_pred)
        train_f1, test_f1 = f1_score(base_train[target_col], train_pred), f1_score(base_test[target_col], test_pred)
        base_train_size, base_test_size = base_train.shape[0], base_test.shape[0]
    
        # Log Metrics
        mlflow.log_metric("Train Log Loss", train_loss)
        mlflow.log_metric("Test Log Loss", test_loss)
        mlflow.log_metric("Train F1-score", train_f1)
        mlflow.log_metric("Test F1-score", test_f1)
        mlflow.log_metric("Train Dataset Size", base_train_size)
        mlflow.log_metric("Test Dataset Size", base_test_size)
        
        # Log Experiment Parameters
        mlflow.log_params(parameters)
        mlflow.log_param("Test Size", test_size) 
        
        # Log Artifacts  
        mlflow.log_artifact(base_train_path)
        mlflow.log_artifact(base_test_path)        
        
        for plot_type in classification_plots:
            print(f"Saving Artifact Plot: {plot_type}")
            try:
                artifact = pc.plot_model(final_model, plot=plot_type, save=True)
                mlflow.log_artifact(artifact)
            except Exception as e:
                print(f"Failed to save artifact plot for: {plot_type}. Error: {e}")
                
        print(">>>>>>>>>> End of run!")
    mlflow.end_run()
    return result, train_loss, test_loss, train_f1, test_f1, base_train_size, base_test_size, leaderboard
    
######################## SERVE MODEL ############################################

def serve_model(model_uri, port, mlflow_db):
    import mlflow
    from sklearn.metrics import accuracy_score, classification_report
    import os
    import subprocess
    import time
    # Set the MLFLOW_TRACKING_URI environment variable
    os.environ['MLFLOW_TRACKING_URI'] = f"sqlite:///Models/{mlflow_db}.db"

    # Create the command to serve the model
    command = f"mlflow models serve -m {model_uri} --no-conda -p {port}"

    # Execute the command using subprocess in a separate process
    subprocess.Popen(command, shell=True)
    
    # Await 10 sec
    time.sleep(2)

######################## EVALUATE MODEL #########################################

def evaluate_model(experiment_id, experiment_name, logged_model, production_data, target_col, mlflow_db):
    import pandas as pd
    import mlflow
    import pycaret.classification as pc
    from sklearn import metrics
    from mlflow.models.signature import infer_signature
    from sklearn.metrics import log_loss, f1_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score, matthews_corrcoef
    
    # Configure MLflow Tracking
    mlflow.set_tracking_uri(f"sqlite:///Models/{mlflow_db}.db") 

    # Setting experiment Name    
    experiment_id = check_experiment_id(experiment_id)

    # Start Mlflow Run and Model Setup
    with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name) as run:
    
        # Load the trained model
        model = mlflow.sklearn.load_model(logged_model)
    
        # Extract features from the production dataset
        X_production = production_data.drop(columns=[target_col])
    
        # Make predictions
        pc.setup(data = production_data)
        # predictions = model.predict(X_production)
        predictions = pc.predict_model(model, raw_score=True)
        predictions.drop('prediction_score_0', axis=1, inplace=True)
        predictions.rename({'prediction_score_1': 'prediction_score'}, axis=1, inplace=True)
        # metrics
        precision = metrics.precision_score(predictions[target_col], predictions['prediction_label'])
        recall = metrics.recall_score(predictions[target_col], predictions['prediction_label'])
        accuracy = metrics.accuracy_score(predictions[target_col], predictions['prediction_label'])
        f1 = metrics.f1_score(predictions[target_col], predictions['prediction_label'])
        mcc = metrics.matthews_corrcoef(predictions[target_col], predictions['prediction_label'])
        roc_auc = metrics.roc_auc_score(predictions[target_col], predictions['prediction_label'])
        classification_report = metrics.classification_report(predictions[target_col], predictions['prediction_label'], output_dict=True)        
        classification_report_df = pd.DataFrame(classification_report)
        confusion_matrix = metrics.confusion_matrix(predictions[target_col], predictions['prediction_label'])


         # Calculating log loss and F1-score
        #prod_loss = log_loss(predictions[target_col], predictions['prediction_label'])
        #prod_f1 = f1_score(predictions[target_col], predictions['prediction_label'])
        #prod_size = production_data.shape[0]
    
        # Log Metrics
        #mlflow.log_metric("Log Loss", prod_loss)
        #mlflow.log_metric("F1-score", prod_f1)
        #mlflow.log_metric("Dataset Size", prod_size)

        parquet_file_path = "../docs/classification_report.parquet"
        classification_report_df.to_parquet(parquet_file_path)
            
        mlflow.log_artifact(parquet_file_path)

        # Create a model signature
        signature = infer_signature(X_production, predictions)

        # Log the baseline model to MLflow
        mlflow.sklearn.log_model(model, "pycaret-model", signature=signature)
    
        # Evaluate the model in MLflow
        model_uri = mlflow.get_artifact_uri("pycaret-model")
        eval_data = production_data 
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets=target_col,
            model_type="classifier",
            evaluators=["default"],
        )
       
        confusion_table = pd.DataFrame({
        'Positivo': ['Verdadeiro Positivo (VP)', confusion_matrix[1, 1], 'Falso Positivo (FP)', confusion_matrix[0, 1]],
        'Negativo': ['Falso Negativo (FN)', confusion_matrix[1, 0], 'Verdadeiro Negativo (VN)', confusion_matrix[0, 0]]
        })
        # Format the dataframe without column names and index
        confusion_table = confusion_table.style.set_table_styles([
            {'selector': 'th', 'props': 'display: none'},
            {'selector': 'td', 'props': 'padding: 0'}
        ])        
        
    mlflow.end_run()
    return result, accuracy, classification_report_df, confusion_table, roc_auc, precision, recall, f1, mcc    

################# COMPARE DATASETS IN THE SAME MODEL #####################
    
def evaluate_and_compare_datasets(mlflow_db, model_uri, target_col, production_df):
    from sklearn import metrics
    import mlflow
    import pandas as pd
    import numpy as np
    from plots_func import plot_predictions
    
    mlflow.set_tracking_uri(f"sqlite:///Models/{mlflow_db}.db") 

    model = mlflow.sklearn.load_model(model_uri)    
    train_run = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri.split('pycaret-model')[0]}base_train.parquet")
    test_run = mlflow.artifacts.download_artifacts(artifact_uri = f"{model_uri.split('pycaret-model')[0]}base_test.parquet")
    train_df = pd.read_parquet(train_run)
    test_df = pd.read_parquet(test_run)
    
    train_predictions = model.predict(train_df.drop(target_col, axis=1))    
    test_predictions = model.predict(test_df.drop(target_col, axis=1))
    
    train_probabilities = model.predict_proba(train_df.drop(target_col, axis=1))
    test_probabilities = model.predict_proba(test_df.drop(target_col, axis=1))

    train_predictions_class_1 = train_probabilities[:, 1]
    test_predictions_class_1 = test_probabilities[:, 1]
    train_predictions_class_0 = train_probabilities[:, 0]
    test_predictions_class_0 = test_probabilities[:, 0]
    
    train_report = metrics.classification_report(train_df[target_col], train_predictions, output_dict=True)
    report_train = pd.DataFrame(train_report)
    test_report = metrics.classification_report(test_df[target_col], test_predictions, output_dict=True)
    report_test = pd.DataFrame(test_report)
       
    if production_df is not False:
        prod_cols_to_keep = train_df.columns.tolist()
        production_df_clean = production_df[prod_cols_to_keep].dropna()
        
        # Make predictions and calculate metrics for production dataset
        production_predictions = model.predict(production_df_clean.drop(target_col, axis=1))
        production_probabilities = model.predict_proba(production_df_clean.drop(target_col, axis=1))
        production_predictions_class_1 = production_probabilities[:, 1]
        production_predictions_class_0 = production_probabilities[:, 0]
        production_report = metrics.classification_report(production_df_clean[target_col], production_predictions, output_dict=True) 
        report_prod = pd.DataFrame(production_report)
        
        plot_predictions(train_predictions, test_predictions, train_predictions_class_1, test_predictions_class_1, train_predictions_class_0, test_predictions_class_0, production_predictions_class_0, production_predictions_class_1, production_predictions)

        return report_train, report_test, report_prod
    else:
        plot_predictions(train_predictions, test_predictions, train_predictions_class_1, test_predictions_class_1, train_predictions_class_0, test_predictions_class_0, production_predictions_class_0=False, production_predictions_class_1=False, production_predictions = False)
    return report_train, report_test, None  

######################## SERVE PRODUCTION MODEL ###############################

def production_model_alvo(df_producao, columns, mlflow_db, target_col, model_uri):
    import os
    import pandas as pd
    import mlflow
    import streamlit as st
    from sklearn.metrics import accuracy_score, classification_report
    
    os.environ['MLFLOW_TRACKING_URI'] = f"sqlite:///Models/{mlflow_db}.db"
    
    # Carregamento do modelo
    model = mlflow.sklearn.load_model(model_uri)
    
    # Carregamento do conjunto de dados
    df = df_producao
    
    # Remoção da coluna alvo do conjunto de dados de produção
    df_producao = df.drop(columns=[target_col])
    
    # Previsão das classes
    classes = model.predict(df_producao)
    
    # Calcular a acurácia
    acuracia = accuracy_score(df[target_col], classes)
    print(f"Acurácia: {acuracia:.2f}")
    
    # Gerar o relatório de classificação
    report = classification_report(df[target_col], classes, output_dict=True)
    report = pd.DataFrame(report)
    
    # Imprimir o relatório de classificação
    print(classification_report(df[target_col], classes))

    return report, acuracia

############################################################################################################
def production_model(df_producao, mlflow_db, model_uri):
    import os
    import pandas as pd
    import mlflow
   
    os.environ['MLFLOW_TRACKING_URI'] = f"sqlite:///Models/{mlflow_db}.db"
   
    # Carregamento do modelo
    model = mlflow.sklearn.load_model(model_uri)
   
    # Previsão das classes no conjunto de dados de produção
    classes_producao = model.predict(df_producao)
   
    # Criar um DataFrame com as previsões
    df_predicoes = pd.DataFrame({'Previsoes': classes_producao})
   
    # Adicionar as colunas originais ao DataFrame de previsões
    df_result = pd.concat([df_producao, df_predicoes], axis=1)
    print(df_result)
   
    return df_result

    
############################################################################################################

def check_experiment_id(experiment_id):
    import mlflow
    experiment = mlflow.get_experiment_by_name(experiment_id)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_id)
        experiment = mlflow.get_experiment(experiment_id)
    experiment_id = experiment.experiment_id
    return experiment_id

def format_run_uri(model_uri):
    try:
        model_uri_parts = model_uri.split("/")  
        model_uri_parts.remove('artifacts')
        model_uri_relative = "/".join(model_uri_parts[-2:])  
        formatted_model_uri = f"runs:/{model_uri_relative}"  
    except:
        formatted_model_uri = "runs:/"
    return formatted_model_uri

def format_registered_model_uri():
    import streamlit as st
    try:
        model_name = st.sidebar.text_input("Nome do modelo:", value=st.session_state.experiment_name)
        alias = st.sidebar.text_input("Estado do Modelo(alias):", value="")
        formatted_model_uri = f"models:/{model_name}@{alias}"
    except:
        model_name, alias, formatted_model_uri = "", "", "models:/"
    return formatted_model_uri

def load_dataframe(file_to_load):
    import pandas as pd
    if file_to_load is not None:
        if file_to_load.name.endswith('.csv'):
            # Carrega o arquivo CSV
            df = pd.read_csv(file_to_load)
        elif file_to_load.name.endswith('.parquet'):
            # Carrega o arquivo Parquet
            df = pd.read_parquet(file_to_load)
        else:
            st.error("Formato de arquivo não suportado. Por favor, forneça um arquivo CSV ou Parquet.")
            st.stop()
        return df
    else:
        st.error("Nenhum arquivo fornecido.")
        st.stop() 