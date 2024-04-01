def generate_plot_eda(df_clean, selected_columns, plot_type):
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    try:
        # Geração de gráficos
        if plot_type == "Histogram": 
            st.subheader("Histogram")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns)
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df_clean, x=x_attr, kde=False)
            st.pyplot()

        elif plot_type == "Violin Plot": 
            st.subheader("Violin Plot")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            plt.figure(figsize=(8, 6))
            violinplot = sns.violinplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr)
            violinplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)  
            st.pyplot()

        elif plot_type == "Scatter Plot": 
            st.subheader("Scatter Plot")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr, size=hue_attr, sizes=(50, 200))
            st.pyplot()

        elif plot_type == "Gráfico de Linha": #-------------
            st.subheader("Gráfico de Linha")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=0)
            plt.figure(figsize=(8, 6))
            sns.lineplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr)
            st.pyplot()

        elif plot_type == "Boxplot": 
            st.subheader("Boxplot")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns, index=None)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            plt.figure(figsize=(8, 6))
            boxplot = sns.boxplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr, native_scale=False, gap=.1)
            boxplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)  
            st.pyplot()

        elif plot_type == "Density(KDE)": 
            st.subheader("Density(KDE)")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns, index=None)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            plt.figure(figsize=(8, 6))
            sns.kdeplot(data=df_clean, x=x_attr, hue=hue_attr, common_norm=True, fill=True, palette="crest", alpha=.5, linewidth=2, legend=True, multiple="stack")
            st.pyplot()

        elif plot_type == "Distribuição": 
            st.subheader("Distribuição")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns, index=None)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns, index=None)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            col_attr = st.sidebar.selectbox("Selecionar coluna para 'col':", options=selected_columns, index=None)
            kind = st.sidebar.selectbox("Selecionar tipo de gráfico ('kind'):", options=["hist", "kde", ], index=0)
            plt.figure(figsize=(8, 6))
            sns.displot(data=df_clean, x=x_attr, hue=hue_attr, col=col_attr, kind=kind, common_norm=True, fill=True, linewidth=2, palette="crest", alpha=.7, multiple="stack", )
            st.pyplot()

        elif plot_type == "Relacional": 
            st.subheader("Relacional")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns, index=None)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns, index=None)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            col_attr = st.sidebar.selectbox("Selecionar coluna para 'col':", options=selected_columns, index=None)
            kind = st.sidebar.selectbox("Selecionar tipo de gráfico ('kind'):", options=["scatter", "line"], index=0)
            plt.figure(figsize=(8, 6))
            sns.relplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr, size=hue_attr, kind=kind, col=col_attr)
            st.pyplot()

        elif plot_type == "Categorias": 
            st.subheader("Categorias")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns, index=0)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns, index=None)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            kind = st.sidebar.selectbox("Selecionar tipo de gráfico ('kind'):", options=["strip", "box", "violin", "boxen", "point", "bar", "count"], index=0)
            plt.figure(figsize=(8, 6))
            sns.catplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr, kind=kind)
            st.pyplot()

        elif plot_type == "Heatmap": #
            st.subheader("Heatmap")
            plt.figure(figsize=(8, 6))
            sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
            st.pyplot()

        elif plot_type == "Joinplot": #
            st.subheader("Joinplot")
            x_attr = st.sidebar.selectbox("Selecionar coluna para eixo x:", options=selected_columns)
            y_attr = st.sidebar.selectbox("Selecionar coluna para eixo y:", options=selected_columns, index=0)
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=0)
            plt.figure(figsize=(8, 6))
            jointplot = sns.jointplot(data=df_clean, x=x_attr, y=y_attr, hue=hue_attr)
            jointplot.ax_joint.legend_.remove() 
            st.pyplot()

        elif plot_type == "Pairplot": #
            st.subheader("Pairplot")
            hue_attr = st.sidebar.selectbox("Selecionar coluna para 'hue':", options=selected_columns, index=None)
            diag_kind = st.sidebar.selectbox("Selecionar plots diagonais(diag_kind):", options=["auto", "hist", "kde"], index=0)
            plt.figure(figsize=(8, 6))
            sns.pairplot(df_clean, hue=hue_attr, diag_kind=diag_kind)
            st.pyplot()
    except Exception as e:
        st.error(f"Ocorreu um erro. Verifque os atributos para plotagem: {e}")