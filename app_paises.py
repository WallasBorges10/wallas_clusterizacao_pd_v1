import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
import os
import tempfile

# Configuração da página
st.set_page_config(
    page_title="Clusterização de Países",
    page_icon="🌍",
    layout="wide"
)

# Título da aplicação
st.title("🌍 Análise de Clusterização de Países")
st.markdown("---")

# Função para carregar dados diretamente do Kaggle
@st.cache_data(show_spinner="Carregando dados do Kaggle...")
def load_country_data_directly():
    """Carrega os dados diretamente do Kaggle"""
    try:
        # Download do dataset diretamente
        with st.spinner("Baixando dataset do Kaggle..."):
            path = kagglehub.dataset_download("rohan0301/unsupervised-learning-on-country-data")
        
        # Listar arquivos no diretório
        files = os.listdir(path)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if not csv_files:
            st.error("Nenhum arquivo CSV encontrado no dataset.")
            return None
        
        # Carregar o primeiro arquivo CSV encontrado
        file_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(file_path)
        
        st.success(f"Dataset carregado com sucesso! {len(df.country.unique())} países encontrados.")
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados do Kaggle: {str(e)}")
        st.info("""
        **Solução de problemas:**
        - Verifique se está conectado à internet
        - Certifique-se de que a biblioteca kagglehub está instalada
        - Tente novamente em alguns instantes
        """)
        return None

# Sidebar para configurações
st.sidebar.header("Configurações")

# Botão para carregar dados
if st.sidebar.button("🔄 Carregar Dados do Kaggle"):
    st.cache_data.clear()
    st.rerun()

# Parâmetros de clusterização
st.sidebar.subheader("Parâmetros de Clusterização")
algorithm = st.sidebar.selectbox(
    "Algoritmo",
    ["K-Means", "DBSCAN"]
)

if algorithm == "K-Means":
    n_clusters = st.sidebar.slider(
        "Número de Clusters",
        min_value=2,
        max_value=8,
        value=3
    )
else:
    eps = st.sidebar.slider("EPS", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)

# Carregar dados
df = load_country_data_directly()

if df is not None:
    # Mostrar informações básicas
    st.header("📋 Visualização dos Dados dos Países")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Primeiras Linhas")
        st.dataframe(df.head(10))

    with col2:
        st.subheader("Informações do Dataset")
        st.write(f"**Formato:** {df.shape[0]} países × {df.shape[1]} características")
        st.write(f"**Colunas:** {list(df.columns)}")
        
        # Estatísticas básicas
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df.describe())

    # Seleção de features
    st.subheader("🔧 Seleção de Features para Clusterização")

    # Colunas numéricas (excluindo a coluna do país se existir)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'country' in df.columns:
        country_column = 'country'
    else:
        country_column = None

    # Features padrão baseadas no dataset típico de países
    default_features = []
    possible_features = ['child_mort', 'exports', 'health', 'imports', 
                        'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
    
    for feature in possible_features:
        if feature in numeric_columns:
            default_features.append(feature)
            if len(default_features) >= 4:
                break

    if not default_features and len(numeric_columns) >= 2:
        default_features = numeric_columns[:min(4, len(numeric_columns))]

    selected_features = st.multiselect(
        "Selecione as características econômicas para clusterização:",
        numeric_columns,
        default=default_features
    )

    if len(selected_features) >= 2:
        # Preparar dados
        X = df[selected_features].copy()
        
        # Tratar valores missing
        if X.isnull().sum().sum() > 0:
            st.warning(f"Encontrados {X.isnull().sum().sum()} valores missing. Preenchendo com a mediana.")
            X = X.fillna(X.median())
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar clusterização
        if algorithm == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            df_clustered = df.copy()
            df_clustered['Cluster'] = labels
            df_clustered['Cluster'] = df_clustered['Cluster'].astype(str)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            df_clustered = df.copy()
            df_clustered['Cluster'] = labels
            df_clustered['Cluster'] = df_clustered['Cluster'].astype(str)
        
        # Visualizações
        st.header("📈 Resultados da Clusterização")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        unique_clusters = len(np.unique(labels))
        with col1:
            st.metric("Número de Clusters", unique_clusters)
        
        if unique_clusters > 1 and algorithm != "DBSCAN":
            try:
                silhouette_avg = silhouette_score(X_scaled, labels)
                with col2:
                    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            except:
                pass
        
        with col3:
            st.metric("Países Analisados", len(df))
        
        # PCA para visualização 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Criar DataFrame para visualização
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels.astype(str)
        })
        
        if country_column:
            viz_df['country'] = df[country_column]
        
        # Gráfico de clusters
        if country_column:
            fig1 = px.scatter(
                viz_df, x='PC1', y='PC2', color='Cluster',
                hover_data=['country'],
                title='Visualização dos Clusters de Países (PCA)',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:
            fig1 = px.scatter(
                viz_df, x='PC1', y='PC2', color='Cluster',
                title='Visualização dos Clusters de Países (PCA)',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        
        # Gráfico de distribuição por cluster
        melted_df = df_clustered.melt(
            id_vars=['Cluster'] + ([country_column] if country_column else []),
            value_vars=selected_features,
            var_name='Feature',
            value_name='Valor'
        )
        
        fig2 = px.box(
            melted_df,
            x='Cluster', y='Valor', color='Feature',
            title='Distribuição das Características por Cluster'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Análise dos clusters
        st.subheader("🔍 Análise Detalhada dos Clusters")
        
        # Estatísticas por cluster
        cluster_stats = df_clustered.groupby('Cluster')[selected_features].mean()
        st.write("**Médias das Características por Cluster:**")
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        
        # Países por cluster
        st.write("**Distribuição de Países por Cluster:**")
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        fig3 = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            title='Número de Países por Cluster',
            labels={'x': 'Cluster', 'y': 'Número de Países'},
            color=cluster_counts.index.astype(str)
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Tabela interativa com países
        st.subheader("📊 Tabela de Países e seus Clusters")
        
        # Ordenar por cluster
        display_df = df_clustered.copy()
        if country_column:
            display_df = display_df.sort_values(['Cluster', country_column])
            st.dataframe(display_df[[country_column, 'Cluster'] + selected_features])
        else:
            display_df = display_df.sort_values('Cluster')
            st.dataframe(display_df[['Cluster'] + selected_features])
        
        # Download dos resultados
        st.subheader("💾 Download dos Resultados")
        
        csv = df_clustered.to_csv(index=False)
        st.download_button(
            label="Baixar dados clusterizados (CSV)",
            data=csv,
            file_name="paises_clusterizados.csv",
            mime="text/csv"
        )
        
        # Interpretação dos resultados
        st.subheader("🎯 Interpretação dos Clusters")
        
        st.markdown("""
        **Possíveis interpretações:**
        - **Clusters com alta renda e PIB**: Países desenvolvidos
        - **Clusters com baixa renda e alta mortalidade infantil**: Países em desenvolvimento
        - **Clusters intermediários**: Países em transição econômica
        
        *Analise as estatísticas de cada cluster para entender melhor os padrões.*
        """)
        
    else:
        st.warning("Selecione pelo menos 2 características para clusterização")

else:
    st.error("Não foi possível carregar os dados. Verifique sua conexão com a internet.")
    
    # Mostrar instruções de instalação
    st.info("""
    **Se estiver executando localmente, certifique-se de que:**
    
    1. A biblioteca kagglehub está instalada:
    ```bash
    pip install kagglehub
    ```
    
    2. Você tem conexão com a internet
    
    3. O dataset ainda está disponível no Kaggle
    """)

# Informações sobre o dataset
st.sidebar.markdown("---")
st.sidebar.subheader("Sobre o Dataset")
st.sidebar.info("""
**Dataset: Country Data**
- Dados socioeconômicos de vários países
- Inclui indicadores como:
  - Mortalidade infantil
  - Renda per capita
  - PIB
  - Expectativa de vida
  - Entre outros
""")

# Rodapé
st.markdown("---")
st.markdown(
    "Desenvolvido com Streamlit | "
    "Análise de Clusterização de Dados de Países | "
    "Dados carregados diretamente do Kaggle"
)