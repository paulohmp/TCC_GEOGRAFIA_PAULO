import streamlit as st
import geemap.foliumap as geemap
import ee
import pandas as pd
import numpy as np
import json
from datetime import datetime
import statsmodels.api as sm
import branca.colormap as cm

# Funções externas
from preprocessing import mask_cloud_and_shadows_sr, load_shapefile_from_zip, export_image
from prediction_model import calcular_todos_os_modelos_turbidez, calcular_todos_os_modelos_chla, calcular_todos_os_modelos_tss, equacao_bandas, aplicar_modelo_na_imagem
from report import gerar_relatorio_pdf

# Configuração da página
st.set_page_config(layout="wide")

st.markdown("""
### 🛰️ QASat
Esta aplicação foi desenvolvida como parte do **Trabalho de Conclusão de Curso (TCC)** de graduação em Geografia da Universidade Federal de Minas Gerais (UFMG).  
Seu objetivo é demonstrar o potencial do **sensoriamento remoto** e do **processamento em nuvem** na análise da qualidade da água em ambientes continentais, integrando dados de satélite, modelagem estatística e visualização interativa.

A plataforma permite **automatizar o fluxo de monitoramento** de parâmetros limnológicos (como turbidez, clorofila-a e sólidos suspensos totais) a partir de imagens do satélite **Sentinel-2**.  
O sistema integra três etapas principais:

1. **Entrada de dados:** upload de um arquivo `.zip` contendo o shapefile da área de estudo, com coordenadas e valores observados *in situ*.  
2. **Processamento:** extração automática das reflectâncias das bandas espectrais correspondentes no Google Earth Engine, cálculo de índices espectrais e ajuste de modelos estatísticos de regressão.  
3. **Saída de resultados:** geração de relatórios em PDF, tabelas CSV e raster em formato GeoTIFF, além de visualização interativa no mapa.
""")

# Autenticação Earth Engine
service_account = 'scriptspaulo@ee-scriptspaulo.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(
    service_account, r'C:\Users\Paulo\Desktop\UFMG\TCC\ee-scriptspaulo-124cc67658c5.json')
ee.Initialize(credentials)

# Inicializar mapa
m = geemap.Map()
roi = None

# Upload shapefile
uploaded_file_roi = st.sidebar.file_uploader(
    "Upload da área de estudo (.zip)", type=['zip'])

if uploaded_file_roi:
    gdf = load_shapefile_from_zip(uploaded_file_roi)
    if gdf is not None:
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf

        # Validar colunas
        parameters = []
        for param in ['CHLA', 'TURBIDEZ', 'TSS']:
            if param in gdf.columns:
                parameters.append(param)
                gdf[param] = gdf[param].astype(
                    str).str.replace(',', '.').astype(float)

        if parameters:
            st.sidebar.success(
                f"Parâmetros encontrados: {', '.join(parameters)}")
        else:
            st.sidebar.warning(
                "Nenhum parâmetro de qualidade da água encontrado.")

        if 'date' not in gdf.columns:
            st.error(
                "O shapefile precisa conter a coluna 'date' no formato YYYY-MM-DD.")
        else:
            gdf['date'] = pd.to_datetime(gdf['date'])
            datas_unicas = gdf['date'].dt.date.unique()
            dados = []

            for data in datas_unicas:
                pontos_data = gdf[gdf['date'].dt.date == data].copy()
                pontos_data = pontos_data.drop(columns='date')
                pontos_ee = ee.FeatureCollection(
                    json.loads(pontos_data.to_json())['features'])

                imagem = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(pontos_ee) \
                    .filterDate(str(data), str(data + pd.Timedelta(days=1))) \
                    .map(mask_cloud_and_shadows_sr) \
                    .first()

                if not imagem:
                    st.warning(f"Sem imagem Sentinel-2 para {data}.")
                    continue

                bandas = ['B2', 'B3', 'B4', 'B5',
                          'B6', 'B8', 'B8A', 'B11', 'B12']
                imagem = imagem.select(bandas)

                reflectance = imagem.sampleRegions(
                    collection=pontos_ee, scale=10, geometries=True)
                features = reflectance.getInfo()['features']

                for f in features:
                    props = f['properties']
                    latlon = f['geometry']['coordinates']
                    row = {b: props.get(b) for b in bandas}
                    row.update({p: props.get(p) for p in parameters})
                    row.update(
                        {'lon': latlon[0], 'lat': latlon[1], 'date': str(data)})
                    dados.append(row)

            if dados:
                df_ref = pd.DataFrame(dados)
                st.success("Dados espectrais extraídos com sucesso!")

                for parametro in parameters:
                    if parametro == 'TURBIDEZ':
                        df_param = calcular_todos_os_modelos_turbidez(
                            df_ref.copy())
                        preditores = ['B2', 'B3', 'B4', 'B5', 'B8', 'Turb1',
                                      'Turb2', 'Turb3', 'Turb4', 'Turb5', 'Turb6']
                    elif parametro == 'CHLA':
                        df_param = calcular_todos_os_modelos_chla(
                            df_ref.copy())
                        preditores = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8',
                                      'B8A', 'Chla1', 'Chla2', 'Chla3', 'Chla4', 'Chla5']
                    elif parametro == 'TSS':
                        df_param = calcular_todos_os_modelos_tss(df_ref.copy())
                        preditores = ['B3', 'B4', 'B5', 'B6', 'B8',
                                      'B11', 'TSS1', 'TSS2', 'TSS3', 'TSS4']
                    else:
                        continue

                    p_limite = st.sidebar.slider(
                        "Limite de significância (p-valor)", 0.01, 0.1, 0.05, step=0.01)

                    y = df_param[parametro]  # Variável dependente
                    X = df_param.drop(
                        columns=[parametro, 'lat', 'lon', 'date'], errors='ignore')
                    # st.write(X)
                    #X = sm.add_constant(X)
                    # st.write(X)
                    # st.write(y)

                    modelo_inicial = sm.OLS(y, X).fit()
                    # st.write(modelo_inicial.summary())
                    p_valores = modelo_inicial.pvalues
                    # st.write(p_valores)
                    significativos = p_valores[p_valores <
                                               p_limite].index.tolist()

                    X_significativos = X[significativos]
                    # st.write(X_significativos)
                    modelo_final = sm.OLS(y, X_significativos).fit()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Modelo Inicial")
                        st.markdown(
                            f"```text\n{modelo_inicial.summary()}\n```")

                    with col2:
                        st.subheader("Modelo Final")
                        st.markdown(f"```text\n{modelo_final.summary()}\n```")

            else:
                st.warning("Falha na extração dos dados espectrais.")

        if 'date' in gdf.columns:
            gdf = gdf.drop(columns='date')
        shp_json = gdf.to_json()
        f_json = json.loads(shp_json)['features']
        roi = ee.FeatureCollection(f_json)
    else:
        st.sidebar.error("Erro ao carregar shapefile.")

point = ee.Geometry.Point(-43.97766, -19.85118)
m.centerObject(point, zoom=15)
m.setOptions("HYBRID")

st.session_state.setdefault('lista_carregada', False)
st.session_state.setdefault('imagem_carregada', False)
st.session_state.setdefault('collection', None)


if 'df_ref' in locals() and not df_ref.empty:
    st.sidebar.markdown("### Configurações de imagem Sentinel-2")
    user_max_cloud_coverage = st.sidebar.slider("Nuvem máxima (%)", 0, 100, 5)
    start_date = st.sidebar.date_input("Data inicial", datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("Data final", datetime.now())
    user_mndwi = st.sidebar.number_input(
        "Limiar MNDWI", -1.0, 1.0, 0.0, step=0.1)

    if st.sidebar.button("Listar imagens disponíveis"):
        st.session_state['lista_carregada'] = True
        st.session_state['imagem_carregada'] = False

    if st.sidebar.button("Resetar"):
        st.session_state['lista_carregada'] = False
        st.session_state['imagem_carregada'] = False
        st.session_state['collection'] = None

    if st.session_state['lista_carregada']:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate(str(start_date), str(end_date)) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', user_max_cloud_coverage)) \
            .map(mask_cloud_and_shadows_sr)

        st.session_state['collection'] = collection

        info_df = pd.DataFrame({
            'ID': collection.aggregate_array('system:index').getInfo(),
            'Data': collection.aggregate_array('system:time_start').map(
                lambda t: ee.Date(t).format('YYYY-MM-dd')).getInfo(),
            '% Nuvem': collection.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        })

        selected_row = st.selectbox(
            "Imagem:",
            info_df.apply(
                lambda row: f"{row['Data']} | % Nebulosidade: {row['% Nuvem']}%", axis=1)
        )
        with st.columns(3)[1]:
            if st.button("Carregar imagem selecionada"):
                selected_id = info_df.loc[info_df.apply(
                    lambda row: f"{row['Data']} | % Nebulosidade: {row['% Nuvem']}%", axis=1) == selected_row, 'ID'].values[0]

                st.session_state['selected_id'] = selected_id
                st.session_state['imagem_carregada'] = True

    if st.session_state['imagem_carregada']:
        image = ee.Image(st.session_state['collection']
                         .filter(ee.Filter.eq('system:index', st.session_state['selected_id']))
                         .first())

        bandas = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B8A', 'B11', 'B12']
        image = image.select(bandas)
        image = equacao_bandas(image, parameters)
        # st.write(image.bandNames().getInfo())

        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        mask = mndwi.gte(user_mndwi)
        mndwi_masked = mndwi.updateMask(mask)

        m.addLayer(image, {'bands': ['B4', 'B3', 'B2'],
                   'min': 0, 'max': 0.2}, 'Imagem RGB')
        m.addLayer(mndwi_masked, {'palette': [
                   'white', 'blue'], 'min': 0, 'max': 1}, 'MNDWI')

        params = modelo_final.params
        # st.write(params)
        coeficientes = params.to_dict()
        st.write(coeficientes)

        estimativa = aplicar_modelo_na_imagem(
            preditores=list(coeficientes.keys()),
            coeficientes=coeficientes,
            image=image
        )

        estimativa = estimativa.where(estimativa.lt(0), 0)
        estimativa = estimativa.updateMask(mask).rename("estimativa")

        vis_params = {'min': 0, 'max': 5, 'palette': [
            'blue', 'cyan', 'green', 'yellow', 'red']}

        m.addLayer(estimativa, vis_params, f"{parametro} Estimado")
        unidade = {
            "TURBIDEZ": "NTU",
            "CHLA": "µg/L",
            "TSS": "mg/L"
        }.get(parametro, "")

        # Cria rampa usando a paleta e limites já definidos
        colormap = cm.LinearColormap(
            colors=vis_params['palette'],
            vmin=vis_params['min'],
            vmax=vis_params['max']
        )

        # Define o texto exibido na legenda
        colormap.caption = f"{parametro} estimado {f'({unidade})' if unidade else ''}"

        # Adiciona ao mapa interativo
        colormap.add_to(m)

        m.addLayer(roi, {'color': 'yellow'}, 'Pontos de coleta')
        m.centerObject(roi)

        with st.columns(3)[1]:
            gerar_relatorio_pdf(
                parametro=parametro,
                modelo=modelo_final,
                X=X[modelo_final.params.index],
                y=df_param[parametro],
                image_id=image.get('system:index').getInfo()
            )

            st.download_button(
                label="Baixar dados no formato CSV",
                data=df_param.to_csv(index=False).encode('utf-8'),
                file_name=f"{parametro}.csv",
                mime="text/csv"
            )

        with st.sidebar:
            if st.button("Download Estimativa"):
                with st.spinner("Baixando.."):
                    export_image(estimativa, roi)


st.sidebar.markdown("""
*Desenvolvido por [Paulo Henrique Maciel Pádua](https://www.linkedin.com/in/paulo-padua/)*  
*Universidade Federal de Minas Gerais — Instituto de Geociências*  
*Trabalho de Conclusão de Curso (2025)*  
""")

m.to_streamlit()
