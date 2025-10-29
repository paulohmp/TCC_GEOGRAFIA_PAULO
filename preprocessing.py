# PRÉ-PROCESSAMENTO
import io
import os
import tempfile
import zipfile
import streamlit as st
import ee
import geopandas as gpd
import geemap


def load_shapefile_from_zip(uploaded_file):
    try:
        # Verificar se o arquivo é um .zip
        if not uploaded_file.name.lower().endswith('.zip'):
            st.error("Por favor, envie um arquivo .zip contendo o Shapefile.")
            return None

        # Extrair o conteúdo do arquivo .zip
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            # Encontrar o arquivo .shp
            shp_files = [f for f in os.listdir(
                tmp_dir) if f.lower().endswith('.shp')]
            if not shp_files:
                st.error("Nenhum arquivo .shp encontrado no .zip.")
                return None

            # Carregar o Shapefile usando o geopandas
            shp_path = os.path.join(tmp_dir, shp_files[0])
            gdf = gpd.read_file(shp_path)

            # Verificar se o GeoDataFrame foi carregado corretamente
            if gdf.empty:
                st.error("O Shapefile está vazio ou não pôde ser lido.")
                return None
            return gdf
    except Exception as e:
        st.error(f"Erro ao carregar o Shapefile: {e}")
        return None


# Função de nuvens e fator de escala
def mask_cloud_and_shadows_sr(image):
    cloud_prob = image.select('MSK_CLDPRB')
    snow_prob = image.select('MSK_SNWPRB')
    cloud = cloud_prob.lt(5)
    snow = snow_prob.lt(5)
    scl = image.select('SCL')
    shadow = scl.eq(3)  # 3 = cloud shadow
    cirrus = scl.eq(10)  # 10 = cirrus
    mask = (cloud.And(snow)).And(cirrus.neq(1)).And(shadow.neq(1))
    return image.updateMask(mask).divide(10000).select("B.*").copyProperties(image, image.propertyNames())


# ------------------------- Funções de Exportação -------------------------
def export_image(image, roi):
    try:
        if image.bandNames().size().getInfo() == 0:
            st.sidebar.error("Erro: A imagem selecionada não contém bandas.")
            return
        roi = roi.geometry().buffer(10000).bounds()
        url = image.getDownloadURL({
            'name': 'image_export',
            'scale': 20,
            'crs': 'EPSG:4674',
            'region': roi,
            'format': 'GEO_TIFF'
        })
        st.sidebar.success(
            f"Imagem exportada com sucesso. [Clique para baixar]({url})")
    except Exception as e:
        if 'Total request size' in str(e):
            st.sidebar.warning(
                "Imagem muito grande, exportando em partes (tiles)...")
            export_image_by_tiles(image, roi)
        else:
            st.sidebar.error(f"Erro ao exportar imagem: {str(e)}")


def export_image_by_tiles(image, roi, tile_size=0.05):
    """Exportar imagem dividida em tiles menores para evitar exceder o limite de 50MB."""
    grid = geemap.fishnet(roi, rows=5, cols=5)
    for idx, feature in enumerate(grid.getInfo()['features']):
        tile_geometry = ee.Feature(feature).geometry()
        try:
            if image.bandNames().size().getInfo() == 0:
                st.sidebar.warning(
                    f"Tile {idx+1} não possui bandas e será ignorado.")
                continue

            url = image.getDownloadURL({
                'name': f'image_export_tile_{idx+1}',
                'scale': 20,
                'crs': 'EPSG:4674',
                'region': tile_geometry,
                'format': 'GEO_TIFF'
            })
            st.sidebar.success(
                f"Tile {idx+1} exportado com sucesso. [Clique para baixar]({url})")
        except Exception as e:
            st.sidebar.error(f"Erro no tile {idx+1}: {str(e)}")
