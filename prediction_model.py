import ee
import streamlit as st
# Modelos de predição

# TURBIDEZ


def calcular_turb1(dados):
    dados['Turb1'] = (dados['B2'] + dados['B8']) * dados['B8']
    return dados


def calcular_turb2(dados, alpha=0.419, beta1=-94.129, beta2=56.261, beta3=135.372, beta4=-110.431):
    dados['Turb2'] = alpha + beta1 * dados['B2'] + beta2 * \
        dados['B3'] + beta3 * dados['B4'] + beta4 * dados['B8']
    return dados


def calcular_turb3(dados):
    dados['Turb3'] = (dados['B5'] - dados['B11']) + \
        (dados['B2'] / dados['B12'])
    return dados


def calcular_turb4(dados):
    dados['Turb4'] = (dados['B8'] * dados['B4']) / dados['B3']
    return dados


def calcular_turb5(dados):
    dados['Turb5'] = (dados['B4'] - dados['B3']) / (dados['B4'] + dados['B3'])
    return dados


def calcular_turb6(dados):
    dados['Turb6'] = dados['B4'] / dados['B2']
    return dados

# Função geral para aplicar todos


def calcular_todos_os_modelos_turbidez(dados):
    dados = calcular_turb1(dados)
    dados = calcular_turb2(dados)
    dados = calcular_turb3(dados)
    dados = calcular_turb4(dados)
    dados = calcular_turb5(dados)
    dados = calcular_turb6(dados)
    return dados


# CLORIFILA
def calcular_chla1(dados):
    dados['Chla1'] = dados['B4'] / dados['B8']
    return dados


def calcular_chla2(dados):
    dados['Chla2'] = dados['B2'] / dados['B3']
    return dados


def calcular_chla3(dados):
    dados['Chla3'] = dados['B5'] - 1.005 + ((dados['B6'] - dados['B4']) * (
        dados['B5'] - dados['B4'])) / (dados['B6'] - dados['B4'] + dados['B4'])
    return dados


def calcular_chla4(dados):
    dados['Chla4'] = (dados['B5'] - dados['B4']) + ((dados['B8A'] - dados['B4'])
                                                    * (dados['B5'] - dados['B4'])) / (dados['B8A'] - dados['B4'])
    return dados


def calcular_chla5(dados):
    dados['Chla5'] = dados['B5'] / dados['B4']
    return dados


def calcular_chla6(dados):
    dados['Chla6'] = (dados['B5'] + dados['B6']) / dados['B4']
    return dados


def calcular_chla7(dados):
    dados['Chla7'] = dados['B5'] - ((dados['B4'] + dados['B6']) / 2)
    return dados


def calcular_todos_os_modelos_chla(dados):
    dados = calcular_chla1(dados)
    dados = calcular_chla2(dados)
    dados = calcular_chla3(dados)
    dados = calcular_chla4(dados)
    dados = calcular_chla5(dados)
    dados = calcular_chla6(dados)
    dados = calcular_chla7(dados)
    return dados

# TSS


def calcular_tss1(dados):
    dados['TSS1'] = dados['B8'] + dados['B4']
    return dados


def calcular_tss2(dados):
    dados['TSS2'] = (dados['B3'] + dados['B4']) / (dados['B8'] + dados['B11'])
    return dados


def calcular_tss3(dados):
    dados['TSS3'] = dados['B6'] - dados['B5']
    return dados


def calcular_tss4(dados):
    dados['TSS4'] = dados['B6'] - ((dados['B6'] + dados['B4']) / 2)
    return dados


def calcular_todos_os_modelos_tss(dados):
    dados = calcular_tss1(dados)
    dados = calcular_tss2(dados)
    dados = calcular_tss3(dados)
    dados = calcular_tss4(dados)
    return dados


# Cria as imagens a partir das equações e adiciona como bandas a imagem
def equacao_bandas(image, parametros):
    expressoes_por_parametro = {
        'TURBIDEZ': {
            'Turb1': '(B2 + B8) * B8',
            'Turb2': '(-8753 * (B8 / B2)) + (5223 * B8) + 2552',
            'Turb3': '(B5 - B11) + (B2 / B12)',
            'Turb4': '(B8 * B4) / B3',
            'Turb5': '(B4 - B3) / (B4 + B3)',
            'Turb6': 'B4 / B2',
        },
        'CHLA': {
            'Chla1': 'B4 / B8',
            'Chla2': 'B2 / B3',
            'Chla3': 'B5 - 1.005 + ((B6 - B4) * (B5 - B4)) / (B6 - B4 + B4)',
            'Chla4': '(B5 - B4) + ((B8A - B4) * (B5 - B4)) / (B8A - B4)',
            'Chla5': 'B5 / B4',
            'Chla6': '(B5 + B6) / B4',
            'Chla7': 'B5 - ((B4 + B6) / 2)',
        },
        'TSS': {
            'TSS1': 'B8 + B4',
            'TSS2': '(B3 + B4) / (B8 + B11)',
            'TSS3': 'B6 - B5',
            'TSS4': 'B6 - ((B6 + B4) / 2)',
        }
    }

    # Garante lista
    if isinstance(parametros, str):
        parametros = [parametros]

    bandas_expressas = []

    for parametro in parametros:
        expressoes = expressoes_por_parametro.get(parametro.upper())
        if not expressoes:
            continue  # ignora parâmetros inválidos

        for nome, formula in expressoes.items():
            banda_expr = image.expression(formula, {
                'B2': image.select('B2'),
                'B3': image.select('B3'),
                'B4': image.select('B4'),
                'B5': image.select('B5'),
                'B6': image.select('B6'),
                'B8': image.select('B8'),
                'B8A': image.select('B8A'),
                'B11': image.select('B11'),
                'B12': image.select('B12')
            }).rename(nome)
            bandas_expressas.append(banda_expr)

    if bandas_expressas:
        return image.addBands(ee.Image.cat(bandas_expressas))
    else:
        return image


# Aplica a regressão na imagem
def aplicar_modelo_na_imagem(preditores, coeficientes, image):
    #st.write("Coeficientes usados:", coeficientes)

    resultado = None  # Começa sem imagem base

    for pred in preditores:
        coef = coeficientes.get(pred)
        if coef is not None:
            #st.write(f"\nProcessando preditor: {pred} (coef = {coef})")

            if pred not in image.bandNames().getInfo():
                st.write(f"Erro: Banda {pred} não encontrada na imagem!")
                continue

            banda = image.select(pred).multiply(coef)

            resultado = banda if resultado is None else resultado.add(banda)

            # Verificação intermediária
            sample = resultado.sample(
                region=image.geometry(), scale=10, numPixels=1)
            #st.write(f"Valor parcial após {pred}:",
                     #sample.first().getInfo()['properties'])

    if resultado is not None:
        resultado = resultado.rename('estimativa')

        # Verificação final
        sample_final = resultado.sample(
            region=image.geometry(), scale=10, numPixels=1)
        #st.write("Valor final:", sample_final.first().getInfo()
                 #['properties']['estimativa'])

    return resultado
