import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from fpdf import FPDF
import numpy as np
import os
import tempfile
import streamlit as st


def gerar_relatorio_pdf(parametro, modelo, X, y, image_id=None):
    y_pred = modelo.predict(X)
    erro_rmse = rmse(y, y_pred)

    # Criar scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y, y=y_pred, ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], '--r', label='1:1')
    ax.set_xlabel('Valor Observado')
    ax.set_ylabel('Valor Estimado')
    ax.set_title(f'{parametro}: Observado vs Estimado')
    ax.legend()

    # Salvar gráfico
    temp_dir = tempfile.mkdtemp()
    graph_path = os.path.join(temp_dir, f'{parametro}_scatter.png')
    plt.savefig(graph_path)
    plt.close()

    # Criar PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(
        200, 10, txt=f"Relatório de Estimativa: {parametro}", ln=True, align='C')
    pdf.ln(10)

    if image_id:
        pdf.cell(200, 10, txt=f"Imagem: {image_id}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt=f"RMSE: {erro_rmse:.4f}", ln=True)

    # Gráfico
    pdf.ln(10)
    pdf.image(graph_path, x=10, w=180)

    # Resumo estatístico
    pdf.add_page()
    pdf.set_font("Courier", size=8)
    summary_lines = modelo.summary().as_text().split('\n')
    for line in summary_lines:
        pdf.multi_cell(0, 4, txt=line)

    # PDF 
    pdf_path = os.path.join(temp_dir, f'relatorio_{parametro}.pdf')
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label=f"📄 Baixar Relatório {parametro}",
            data=f,
            file_name=f"relatorio_{parametro}.pdf",
            mime="application/pdf"
        )

