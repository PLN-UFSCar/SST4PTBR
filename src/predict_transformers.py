import os
import joblib
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import sys

# Caminhos dos arquivos salvos
MODELO_DIR = "modelo_finetunado_sarcasmo"
CLASSIFICADOR_PATH = os.path.join(MODELO_DIR, "classificador_logreg.pkl")


def carregar_modelo():
    if not os.path.exists(MODELO_DIR):
        raise FileNotFoundError(f"Diretório '{MODELO_DIR}' não encontrado.")
    if not os.path.exists(CLASSIFICADOR_PATH):
        raise FileNotFoundError(f"Classificador '{CLASSIFICADOR_PATH}' não encontrado.")

    print("[INFO] Carregando modelo e classificador...")
    modelo = SentenceTransformer(MODELO_DIR)
    classificador = joblib.load(CLASSIFICADOR_PATH)
    return modelo, classificador


def prever_ironia(frase, modelo, classificador, limiar=0.5):
    embedding = modelo.encode([frase], convert_to_tensor=True).cpu().numpy()
    prob = classificador.predict_proba(embedding)[0][1]  # Probabilidade de ironia

    if prob >= limiar:
        return "Sarcasmo detectado", prob
    else:
        return "Sarcasmo não detectado", prob


def main():
    modelo, classificador = carregar_modelo()

    print("\nDigite uma frase para detectar sarcasmo (ou 'sair' para encerrar):")
    while True:
        frase = input("\n> ")
        if frase.strip().lower() == 'sair':
            break
        if len(frase.strip()) == 0:
            print("[ERRO] Frase vazia. Tente novamente.")
            continue

        resultado, prob = prever_ironia(frase, modelo, classificador)
        print(f"{resultado} (confiança: {prob:.2f})")


if __name__ == "__main__":
    main()
