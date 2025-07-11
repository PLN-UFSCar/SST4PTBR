import sys
import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
import joblib

OUTPUT_DIR = "modelo_finetunado_sarcasmo"


def sentence_transformer(df, modelo, salvar_em=OUTPUT_DIR):
    df = df.dropna(subset=['text', 'is_sarcastic'])
    df['text'] = df['text'].apply(lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))
    df['headline'] = df['headline'].apply(lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens))
    df['is_sarcastic'] = df['is_sarcastic'].astype(int)

    # Limite opcional de tamanho do texto
    df['text'] = df['text'].apply(lambda x: x[:512])

    print('  Separando dados em treino e teste...')
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], random_state=42)

    print('  Preparando datasets...')
    train_ds = Dataset.from_pandas(train_df.rename(columns={'text': 'text', 'is_sarcastic': 'label'}), preserve_index=False)
    eval_ds = Dataset.from_pandas(test_df.rename(columns={'text': 'text', 'is_sarcastic': 'label'}), preserve_index=False)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    training_args = SentenceTransformerTrainingArguments(
        output_dir=salvar_em,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        logging_steps=10,
        save_total_limit=1,
        learning_rate=2e-5,
        warmup_steps=10,
        fp16=False
    )

    train_loss = losses.SoftmaxLoss(
        model=modelo,
        sentence_embedding_dimension=modelo.get_sentence_embedding_dimension(),
        num_labels=2
    )

    trainer = SentenceTransformerTrainer(
        model=modelo,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=train_loss,
        compute_metrics=compute_metrics
    )

    print('  Iniciando treinamento...')
    trainer.train()
    print('  Treinamento finalizado.')

    print('  Gerando embeddings para os conjuntos de treino e teste...')
    X_train = modelo.encode(train_df['text'].tolist(), convert_to_tensor=True).cpu().numpy()
    X_test = modelo.encode(test_df['text'].tolist(), convert_to_tensor=True).cpu().numpy()
    y_train = train_df['is_sarcastic'].values
    y_test = test_df['is_sarcastic'].values

    print('  Treinando classificador...')
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print('  Salvando modelo e classificador...')
    modelo.save(salvar_em)
    joblib.dump(clf, os.path.join(salvar_em, "classificador_logreg.pkl"))
    print(f"  Modelo salvo em: {salvar_em}")

    return modelo


def load_fine_tuning_sarcasmo(caminho=OUTPUT_DIR):
    if not os.path.exists(caminho):
        print(f"[ERRO] O diretório '{caminho}' não foi encontrado.")
        return None
    print(f"  Carregando modelo de: {caminho}")
    return SentenceTransformer(caminho)


if __name__ == "__main__":
    caminho_df = sys.argv[1]
    caminho_modelo = sys.argv[2]

    print("  Carregando modelo base e dados...")
    modelo = SentenceTransformer(caminho_modelo)
    df = pd.read_parquet(caminho_df, engine="pyarrow")

    modelo = sentence_transformer(df, modelo)  # treino e salvamento