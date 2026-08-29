import os
import argparse
import mlflow
import torch  # importa a biblioteca PyTorch
from torch import nn  # importa o módulo de redes neurais
import tempfile

try:
    from dotenv import load_dotenv  # carrega variáveis do arquivo .env para o ambiente
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()  # lê o arquivo .env na raiz do projeto e popula os.environ

# mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI",None)
# if mlflow_tracking_uri is not None:
#     mlflow.set_tracking_uri(mlflow_tracking_uri)

parser = argparse.ArgumentParser(description="Treina um classificador simples de sentimento.")
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=0.1,
    help="Taxa de aprendizado usada pelo otimizador SGD.",
)

args = parser.parse_args()
learning_rate = args.learning_rate

mlflow.set_experiment("demo11-nlp-sentiment")

script_path = os.path.abspath(__file__)
requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

# Exemplo simples de NLP: classificar frases como positivas ou negativas
# Usaremos uma representação bem básica: cada frase vira um vetor de frequência de palavras

texts = [  # armazena as frases de treinamento
    "eu amo este produto",  # frase positiva
    "isso é incrível",  # frase positiva
    "odiei esse filme",  # frase negativa
    "isso foi terrível",  # frase negativa
]

# Labels: 1 = positivo, 0 = negativo
labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32)  # rótulos das frases

# Dicionário simples de palavras
vocab = {  # cria vocabulário com todas as palavras únicas
    "eu": 0,
    "amo": 1,
    "este": 2,
    "produto": 3,
    "isso": 4,
    "é": 5,
    "incrível": 6,
    "odiei": 7,
    "esse": 8,
    "filme": 9,
    "foi": 10,
    "terrível": 11,
}

# Converte cada frase em vetor bag-of-words
# Iremos contar a frequência de cada palavra
X = torch.zeros(len(texts), len(vocab), dtype=torch.float32)  # matriz de entradas: frases x palavras

for i, text in enumerate(texts):  # percorre cada frase
    words = text.split()  # separa a frase em palavras
    for word in words:  # percorre cada palavra da frase
        if word in vocab:  # verifica se a palavra está no vocabulário
            X[i, vocab[word]] += 1.0  # aumenta a contagem dessa palavra

# Modelo: uma simples camada linear
# A entrada tem tamanho do vocabulário e a saída terá apenas 1 valor (probabilidade)
model = nn.Linear(len(vocab), 1)  # camada linear simples para classificação
criterion = nn.BCELoss()  # Binary Cross Entropy Loss para classificação binária
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # otimizador SGD

with mlflow.start_run(run_name="demo07-sentiment-nlp") as run:
    mlflow.autolog()
    mlflow.log_params(
        {
            "vocab_size": len(vocab),
            "learning_rate": learning_rate,
            "optimizer": "SGD",
            "epochs": 1000,
        }
    )
    mlflow.set_tag("model_type", "linear_classifier")
    mlflow.set_tag("task", "sentiment_analysis")

    log_text = []
    log_text.append(f"Run ID: {run.info.run_id} - Treinamento")

    # Treinamento
    for epoch in range(1000):  # repete 1000 épocas
        optimizer.zero_grad()  # limpa gradientes acumulados
        logits = model(X)  # calcula as saídas do modelo para todas as frases
        probs = torch.sigmoid(logits)  # converte os valores em probabilidades entre 0 e 1
        loss = criterion(probs, labels.view(-1, 1))  # calcula erro entre previsão e rótulo real
        loss.backward()  # calcula os gradientes
        optimizer.step()  # atualiza os pesos do modelo

        if epoch % 100 == 0:  # imprime a perda a cada 100 épocas
            print(f"Epoch {epoch:03d} | loss: {loss.item():.4f}")  # mostra o valor da perda
            mlflow.log_metric("loss", float(loss.item()), step=epoch)

    log_text.append(f"Run ID: {run.info.run_id} - Teste")

    # Teste com uma nova frase
    nova_frase = "eu amei este produto"  # frase nova para classificação
    nova_X = torch.zeros(1, len(vocab), dtype=torch.float32)  # cria vetor para a nova frase
    for word in nova_frase.split():  # percorre as palavras da nova frase
        if word in vocab:  # verifica se a palavra existe no vocabulário
            nova_X[0, vocab[word]] += 1.0  # conta a palavra

    with torch.no_grad():  # desativa cálculo de gradiente durante a avaliação
        prob = torch.sigmoid(model(nova_X)).item()  # calcula probabilidade da frase ser positiva
        print("Frase:", nova_frase)  # mostra a frase testada
        print("Probabilidade de ser positiva:", prob)  # exibe a probabilidade
        mlflow.log_metric("positive_probability", float(prob))
        mlflow.log_param("test_sentence", nova_frase)

    log_text.append(f"Run ID: {run.info.run_id} - Exportando o modelo")
    if os.path.exists(script_path):
        mlflow.log_artifact(script_path, artifact_path="source")
        print(f"Arquivo fonte registrado: {script_path}")
    else:
        print(f"Arquivo fonte não encontrado em: {script_path}")

    if os.path.exists(requirements_path):
        mlflow.log_artifact(requirements_path, artifact_path="dependencies")
        print(f"Arquivo requirements registrado: {requirements_path}")
    else:
        print(f"Arquivo requirements não encontrado em: {requirements_path}")

    mlflow.pytorch.log_model(model, artifact_path="model")
    print(f"Run do MLflow: {run.info.run_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    log_text.append(f"Run ID: {run.info.run_id} - {mlflow.get_tracking_uri()}")
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write("\n".join(log_text))
        tmp_file_path = tmp_file.name

    # Envia o arquivo como artifact para o MLflow
    mlflow.log_artifact(tmp_file_path, artifact_path="logs")

    # Remove o arquivo local temporário (opcional)
    os.remove(tmp_file_path)