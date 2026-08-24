import torch  # importa a biblioteca PyTorch
from torch import nn  # importa o módulo de redes neurais

# Exemplo simples de NLP: classificar frases como positivas ou negativas
# Usaremos uma representação bem básica: cada frase vira um vetor de frequência de palavras

# Lista de frases de exemplo
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # otimizador SGD

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
