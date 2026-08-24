import torch  # importa a biblioteca PyTorch
from torch import nn  # importa módulos de redes neurais

# Exemplo simples de Computer Vision: classificar imagens 2x2 em duas classes
# Classe 0: imagem com pixels mais claros em uma região
# Classe 1: imagem com pixels mais claros em outra região

# Cada imagem é uma matriz 2x2 com valores entre 0 e 1
# Aqui criamos 4 exemplos de treinamento
X = torch.tensor([
    [[0.0, 0.0], [0.0, 1.0]],  # imagem da classe 1
    [[1.0, 0.0], [0.0, 0.0]],  # imagem da classe 0
    [[0.0, 1.0], [0.0, 0.0]],  # imagem da classe 0
    [[0.0, 0.0], [1.0, 0.0]],  # imagem da classe 1
], dtype=torch.float32)  # define o tipo dos dados como float32

y = torch.tensor([
    [1.0],  # classe 1
    [0.0],  # classe 0
    [0.0],  # classe 0
    [1.0],  # classe 1
], dtype=torch.float32)  # rótulos das imagens

# A imagem 2x2 precisa ser achatada para entrar em uma camada linear
# Isso transforma cada imagem em um vetor de 4 valores
X_flat = X.view(-1, 4)  # reshape: 4 imagens x 4 pixels

# Modelo simples: camada linear recebendo 4 pixels e retornando 1 valor
model = nn.Linear(4, 1)  # 4 entradas, 1 saída
criterion = nn.BCELoss()  # perda para classificação binária
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # otimizador SGD

# Treinamento
for epoch in range(2000):  # repete 2000 épocas
    optimizer.zero_grad()  # limpa gradientes anteriores
    logits = model(X_flat)  # calcula saída do modelo para todas as imagens
    probs = torch.sigmoid(logits)  # transforma em probabilidades entre 0 e 1
    loss = criterion(probs, y)  # compara com os rótulos reais
    loss.backward()  # calcula gradientes
    optimizer.step()  # atualiza pesos

    if epoch % 100 == 0:  # imprime perda a cada 100 épocas
        print(f"Epoch {epoch:03d} | loss: {loss.item():.4f}")  # mostra a perda atual

# Teste com uma nova imagem
nova_imagem = torch.tensor([
    [[1.0, 0.0], [0.0, 0.0]],  # imagem parecida com a classe 0
], dtype=torch.float32)  # cria nova imagem

nova_imagem_flat = nova_imagem.view(-1, 4)  # achata a nova imagem

with torch.no_grad():  # desativa cálculo de gradiente durante a avaliação
    prob = torch.sigmoid(model(nova_imagem_flat)).item()  # calcula probabilidade da imagem ser da classe 1
    print("Probabilidade da nova imagem ser classe 1:", prob)  # mostra a previsão
