import torch  # importa a biblioteca PyTorch
from torch import nn  # importa o módulo de redes neurais do PyTorch

# Dados simples para regressão linear
# Queremos prever o valor y a partir de x
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)  # entrada: valores de x
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)  # saída esperada: y = 2x

# Define um modelo linear: y = w*x + b
model = nn.Linear(1, 1)  # camada com 1 entrada e 1 saída

# Define a perda e o otimizador
criterion = nn.MSELoss()  # Mean Squared Error: calcula o erro quadrático médio
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD ajusta os pesos para reduzir o erro

# Treinamento
for epoch in range(2000):  # repete 2000 vezes
    optimizer.zero_grad()  # limpa os gradientes acumulados
    pred = model(x)  # faz a previsão do modelo para os valores de x
    loss = criterion(pred, y)  # calcula o erro entre previsão e valor real
    loss.backward()  # calcula gradientes do erro em relação aos pesos
    optimizer.step()  # atualiza os pesos do modelo

    if epoch % 100 == 0:  # imprime a perda a cada 100 épocas
        print(f"Epoch {epoch:03d} | loss: {loss.item():.4f}")  # mostra o valor da perda atual

# Teste final
with torch.no_grad():  # desativa cálculo de gradiente para avaliação
    print("Previsão para x = 5:", model(torch.tensor([[5.0]])))  # imprime a previsão para x = 5
