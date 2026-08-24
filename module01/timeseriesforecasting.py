import torch  # importa a biblioteca PyTorch

# Série temporal simples: valores que crescem linearmente ao longo do tempo
# Exemplo: [1, 2, 3, 4, 5, 6, 7, 8]
series = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)  # dados históricos

# Criamos entradas e saídas para prever o próximo valor
# Para cada ponto, usamos os 3 valores anteriores para prever o próximo
X = torch.tensor([
    [1.0, 2.0, 3.0],  # entrada 1 -> prevendo 4
    [2.0, 3.0, 4.0],  # entrada 2 -> prevendo 5
    [3.0, 4.0, 5.0],  # entrada 3 -> prevendo 6
    [4.0, 5.0, 6.0],  # entrada 4 -> prevendo 7
    [5.0, 6.0, 7.0],  # entrada 5 -> prevendo 8
], dtype=torch.float32)  # conjunto de entradas

y = torch.tensor([
    [4.0],  # alvo 1
    [5.0],  # alvo 2
    [6.0],  # alvo 3
    [7.0],  # alvo 4
    [8.0],  # alvo 5
], dtype=torch.float32)  # conjunto de alvos

# Modelo linear simples: recebe 3 valores e retorna 1 valor previsto
model = torch.nn.Linear(3, 1)  # camada com 3 entradas e 1 saída
criterion = torch.nn.MSELoss()  # perda de erro quadrático médio
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # otimizador para ajustar os pesos

# Treinamento
for epoch in range(2000):  # repete 2000 épocas
    optimizer.zero_grad()  # limpa os gradientes antigos
    pred = model(X)  # calcula as previsões para as entradas
    loss = criterion(pred, y)  # compara previsão com os valores esperados
    loss.backward()  # calcula gradientes da perda
    optimizer.step()  # atualiza os pesos do modelo

    if epoch % 100 == 0:  # imprime a perda a cada 100 épocas
        print(f"Epoch {epoch:03d} | loss: {loss.item():.4f}")  # mostra o valor da perda atual

# Previsão do próximo valor usando os últimos 3 pontos conhecidos
last_window = torch.tensor([[6.0, 7.0, 8.0]], dtype=torch.float32)  # últimos 3 valores da série
next_value = model(last_window)  # prevê o próximo valor

print("Últimos 3 valores da série:", last_window.numpy())  # mostra os dados usados na previsão
print("Próximo valor previsto:", next_value.item())  # mostra a previsão final
