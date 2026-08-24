import torch  # importa a biblioteca PyTorch
from torch import nn  # importa o módulo de redes neurais do PyTorch

# Dados muito simples: 2 entradas e 1 saída binária
X = torch.tensor([  # cria a matriz de entradas do modelo
    [0.0, 0.0],  # exemplo da classe 0
    [1.0, 1.0],  # exemplo da classe 1
    [1.0, 0.0],  # exemplo da classe 1
    [0.0, 1.0],  # exemplo da classe 0
    [2.0, 2.0],  # exemplo da classe 1
    [2.0, 1.0],  # exemplo da classe 1
], dtype=torch.float32)  # define os valores como float32

y = torch.tensor([  # cria os rótulos das classes
    [0.0],  # classe 0
    [1.0],  # classe 1
    [1.0],  # classe 1
    [0.0],  # classe 0
    [1.0],  # classe 1
    [1.0],  # classe 1
], dtype=torch.float32)  # define os rótulos como float32

# Modelo bem simples: uma camada linear
model = nn.Linear(2, 1)  # cria uma camada linear com 2 entradas e 1 saída
criterion = nn.BCELoss()  # BCELoss calcula o erro entre as probabilidades previstas e os rótulos reais (0 ou 1) em problemas de classificação binária; quanto maior a diferença, maior a perda
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # define o otimizador SGD com taxa de aprendizado 0.1

# Treinamento
for epoch in range(1000):  # repete o treinamento 1000 vezes
    optimizer.zero_grad()  # zera os gradientes acumulados
    pred = torch.sigmoid(model(X))  # calcula a probabilidade da previsão para cada entrada
    loss = criterion(pred, y)  # calcula o erro entre previsão e rótulo real
    loss.backward()  # calcula os gradientes do erro
    optimizer.step()  # atualiza os pesos do modelo

    if epoch % 100 == 0:  # a cada 100 épocas, imprime o valor da perda
        print(f"Epoch {epoch:03d} | loss: {loss.item():.4f}")  # mostra a perda atual

# Teste
with torch.no_grad():  # desativa cálculo de gradiente para avaliação
    print("Previsões:")  # imprime o título
    print(torch.sigmoid(model(X)))  # mostra as probabilidades finais para os dados de treino
