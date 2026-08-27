import os  # acesso às variáveis de ambiente carregadas do .env
from azure.ai.ml import MLClient, command  # cliente principal e construtor de command job do Azure ML
from azure.ai.ml.sweep import Choice, MedianStoppingPolicy  # configurações de busca de hiperparâmetros
from azure.identity import DefaultAzureCredential  # autenticação usando credenciais do Azure
try:
    from dotenv import load_dotenv  # carrega variáveis do arquivo .env para o ambiente
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()  # lê o arquivo .env na raiz do projeto e popula os.environ

# Dados da assinatura e do workspace do Azure ML, lidos do arquivo .env
subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_WORKSPACE_NAME"]

# Conecta ao workspace do Azure Machine Learning
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Nome da Compute Instance/Cluster onde o job vai rodar
compute_name = "cpu-cluster"

# Define o Command Job de teste para executar o script demo09.py
job = command(
    code="./",  # diretório contendo o código-fonte
    command="python demo09.py --learning-rate=${{search_space.learning_rate}}",  # executa o treino com o valor avaliado no sweep
    environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu:36",  # ambiente curado do Azure ML
    environment_variables={
        # "CHAVE": "VALOR"
    },
    compute=compute_name,  # destino de computação
    experiment_name="demo09-nlp-sentiment",  # deve corresponder ao mlflow.set_experiment() em demo09.py
    display_name="demo09-command-job",  # nome de exibição do job
    description="Job de teste do Azure ML para executar o demo09.py",
)

# Cria um sweep job que testa cada learning rate usando grid sampling.
sweep_job = job.sweep(
    compute=compute_name,
    sampling_algorithm="grid",
    primary_metric="validation_accuracy",
    goal="Maximize",
    search_space={
        "learning_rate": Choice(values=[0.1, 0.05, 0.01]),
    },
)
sweep_job.early_termination = MedianStoppingPolicy(
    delay_evaluation=5,
    evaluation_interval=1,
)
sweep_job.display_name = "demo09-learning-rate-sweep"
sweep_job.description = "Sweep com grid sampling para avaliar o learning-rate do demo09.py"

# Envia o sweep job para execução no workspace do Azure ML
returned_job = ml_client.jobs.create_or_update(sweep_job)

print(f"Job enviado com sucesso: {returned_job.name} | status: {returned_job.status}")
print(f"Acompanhe a execução no Azure ML Studio: {returned_job.studio_url}")
