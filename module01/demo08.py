import os  # acesso às variáveis de ambiente carregadas do .env

from azure.ai.ml import MLClient, command  # cliente principal e construtor de command job do Azure ML
from azure.identity import DefaultAzureCredential  # autenticação usando credenciais do Azure
from dotenv import load_dotenv  # carrega variáveis do arquivo .env para o ambiente

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

# Define o Command Job para executar o script demo07.py
job = command(
    code="./",  # diretório contendo o código-fonte
    command="python demo07.py",  # instala dependências e executa o treino
    environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu:36",  # ambiente curado do Azure ML
    environment_variables={
        "AZURE_SUBSCRIPTION_ID": subscription_id,
        "AZURE_RESOURCE_GROUP": resource_group,
        "AZURE_WORKSPACE_NAME": workspace_name,
    },
    compute=compute_name,  # destino de computação
    experiment_name="demo07-experiment",  # nome do experimento no Azure ML Studio
    display_name="demo07-command-job",  # nome de exibição do job
    description="Job do Azure ML para executar o demo07.py",
)

# Envia o job para execução no workspace do Azure ML
returned_job = ml_client.jobs.create_or_update(job)

print(f"Job enviado com sucesso: {returned_job.name} | status: {returned_job.status}")
print(f"Acompanhe a execução no Azure ML Studio: {returned_job.studio_url}")
