import os  # acesso às variáveis de ambiente carregadas do .env

from azure.ai.ml import MLClient  # cliente principal do SDK v2 do Azure Machine Learning
from azure.ai.ml.constants import AssetTypes  # tipos de asset de dados suportados (ex.: mltable)
from azure.ai.ml.entities import Data  # entidade usada para registrar assets de dados
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

# Caminho local para a pasta que contém o diabetes.csv e o arquivo de definição MLTable
mltable_path = os.path.join(os.path.dirname(__file__), "..", "data", "train-data")

# Nome e versão que o asset vai receber no workspace
data_name = "diabetes-mltable"
data_version = "1"

# Define o asset de dados do tipo MLTable apontando para a pasta com o diabetes.csv
diabetes_data_asset = Data(
    name=data_name,
    version=data_version,
    description="Dataset diabetes.csv carregado como MLTable",
    path=mltable_path,
    type=AssetTypes.MLTABLE,
)

# Envia o registro do asset de dados no workspace
result = ml_client.data.create_or_update(diabetes_data_asset)

print(
    f"Asset de dados registrado: {result.name} | versão: {result.version} | tipo: {result.type}"
)
