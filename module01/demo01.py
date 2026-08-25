import os  # acesso às variáveis de ambiente carregadas do .env

from azure.ai.ml import MLClient  # cliente principal do SDK v2 do Azure Machine Learning
from azure.ai.ml.entities import AssignedUserConfiguration, ComputeInstance  # entidades de Compute Instance e usuário atribuído
from azure.identity import DefaultAzureCredential  # autenticação usando credenciais do Azure
from dotenv import load_dotenv  # carrega variáveis do arquivo .env para o ambiente

load_dotenv()  # lê o arquivo .env na raiz do projeto e popula os.environ

# Dados da assinatura e do workspace do Azure ML, lidos do arquivo .env
subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_WORKSPACE_NAME"]
#tenant_id = os.environ["AZURE_TENANT_ID"]
#user_object_id = os.environ["AZURE_USER_OBJECT_ID"]

# Conecta ao workspace do Azure Machine Learning
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Nome que a Compute Instance vai receber no workspace
compute_name = "ci-standard-ds11-v2"

# Define a Compute Instance com o tamanho de VM Standard_DS11_v2
compute_instance = ComputeInstance(
    name=compute_name,
    size="STANDARD_DS11_V2",
    # create_on_behalf_of=AssignedUserConfiguration(
    #     user_tenant_id=tenant_id,
    #     user_object_id=user_object_id,
    # ),
)

# Envia a criação da Compute Instance e aguarda a conclusão
compute_operation = ml_client.compute.begin_create_or_update(compute_instance)
result = compute_operation.result()

print(
    f"Compute Instance criada: {result.name} | tamanho: {result.size} "
)
