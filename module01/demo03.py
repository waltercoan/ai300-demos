import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


load_dotenv()

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_WORKSPACE_NAME"]
storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
storage_container_name = os.environ["AZURE_STORAGE_CONTAINER_NAME"]

ml_client = MLClient(
	credential=DefaultAzureCredential(),
	subscription_id=subscription_id,
	resource_group_name=resource_group,
	workspace_name=workspace_name,
)

datastore = AzureBlobDatastore(
	name="storage-datastore",
	account_name=storage_account_name,
	container_name=storage_container_name,
)

result = ml_client.datastores.create_or_update(datastore)

print(
	f"Conexao criada: {result.name} | conta de armazenamento: "
	f"{result.account_name} | container: {result.container_name}"
)
