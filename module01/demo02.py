import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


load_dotenv()

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name = os.environ["AZURE_WORKSPACE_NAME"]

ml_client = MLClient(
	credential=DefaultAzureCredential(),
	subscription_id=subscription_id,
	resource_group_name=resource_group,
	workspace_name=workspace_name,
)

cluster_name = "cpu-cluster"
compute_cluster = AmlCompute(
	name=cluster_name,
	type="amlcompute",
	size="STANDARD_DS11_V2",
	min_instances=0,
	max_instances=1,
)

result = ml_client.compute.begin_create_or_update(compute_cluster).result()

print(
	f"Computer cluster criado: {result.name} | tipo: {result.size} "
	f"| mínimo: {result.min_instances} | máximo: {result.max_instances}"
)
