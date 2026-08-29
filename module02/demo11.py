import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
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

@pipeline(
    name="pipeline_treinar_modelo",
    display_name="Pipeline de treinamento de sentimento",
    description="Pipeline que executa o treinamento do classificador de sentimento",
)
def pipeline_treinar_modelo():
    compute_name = "cpu-cluster"
    environment_name = "AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu:36"

    treinar_modelo_component = command(
        name="treinar_modelo",
        display_name="Treinar modelo de sentimento",
        experiment_name="demo11-nlp-sentiment",
        description="Executa o component01.py e treina o classificador de sentimento",
        code="./",
        command=(
            "python component01.py --learning-rate=${{inputs.learning_rate}}"
        ),
        inputs={
            "learning_rate": 0.1,
        },
        environment=environment_name,
        compute=compute_name
    )

    treinar_modelo_component()


job = pipeline_treinar_modelo()
job.display_name = "demo11-pipeline-treinar-modelo"
job.experiment_name = "demo11-nlp-sentiment"

returned_job = ml_client.jobs.create_or_update(job)

print(f"Job enviado com sucesso: {returned_job.name} | status: {returned_job.status}")
print(f"Acompanhe a execução no Azure ML Studio: {returned_job.studio_url}")
