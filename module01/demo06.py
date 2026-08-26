import os  # acesso às variáveis de ambiente carregadas do .env

from azure.ai.ml import MLClient, Input, automl  # cliente principal, entrada de dados e API do AutoML
from azure.ai.ml.automl import ClassificationModels  # enum com os algoritmos de classificação suportados pelo AutoML
from azure.ai.ml.constants import AssetTypes  # tipos de asset de dados suportados (ex.: mltable)
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

# Recupera os data assets MLTable já registrados no workspace
train_data_asset = ml_client.data.get(name="diabetes-mltable", version="1")
test_data_asset = ml_client.data.get(name="diabetes-test-mltable", version="1")

# Referencia os assets pelo id para uso no job de AutoML
training_data = Input(type=AssetTypes.MLTABLE, path=train_data_asset.id)
test_data = Input(type=AssetTypes.MLTABLE, path=test_data_asset.id)

# Nome da Compute Instance/Cluster onde o job de AutoML vai rodar
compute_name = "cpu-cluster"

# Define o experimento de AutoML de classificação para prever a coluna Diabetic
classification_job = automl.classification(
    experiment_name="diabetes-automl-classification",
    compute=compute_name,
    training_data=training_data,
    test_data=test_data,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    # divide os dados em 5 folds; a cada rodada 4 folds treinam e 1 valida, e a accuracy final é a média das 5 rodadas
    n_cross_validations=5,
)

# Define os limites de execução do experimento (tempo e número de trials)
classification_job.set_limits(
    timeout_minutes=30,
    trial_timeout_minutes=10,
    max_trials=5,
    enable_early_termination=True,
)

# Restringe o AutoML aos algoritmos de classificação informados nesta lista
classification_job.set_training(
    allowed_training_algorithms=[
        ClassificationModels.LOGISTIC_REGRESSION,
        ClassificationModels.DECISION_TREE,
        ClassificationModels.RANDOM_FOREST,
        ClassificationModels.LIGHT_GBM,
        ClassificationModels.XG_BOOST_CLASSIFIER,
        ClassificationModels.KNN,
    ]
)

# Envia o job de AutoML para execução no workspace
returned_job = ml_client.jobs.create_or_update(classification_job)

print(f"Job de AutoML enviado: {returned_job.name} | status: {returned_job.status}")
print(f"Acompanhe no Studio: {returned_job.studio_url}")
