# AI-300 DEMOS

## .env file na raiz do projeto
```bash
AZURE_SUBSCRIPTION_ID=<SUBSCRIPTION_ID>
AZURE_RESOURCE_GROUP=<RESOURCE_GROUP>
AZURE_WORKSPACE_NAME=<WORKSPACE_NAME>
AZURE_STORAGE_ACCOUNT_NAME=<STORAGE_ACCOUNT_NAME>
AZURE_STORAGE_CONTAINER_NAME=<STORAGE_CONTAINER_NAME>
AZURE_USER_ASSIGNED_IDENTITY_ID=/subscriptions/<SUBSCRIPTION_ID>/resourcegroups/<RESOURCE_GROUP>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<NAME_IDENTITY>
```

## Configuração dependencias Python

```jupyternotebook
%pip install --upgrade --force-reinstall -r ../../dev-requirements.txt
```

```bash
az group create --name 'rg-ai300-test-brazilsouth-001' --location brazilsouth

az ml workspace create --name 'aml-ai300-course' -g 'rg-ai300-test-brazilsouth-001'
```

## Modulo 01

### Configuração COMPUTE

[Compute Instances](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2)

- [Python :: demo01](./module01/demo01.py) 
- [Python :: demo02](./module01/demo02.py)

```bash
az ml compute create --name 'waltecoan-compute' --size STANDARD_DS11_V2 --type ComputeInstance -w aml-ai300-course -g rg-ai300-test-brazilsouth-001

az ml compute create --name 'cluster-compute' --size STANDARD_DS11_V2 --type AmlCompute --max-instance 2 -w aml-ai300-course -g rg-ai300-test-brazilsouth-001
```

### Criação de um Data Asset

[Data Asset](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets)

- [Python :: demo04](./module01/demo04.py) 
- [Python :: demo05](./module01/demo05.py)

```bash
az ml data create \
	--name diabetes-data \
	--version 1 \
	--path ./data/train-data/MLTable \
	--type mltable \
	--workspace-name "$AZURE_WORKSPACE_NAME" \
	--resource-group "$AZURE_RESOURCE_GROUP"
```

### AutoML

[AutoML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)

![AutoML Process](https://learn.microsoft.com/en-us/azure/machine-learning/media/concept-automated-ml/automl-concept-diagram2.png?view=azureml-api-2)

- [Python :: demo06](./module01/demo06.py)

```bash
# O arquivo YAML deve conter `type: automl` e `experiment_name: automl-diabetes`.
az ml job create \
	--file ./module01/automl-job.yml \
	--workspace-name "$AZURE_WORKSPACE_NAME" \
	--resource-group "$AZURE_RESOURCE_GROUP"
```

### Create Job

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: .
command: python ./module01/demo07.py
environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1
compute: azureml:cluster-compute
experiment_name: demo07
description: Run demo07.py
```

```bash
az ml job create \
  --file ./module01/demo07-job.yml \
  --workspace-name "$AZURE_WORKSPACE_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP"
```

### Parameter command job and Sweep jobs

- [Python :: demo09](./module01/demo09.py)
- [Python :: demo10](./module01/demo10.py)

- [YAML :: demo10-sweep-job](./module01/demo10-sweep-job.yml)
```bash
az ml job create \
	--file ./module01/demo10-sweep-job.yml \
	--workspace-name "$AZURE_WORKSPACE_NAME" \
	--resource-group "$AZURE_RESOURCE_GROUP"

```

## Modulo 02

### Pipelines

- [Python :: component01](./module01/component01.py)
- [Python :: demo11](./module01/demo11.py)


### Endpoint Online test

```json
{
  "input_data": {
    "columns": [
      "PatientID",
      "Pregnancies",
      "PlasmaGlucose",
      "DiastolicBloodPressure",
      "TricepsThickness",
      "SerumInsulin",
      "BMI",
      "DiabetesPedigree",
      "Age"
    ],
    "index": [0],
    "data": [[4, 8,171,42,29,160,35.48224692,0.082671083,22]]
  },
  "params": {}
}
```
