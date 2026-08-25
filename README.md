# AI-300 DEMOS

## .env file na raiz do projeto
```bash
AZURE_SUBSCRIPTION_ID=<SUBSCRIPTION_ID>
AZURE_RESOURCE_GROUP=<RESOURCE_GROUP>
AZURE_WORKSPACE_NAME=<WORKSPACE_NAME>
AZURE_STORAGE_ACCOUNT_NAME=<STORAGE_ACCOUNT_NAME>
AZURE_STORAGE_CONTAINER_NAME=<STORAGE_CONTAINER_NAME>
```

O `demo03.py` usa a identidade gerenciada do workspace para acessar a Storage
Account. Conceda a ela a função `Storage Blob Data Contributor` no escopo da
Storage Account ou do container antes de executar o programa.

```bash
az group create --name 'rg-ai300-test-brazilsouth-001' --location brazilsouth

az ml workspace create --name 'aml-ai300-course' -g 'rg-ai300-test-brazilsouth-001'

az ml compute create --name 'waltecoan-compute' --size STANDARD_DS11_V2 --type ComputeInstance -w aml-ai300-course -g rg-ai300-test-brazilsouth-001

az ml compute create --name 'cluster-compute' --size STANDARD_DS11_V2 --type AmlCompute --max-instance 2 -w aml-ai300-course -g rg-ai300-test-brazilsouth-001
```

## Configuração dependencias Python

```jupyternotebook
%pip install --upgrade --force-reinstall -r ../../dev-requirements.txt
```
