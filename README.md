# AI-300 DEMOS

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
