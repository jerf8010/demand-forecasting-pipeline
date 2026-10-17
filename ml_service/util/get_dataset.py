from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import traceback
from azureml.core import Workspace
from ml_service.util.env_variables import Env
from azureml.exceptions import ComputeTargetException


def get_dataset(aml_workspace: Workspace, dataset_name: str, datastore_name: str):
    
    e = Env()
    try:
        if dataset_name in Dataset.get_all(aml_workspace):
            dataset = Dataset.get_by_name(
                aml_workspace, 
                dataset_name, 
                e.dataset_version
                )
        
        else:
            datastore = Datastore.get(aml_workspace, datastore_name)
            dataset = Dataset.Tabular.from_parquet_files(path = (datastore, "sales_data.parquet"))
            dataset = dataset.register(
                workspace=aml_workspace,
                name=dataset_name,
                create_new_version=True
                )
        return dataset
    except ComputeTargetException:
        traceback.print_exc()
        print("An error occurred registering dataset.")
        exit(1)            






