from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import traceback
from azureml.core import Workspace
from ml_service.util.env_variables import Env
from azureml.exceptions import ComputeTargetException
from azureml.exceptions import UserErrorException


def get_datastore(aml_workspace: Workspace, datastore_name: str, blob_name: str):
    
    e = Env()

    try:
        if datastore_name in aml_workspace.datastores:
            datastore = Datastore.get(aml_workspace, datastore_name)
        else:
            datastore = Datastore.register_azure_blob_container(
                workspace=aml_workspace,
                datastore_name=datastore_name,
                account_name=e.storage_account, 
                container_name=blob_name,
                account_key=e.storage_account_access
                )
    except ComputeTargetException:
        traceback.print_exc()
        print("An error occurred registering datastore.")
        exit(1)            

       
    return datastore

