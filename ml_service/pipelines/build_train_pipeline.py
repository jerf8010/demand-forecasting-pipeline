from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps      import PythonScriptStep
from azureml.pipeline.core       import Pipeline, PipelineData, PublishedPipeline
from azureml.core                import Dataset
from azureml.core.runconfig      import RunConfiguration

import sys, os
sys.path.append(os.path.abspath(os.path.join('.', 'AzureMachineLearning')))

from ml_service.util             import Env
from ml_service.util             import get_compute
from ml_service.util             import get_environment
from ml_service.util             import get_dataset
from ml_service.util             import get_datastore

def main():
    
    # get environment
    e = Env()
    # get workspace
    from azureml.core import Workspace

    print('>>>> 1) Getting Workspace... ')
    aml_workspace = Workspace.get(
        name            = e.workspace_name,
        subscription_id = e.subscription_id,
        resource_group  = e.resource_group,
        )
    print("\t - Workspace: "+ str(aml_workspace))
    print('done \n')

    print('>>>> 2) Getting Compute Instance... ')
    # get compute
    aml_compute = get_compute(
        workspace    = aml_workspace, 
        compute_name = e.compute_name, 
        vm_size      = e.vm_size
        )
    print("\t - AML_compute: " + str(aml_compute))
    print('done \n')

    print('>>>> 3) Getting Environment... ')
    # environment
    environment = get_environment(
        workspace               = aml_workspace,
        environment_name        = e.aml_env_name,
        conda_dependencies_file = e.aml_env_train_conda_dep_file,
        create_new              = e.rebuild_env
    ) 
    print("\t - Environment: " + str(environment))
    print('done \n')

    print('>>>> 4) Getting Datastore... ')
    # datastore
    datastore_input_ml = get_datastore(
        aml_workspace  = aml_workspace,  
        datastore_name = e.datastore_input_ml_name,
        blob_name      = e.datastore_input_blob_name
    )
    datastore_output_ml = get_datastore(
        aml_workspace  = aml_workspace,  
        datastore_name = e.datastore_output_ml_name,
        blob_name      = e.datastore_output_blob_name
    )
    print('done \n')

    print('>>>> 5) Getting Dataset... ')
    # dataset
    dataset_name = get_dataset(
        aml_workspace  = aml_workspace, 
        dataset_name   = e.dataset_name, 
        datastore_name = datastore_input_ml.name
    )
    print("done \n")

    print('>>>> 6) Setting Pipeline Parameters... ')
    # run config 
    run_config                                                     = RunConfiguration()
    run_config.environment                                         = environment
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_input_ml.name  

    # pipelines parameters
    dataset_version_param = PipelineParameter(
        name          = "dataset_version",
        default_value = e.dataset_version
        )
    dataset_name = PipelineParameter(
        name          = "dataset_name",
        default_value = e.dataset_name
        )    
    pipeline_prep_out = PipelineData(
        name      = "pipeline_prep_out",
        datastore = aml_workspace.get_default_datastore()
        )
    pipeline_train_out = PipelineData(
        name      = "pipeline_train_out",
        datastore = aml_workspace.get_default_datastore()
        )
    pipeline_score_out = PipelineData(
        name      = "pipeline_score_out",
        datastore = aml_workspace.get_default_datastore()
        )
    pipeline_split_out = PipelineData(
        name      = "pipeline_split_out",
        datastore = aml_workspace.get_default_datastore()
        )
    print("done \n")

    print('>>>> 7) Creatting Preprocess Step... ')
    preprocess_step = PythonScriptStep(
        name="Preprocess Data",
        script_name      = e.preprocess_script,
        compute_target   = aml_compute,
        source_directory = e.sources_directory_train,
        outputs          = [pipeline_prep_out],
        arguments=[
            "--dataset_name",    dataset_name,
            "--dataset_version", dataset_version_param,
            "--step_output",     pipeline_prep_out
        ],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("done \n")

    # train Model step
    print('>>>> 8) Creatting Traing Step... ')
    train_step = PythonScriptStep(
        name="Train Model",
        script_name      = e.train_script_path,
        compute_target   = aml_compute,
        source_directory = e.sources_directory_train,
        outputs          = [pipeline_train_out],
        inputs           = [pipeline_prep_out],
        arguments=[
            "--step_output", pipeline_train_out,
            "--step_input", pipeline_prep_out
        ],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("done \n")
    
    print('>>>> 9) Creatting Scoring Step... ')
    scoring_step = PythonScriptStep(
        name="Scoring Model ",
        script_name      = e.score_script,
        compute_target   = aml_compute,
        source_directory = e.sources_directory_train,
        outputs          = [pipeline_score_out],
        inputs           = [pipeline_prep_out, pipeline_train_out],
        arguments=[
            "--step_output", pipeline_score_out,
            "--prep_output", pipeline_prep_out,
            "--train_output", pipeline_train_out,
#            "--storage_account", e.storage_account,
#            "--scoring_datastore_output_container", e.scoring_datastore_output_container,
#            "--scoring_datastore_access_key", e.scoring_datastore_access_key,
#            "--scoring_datastore_output_filename", e.scoring_datastore_output_filename
            ], 
        runconfig=run_config,
        allow_reuse=False,
    )
    print("done \n")  
    
    print('>>>> 10) Creatting Split Step... ')
    splitting_step = PythonScriptStep(
        name             = "Splitting Outputs ",
        script_name      = e.split_script_path,
        compute_target   = aml_compute,
        source_directory = e.sources_directory_train,
        outputs          = [pipeline_split_out],
        inputs           = [pipeline_prep_out, pipeline_train_out],
        arguments=[
            "--dataset_name", dataset_name,
            "--dataset_version", dataset_version_param,
            "--step_output", pipeline_split_out,
            "--prep_output", pipeline_prep_out,
            "--train_output", pipeline_train_out,
            ], 
        runconfig=run_config,
        allow_reuse=False,
    )
    print("done \n")  

    train_step.run_after(preprocess_step)
    scoring_step.run_after(train_step)
    splitting_step.run_after(scoring_step)
    
    steps = [preprocess_step, 
             train_step, 
             scoring_step,
             splitting_step
            ]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()
        
    published_pipeline = train_pipeline.publish(
            name=e.pipeline_name,
            description="Test",
            version=e.build_id
        )

    print('>>>> 11) Publicate Pipeline... ')

    pipelines = PublishedPipeline.list(aml_workspace)
    matched_pipes = []

    for p in pipelines:
        if p.name == e.pipeline_name:
            if p.version == e.build_id:
                matched_pipes.append(p)

    published_pipeline = matched_pipes[0]
    print('\t - ', published_pipeline)

    tags = {"dataset_name": e.dataset_name}
    tags["BuildId"] = e.build_id
    tags["BuildUri"] = e.build_uri
    tags["dataset_name"] = e.dataset_name
    print(tags)

    pipeline_parameters = {"model_name": e.model_name}
    print(pipeline_parameters)

    from azureml.core import Experiment, Workspace

    experiment = Experiment(
        workspace=aml_workspace,
        name=e.experiment_name
        )
    run = experiment.submit(
        published_pipeline,
        tags=tags,
        pipeline_parameters=pipeline_parameters)

if __name__ == "__main__":
    main()
