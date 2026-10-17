import os
import argparse
import joblib
import json
from   azureml.core.run import Run
from   azureml.core     import Dataset
from   preprocess_utils import data_process
from   datetime         import datetime 

import warnings
warnings.filterwarnings("ignore")

def pipeline_text():
    return f'>>> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} Azure ML Preprocess Step:'


def main():

    run = Run.get_context()
    print(f"{pipeline_text()} Getting context... ", sep=' ', end='', flush=True)
    # get argpars
    parser = argparse.ArgumentParser("preprocess")
    parser.add_argument("--dataset_name",    type = str, help = ("Dataset name"))
    parser.add_argument("--dataset_version", type = str, help = ("dataset version"))
    parser.add_argument("--step_output",     type = str, help = ("Output next step"))
    args = parser.parse_args()

    step_output_path = args.step_output
    dataset_version  = args.dataset_version
    dataset_name     = args.dataset_name
    print('done \n')

    print(f"{pipeline_text()} Getting argpars parameters... ", sep=' ', end='', flush=True)
    # Load the preprocess parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        preprocess_args = pars["preprocess"]
        week            = preprocess_args["parameters"]["week"]
    except KeyError:
        print("\n \t - Error: Could not load preprocess values from file \n")
        preprocess_args = {}
    # Log the preprocess parameters
    print(f"\n \t - Parameters: {preprocess_args} \n done \n")
    for (k, v) in preprocess_args.items():
        run.log(k, v)
        run.parent.log(k, v)

    print(f"{pipeline_text()} Getting  input datasets and parameters... ", sep = ' ', end='', flush=True)
    dataset = Dataset.get_by_name(
        run.experiment.workspace, dataset_name, dataset_version)
    iti_dataset = Dataset.get_by_name(
        run.experiment.workspace, name=f'Itinerary{week}')
    cal_dataset = Dataset.get_by_name(
        run.experiment.workspace, name = f'Calendar{week}'
    )
    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    run.parent.tag("dataset_id", value=dataset.id)
    df        = dataset.to_pandas_dataframe()

    itinerary = iti_dataset.to_pandas_dataframe()
    calendar = cal_dataset.to_pandas_dataframe()
    print('done \n')

    print(f'{pipeline_text()} Staring preprocess... ')
    run.tag("run_type", value="preprocess")
    print(f"\t - tags now present for run: {run.tags} \n")
    pivot_df, pivot_df_returns, \
        train_pivot, test_pivot = data_process(df=df, calendar = calendar, 
                                                                    itinerary = itinerary, params = preprocess_args["parameters"])

    ################################ QUITAR
    #pivot_df = pivot_df.iloc[:5, :]
    #train_pivot = train_pivot.iloc[:5, :]
    #test_pivot = test_pivot.iloc[:5, :]
    #print(train_pivot)
    ################################ <------------------

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok=True)
    os.makedirs('outputs',        exist_ok=True)

    joblib.dump(
        value=pivot_df, 
        filename=os.path.join(step_output_path, f'pivot_df{week}.pkl')
        )
    joblib.dump(
        value=pivot_df, 
        filename=os.path.join('outputs', f'pivot_df{week}.pkl')
        )
    joblib.dump(
        value=pivot_df_returns, 
        filename=os.path.join(step_output_path, f'pivot_df_returns{week}.pkl')
        )
    joblib.dump(
        value=pivot_df_returns, 
        filename=os.path.join('outputs', f'pivot_df_returns{week}.pkl')
        )
    joblib.dump(
        value=train_pivot, 
        filename=os.path.join(step_output_path, f'train_pivot{week}.pkl')
        )
    joblib.dump(
        value=train_pivot, 
        filename=os.path.join('outputs', f'train_pivot{week}.pkl')
        )

    joblib.dump(
        value=test_pivot, 
        filename=os.path.join(step_output_path, f'test_pivot{week}.pkl')
        )
    joblib.dump(
        value=test_pivot, 
        filename=os.path.join('outputs', f'test_pivot{week}.pkl')
        )

if __name__ == '__main__':
    main()
