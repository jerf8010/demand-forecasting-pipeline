import os, sys
import argparse
import joblib
import json
from   azureml.core.run import Run
from   azureml.core     import Dataset
from   datetime         import datetime 

from splitting_utils import split
from auxiliar_func.models.auxiliar_models.suggested import summary_suggested_order

import warnings
warnings.filterwarnings("ignore")

def pipeline_text():
    return f'>>> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} Azure ML Splitting Step:'

def main():

    print(f"{pipeline_text()} Getting context... ", sep=' ', end='', flush=True)
    run = Run.get_context()
    print('done \n')

    print(f"{pipeline_text()} Getting argpars parameters... ", sep=' ', end='', flush=True)
    parser = argparse.ArgumentParser("preprocess")
    parser.add_argument("--dataset_name",    type = str, help = ("Dataset name"))
    parser.add_argument("--dataset_version", type = str, help = ("dataset version"))
    parser.add_argument("--step_output",     type = str, help = ("Output next step"))
    parser.add_argument("--prep_output",  type=str,  help=("input for passing data to next step"))
    parser.add_argument("--train_output", type=str,  help=("input for passing data to next step"))
    #parser.add_argument("--step_input",      type = str, help = ("Inputs from scoring step"))

    args = parser.parse_args()
    dataset_version  = args.dataset_version
    dataset_name     = args.dataset_name

    step_output_path  = args.step_output
    step_input_path  = [args.prep_output, args.train_output]
    print('done \n')

    print(f"{pipeline_text()} Getting  input datasets and parameters... ", sep = ' ', end='', flush=True)
    # Load the preprocess parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        preprocess_args = pars["preprocess"]
        week            = preprocess_args["parameters"]["week"]

    except KeyError:
        preprocess_args = {}
    # Log the preprocess parameters
    for (k, v) in preprocess_args.items():
        run.log(k, v)
        run.parent.log(k, v)
    dataset = Dataset.get_by_name(
        run.experiment.workspace, dataset_name, dataset_version)
    iti_dataset = Dataset.get_by_name(
        run.experiment.workspace, name=f'Itinerary{week}')
    cal_dataset = Dataset.get_by_name(
        run.experiment.workspace, name = f'Calendar{week}'
    )

    df        = dataset.to_pandas_dataframe()
    itinerary = iti_dataset.to_pandas_dataframe()
    calendar = cal_dataset.to_pandas_dataframe()

    pivot_df_input_path = os.path.join(step_input_path[0], f'pivot_df{week}.pkl')
    pivot_df     = joblib.load(pivot_df_input_path)

    future_preds_input_path = os.path.join(step_input_path[1], f'futurePredictions{week}.pkl')
    future_preds     = joblib.load(future_preds_input_path)

    pesos_input_path = os.path.join(step_input_path[1], f'pesos{week}.pkl')
    pesos     = joblib.load(pesos_input_path)

    model_client_input_path = os.path.join(step_input_path[1], f'model_client{week}.pkl')
    model_client     = joblib.load(model_client_input_path)

    suggested_input_path = os.path.join(step_input_path[1], f'suggested{week}.pkl')
    suggested     = joblib.load(suggested_input_path)

    pct_sales_input_path = os.path.join(step_input_path[1], f'pct_sales{week}.pkl')
    pct_sales     = joblib.load(pct_sales_input_path)

    
    summary, _ = summary_suggested_order(pivot_df, future_preds, pesos, model_client, suggested, pct_sales)

    import numpy as np
    predsSuggested = summary[[summary.columns[2]]]
    predsSuggested = predsSuggested.rename(columns = {summary.columns[2] : 'predictions'})
    predsSuggested = predsSuggested.reset_index()
    predsSuggested['predictions'] = abs(np.ceil(predsSuggested['predictions']))

    splittingSuggested = split(predsSuggested, df, itinerary, calendar, preprocess_args["parameters"])

    suggested = splittingSuggested[['Date', 'Branch', 'Route', 'Customer', 'Product', 'SuggestedRecal']]
    suggested = suggested.rename(columns = {'SuggestedRecal' : 'SuggestedOrder'})

    suggested.to_csv(f'./outputs/preds_to_deliver{week}.csv',  index = False)
    joblib.dump(
        value=suggested, 
        filename=os.path.join('outputs', f'preds_to_deliver{week}.pkl')
        )
        
    run.complete()
if __name__ == '__main__':
    main()
