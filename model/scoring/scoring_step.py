import os, sys
import argparse
import joblib
import json
from   azureml.core.run import Run
from   datetime         import datetime
from   scoring_utils import scoring

import warnings
warnings.filterwarnings("ignore")

def pipeline_text():
    return f'>>> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} Azure ML Scoring Step:'


def main():
    
    print(f"{pipeline_text()} Get context... ", sep=' ', end='', flush=True)
    run = Run.get_context()
    # get argpars
    parser = argparse.ArgumentParser("scoring")
    parser.add_argument("--step_output",  type=str, help=("output for passing data to next step"))
    parser.add_argument("--prep_output",  type=str,  help=("input for passing data to next step"))
    parser.add_argument("--train_output", type=str,  help=("input for passing data to next step"))
#    parser.add_argument("--storage_account",type=str,help=("storage account"))
#    parser.add_argument("--scoring_datastore_output_container",type=str,help=("scoring_datastore_output_container"))
#    parser.add_argument("--scoring_datastore_access_key",type=str,help=("scoring_datastore_access_key"))
#    parser.add_argument("--scoring_datastore_output_filename",type=str,help=("scoring_datastore_output_filename"))

    args = parser.parse_args()
    
    step_output_path  = args.step_output
    step_input_path  = [args.prep_output, args.train_output]
#    storage_account   = args.storage_account
#    scoring_datastore_output_container = args.scoring_datastore_output_container
#    scoring_datastore_access_key = args.scoring_datastore_access_key
#    scoring_datastore_output_filename = args.scoring_datastore_output_filename

    print('done \n')

    print(f"{pipeline_text()} Getting argpars parameters... ", sep=' ', end='', flush=True)
    # Load the training parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        week = pars["preprocess"]["parameters"]["week"]
        # Log the training parameters
        print(f"\n \t - Scoring week: {week} \n done \n")
        #for (k, v) in train_args.items():
        #    run.log(k, v)
        #    run.parent.log(k, v)
    except KeyError:
        print("\n \t - Error: Could not load preprocess values from file \n")
        train_args = {}
    

    print(f"{pipeline_text()} Getting  input datasets... ", sep = ' ', end='', flush=True)
    # Link dataset to the step run so it is trackable in the UI

    train_model_input_path = os.path.join(step_input_path[0], f'train_pivot{week}.pkl')
    train_pivot            = joblib.load(train_model_input_path)

    test_model_input_path = os.path.join(step_input_path[0], f'test_pivot{week}.pkl')
    test_pivot            = joblib.load(test_model_input_path)

    validation_preds_input_path = os.path.join(step_input_path[1], f'validationPredictions{week}.pkl')
    validation_preds            = joblib.load(validation_preds_input_path)
    
    # Scoring the model
    print(f"{pipeline_text()} Scoring model... ", sep = ' ', end='', flush=True)
    pred, mae, tab = scoring(train_pivot, test_pivot, validation_preds)
    print("done \n")

    if pred is None:
        sys.exit()

    print(f"{pipeline_text()} Saving outputs... ", sep = ' ', end='', flush=True)

    os.makedirs(step_output_path, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(
        value=pred, 
        filename=os.path.join(step_output_path, f'pred{week}.pkl')
        )
    joblib.dump(
        value=pred, 
        filename=os.path.join('outputs', f'pred{week}.pkl')
        )
    joblib.dump(
        value=mae, 
        filename=os.path.join(step_output_path, f'mae{week}.pkl')
        )
    joblib.dump(
        value=mae, 
        filename=os.path.join('outputs', f'mae{week}.pkl')
        )
    joblib.dump(
        value=tab, 
        filename=os.path.join(step_output_path, f'tab{week}.pkl')
        )
    joblib.dump(
        value=tab, 
        filename=os.path.join('outputs', f'tab{week}.pkl')
        )

    # global Mae log
    run.log('Benchmark - mean absolute error', mae.iloc[0, 0])
    run.log('Benchmark - median absolute error', mae.iloc[1, 0])

    run.log('Model - mean absolute error', mae.iloc[0, 1])
    run.log('Model - median absolute error', mae.iloc[1, 1])

    # classification errors
    run.log('Customers Compras - bajas: Benchmark median', tab.iloc[0, 0])
    run.log('Customers Compras - bajas: Model median',     tab.iloc[1, 0])
    run.log('Customers Compras - bajas: Benchmark mean',   tab.iloc[2, 0])
    run.log('Customers Compras - bajas: Model mean',       tab.iloc[3, 0])
    run.log('Low-volume customers: number of series',    tab.iloc[4, 0])
    run.log('Customers Compras - bajas: Sale Promedio',   tab.iloc[5, 0])
    run.log('Low-volume customers: error (%)',        tab.iloc[6, 0])

    run.log('Customers Compras - medias: Benchmark median', tab.iloc[0, 1])
    run.log('Customers Compras - medias: Model median',     tab.iloc[1, 1])
    run.log('Customers Compras - medias: Benchmark mean',   tab.iloc[2, 1])
    run.log('Customers Compras - medias: Model mean',       tab.iloc[3, 1])
    run.log('Mid-volume customers: number of series',    tab.iloc[4, 1])
    run.log('Customers Compras - medias: Sale Promedio',   tab.iloc[5, 1])
    run.log('Mid-volume customers: error (%)',        tab.iloc[6, 1])

    run.log('Customers Compras - altas: Benchmark median', tab.iloc[0, 2])
    run.log('Customers Compras - altas: Model median',     tab.iloc[1, 2])
    run.log('Customers Compras - altas: Benchmark mean',   tab.iloc[2, 2])
    run.log('Customers Compras - altas: Model mean',       tab.iloc[3, 2])
    run.log('High-volume customers: number of series',    tab.iloc[4, 2])
    run.log('Customers Compras - altas: Sale Promedio',   tab.iloc[5, 2])
    run.log('High-volume customers: error (%)',        tab.iloc[6, 2])


    run.complete()
    

if __name__ == '__main__':
    main()