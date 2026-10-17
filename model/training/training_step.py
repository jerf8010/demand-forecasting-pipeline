import os, sys
import argparse
import joblib
import json
import pandas as pd
#from   tools  import get_metrics
from   training_utils import fit_model

from darts.models.forecasting.croston import Croston
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing

from   azureml.core.run      import Run
from   datetime              import datetime


import warnings
warnings.filterwarnings("ignore")

def pipeline_text():
    return f'>>> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} Azure ML Training Step:'


def main():
    
    print(f"{pipeline_text()} Get context... ", sep=' ', end='', flush=True)
    run = Run.get_context()
    # get argpars
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--step_output", type=str, help=("output for passing data to next step"))
    parser.add_argument("--step_input",  type=str,  help=("input for passing data to next step"))
    args = parser.parse_args()
    step_output_path = args.step_output
    step_input_path = args.step_input
    print('done \n')


    print(f"{pipeline_text()} Getting argpars parameters... ", sep=' ', end='', flush=True)
    # Load the training parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
        week       = pars["preprocess"]["parameters"]["week"]
        # Log the training parameters
        print(f"\n \t - Parameters: {train_args} \n done \n")
        for (k, v) in train_args.items():
            run.log(k, v)
            run.parent.log(k, v)
    except KeyError:
        print("\n \t - Error: Could not load preprocess values from file \n")
        train_args = {}
    

    print(f"{pipeline_text()} Getting  input datasets and parameters... ", sep = ' ', end='', flush=True)
    # Link dataset to the step run so it is trackable in the UI

    model_input_path = os.path.join(step_input_path, f'pivot_df{week}.pkl')
    pivot_df     = joblib.load(model_input_path)

    print("done \n")

    # Train Croston model
    print(f"{pipeline_text()} Training Model... ", sep = ' ', end='', flush=True)
    futurePredictions, validationPredictions, \
        pesos, model_client, suggested, pct_sales = fit_model(data_for_train = pivot_df,
                                                               darts_models = [
                                                                                  KalmanForecaster(dim_x = 1),
                                                                                  Croston(version = 'sba'),
                                                                                  RandomForest(lags = 1),
                                                                                  FFT()
                                                                              ],
                                                               params = train_args["parameters"])

    print('done \n')

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok = True)
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(
        value=futurePredictions, 
        filename=os.path.join(step_output_path, f'futurePredictions{week}.pkl')
        )
    joblib.dump(
        value=futurePredictions, 
        filename=os.path.join('outputs', f'futurePredictions{week}.pkl')
        )
    joblib.dump(
        value=validationPredictions, 
        filename=os.path.join(step_output_path, f'validationPredictions{week}.pkl')
        )
    joblib.dump(
        value=validationPredictions, 
        filename=os.path.join('outputs', f'validationPredictions{week}.pkl')
        )
    joblib.dump(
        value=pesos, 
        filename=os.path.join(step_output_path, f'pesos{week}.pkl')
        )
    joblib.dump(
        value=pesos, 
        filename=os.path.join('outputs', f'pesos{week}.pkl')
        )

    joblib.dump(
        value=model_client, 
        filename=os.path.join(step_output_path, f'model_client{week}.pkl')
        )
    joblib.dump(
        value=model_client, 
        filename=os.path.join('outputs', f'model_client{week}.pkl')
        )
    joblib.dump(
        value=suggested, 
        filename=os.path.join(step_output_path, f'suggested{week}.pkl')
        )
    joblib.dump(
        value=suggested, 
        filename=os.path.join('outputs', f'suggested{week}.pkl')
        )
    joblib.dump(
        value=pct_sales, 
        filename=os.path.join(step_output_path, f'pct_sales{week}.pkl')
        )
    joblib.dump(
        value=pct_sales, 
        filename=os.path.join('outputs', f'pct_sales{week}.pkl')
        )
   
    #run.tag("run_type", value="train")
    #print(f"tags now present for run: {run.tags}")
    #run.complete()

if __name__ == '__main__':
    main()
