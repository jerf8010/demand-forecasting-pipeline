import pandas as pd
import numpy as np
from itertools import product

from darts import TimeSeries
from sklearn.metrics import mean_absolute_error

from joblib import Parallel, delayed

from darts.models.forecasting.croston import Croston
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing

from auxiliar_func.models.auxiliar_models.stock import predictionBasedOnStock
from auxiliar_func.models.auxiliar_models.suggested import suggested_order
from auxiliar_func.models.auxiliar_models.Croston   import parameters_search, modified_Croston_TSB
from auxiliar_func.auxiliars import preprocess_text

# Modelos permitidos
from typing import List, Union, Tuple
ModelForecaster = Union[Croston, FFT, RandomForest, 
                        LightGBMModel, KalmanForecaster, 
                        StatsForecastAutoARIMA, ExponentialSmoothing]


def transform_data_to_timeseries(data_for_train     : pd.DataFrame, 
                                 client             : str, 
                                 validation_window  : int) -> Tuple[TimeSeries, pd.Series]:

    """
    This function transform a pd.Series 
    of interest in a TimeSeries in darts format
    and divide the data in training and validation

    Parameters:
    -----------
    data_for_train: pd.DataFrame
        A data frame with data for train
    client: str
        unique id of the series to be transformed
    validation_window: int
        Integer number that indicates the size of validation data
    
    Returns:
    ----------
    series_for_train: TimeSeries
        The training series part transformed to 
        TimeSeries format
    data_for_validation: pd.Series
        The validation series
    
    How to use:
    series_for_train, data_for_validation = transform_data_to_timeseries(data_for_train, client, validation_window)

    -----------

    """

    # Transform data into the format required by Darts
    all_data_train = pd.DataFrame(data_for_train.loc[client, :]).reset_index()#.drop(columns = 'level_0')
    
    # Split into training and validation
    data_for_train, data_for_validation = all_data_train.iloc[:len(all_data_train) - validation_window],  \
                                          all_data_train.iloc[len(all_data_train)  - validation_window:, 1]
    
    # Transformacion en TimeSeries class
    series_for_train = TimeSeries.from_dataframe(df = data_for_train, time_col = 'index', value_cols = client, fill_missing_dates = True, fillna_value = 0, freq = 'W-THU')

    return series_for_train, data_for_validation


def simple_model(series_for_train:   TimeSeries, 
                 darts_models    :   List[ModelForecaster], 
                 n_pred          :   int) -> List[float]:

    """
    This function trains every model introduced
    by user and obtains the predictions for the
    period n_pred

    Parameters:
    -----------
    series_for_train: TimeSeries
        A series in TimeSeries (darts) format to train
    darts_models: List[ModelForecaster]
        List of models introduced by user to be considered
        in the final model
    n_pred: int
        Predictions will be obtained in this period
    
    Returns:
    --------
    predictions: List[float]
        A list with the predictions for each model and for
        this series
    
    How to use:
    -----------
    preds_for_validation = simple_model(series_for_train, 
                                        [
                                            KalmanForecaster(dim_x = 1),
                                            Croston(version = 'sba'),
                                            StatsForecastAutoARIMA()
                                        ],
                                        4)

    """
    predictions = []

    for model in darts_models: 
        model.fit(series_for_train)
        simple_prediction = model.predict(n_pred)

        predictions.append(simple_prediction.pd_dataframe().iloc[:, 0].values)

    return predictions

def train_croston_modified(client_for_train      : TimeSeries,
                           validation_window     : int            = 4,
                           parameters            : dict           = {
                            'alphas' : list(np.linspace(0.1, 0.9, 9)),
                            'betas'  : list(np.linspace(0.1, 0.9, 9))
                           })  -> Tuple[pd.Series, List[float]]:

    """
    This function calls the necesary function to train the modified Croston TSB

    Parameters:
    ----------
    client_for_train: TimeSeries
        A series in TimeSeries (darts) format to train
    validation_window: int
        Integer number that indicates the size of validation data
    parameters: Dict[str, List[float]]
        Alpha and beta grid to be considered
    
    Returns:
    --------
    predictions_client: pd.Series
        The predictions based in modified Croston TSB
    best_alpha_beta: List[float]
        Optimal alpha beta founded

    How to use:
    ------------
    preds_modified_croston, best_alpha_beta = train_croston_modified(series_for_train, 4)

    """
    client_for_train_series = client_for_train.pd_series()
    best_alpha_beta    = parameters_search(client_for_train_series, validation_window, parameters)
    predictions_client = modified_Croston_TSB(client_for_train_series, 
                                              n_pred = validation_window, 
                                              alpha = best_alpha_beta[0], 
                                              beta = best_alpha_beta[1])

    return predictions_client, best_alpha_beta

def search_weights(series_for_train      : TimeSeries, 
                   data_for_validation   : pd.Series, 
                   darts_models          : List[ModelForecaster], 
                   weights               : List[float], 
                   validation_window     : int
                ) -> Tuple[List[float], List[float]]:

    """
    This function is in charge of estimate
    the optimal weights and the optimal alpha
    and beta parameters

    Paramters:
    ----------
    series_for_train: TimeSeries
        A series in TimeSeries (darts) format to train
    data_for_validation: pd.Series
        A series with data for validation
    darts_models: List[ModelForecaster]
        List of models introduced by user to be considered
        in the final model
    weights: List[float] = [0., 1.]
        List of float numbers to be estimated in each series to
        obtain the optimal model combination from modified Croston TSB
        and darts models list (pondered weights)
    validation_window: int
        Integer number that indicates the size of validation data
    
    Returns:
    --------
    best_weights: List[float]
        Optimal weights founded 
    best_alpha_beta: List[float]
        Optimal alpha, beta parameters founded
    
    How to use:
    -----------
    founded_weights, best_alpha_beta = search_weights(series_for_train, 
                                                      data_for_validation, 
                                                      [
                                                        KalmanForecaster(dim_x = 1)
                                                      ], 
                                                      [0., 0.1, 0.8], 
                                                      4)
    """
    # Model predictions on the validation set
    preds_for_validation   = simple_model(series_for_train, darts_models, validation_window)
    preds_modified_croston, best_alpha_beta = train_croston_modified(series_for_train, validation_window)
    preds_for_validation.append(preds_modified_croston.values)

    # Weight search
    errors        = []
    weights_used  = []
    for weight_combination in product(weights, repeat = len(darts_models) + 1):
        if (sum(weight_combination) == 0): continue

        prediction_for_weights = 0.
        for i in range(len(weight_combination)): 
            prediction_for_weights += weight_combination[i] * preds_for_validation[i]

        prediction_for_weights = prediction_for_weights / sum(weight_combination)    
        weights_used.append(weight_combination)
        errors.append(mean_absolute_error(data_for_validation, prediction_for_weights))
         
    # Weights with the lowest error
    best_weights_index  : int          = errors.index(min(errors))
    best_weights        : List[float]  = weights_used[best_weights_index]

    return best_weights, best_alpha_beta

def prediction_model_train(client              : str, 
                           data_for_train      : pd.DataFrame, 
                           darts_models        : List[ModelForecaster], 
                           best_weights        : List[float],
                           best_alpha_beta     : List[float] ,
                           n_pred              : int) -> List[float]:
    """
    This function is used to obtain the predictions
    based in statistical models

    Parameters:
    -----------
    client: str
        Unique id for each series (predictions is for this series)
    data_for_train: pd.DataFrame
        A data frame with data for train
    darts_models: List[ModelForecaster]
        List of models introduced by user to be considered
        in the final model
    best_weights: List[float]
        Weights for every model to be considered for every model
    best_alpha_beta: List[float]
        Alpha and beta paramters to be considered for this
        series in modified Croston TSB
    n_pred: int
        Predictions will be obtained in this period
    
    Return:
    --------
    predictions: List[float]
        A list with the number of predictions equal to n_pred
        periods for this unique id

    How to use:
    -----------    
    validation_prediction = prediction_model_train(client, 
                                                   data_for_train, 
                                                   darts_models, 
                                                   [0.2, 0., 0.7], 
                                                   [0.7, 0.3], 
                                                   4
                                                   )
    """
    # Obtaining the pd.Series for this client and transforming to
    # darts time series format
    data_client       = data_for_train.loc[client, :]
    data_client_df    = pd.DataFrame(data_for_train.loc[client, :]).reset_index()
    data_timeseries   = TimeSeries.from_dataframe(df = data_client_df, time_col = 'index', value_cols = client, fill_missing_dates = True, fillna_value = 0, freq = 'D')


    # Future prediction
    prediction_for_models  = simple_model(data_timeseries, darts_models, n_pred)
    prediction_for_croston = list(modified_Croston_TSB(data_client, n_pred, alpha = best_alpha_beta[0], beta = best_alpha_beta[1]))[:n_pred]
    prediction_for_models.append(prediction_for_croston)
    
    # Weighted-average prediction
    prediction = [0.] * n_pred
    for j in range(len(prediction)):
        for i in range(len(best_weights)): 
            prediction[j] += best_weights[i] * prediction_for_models[i][j] 
        prediction[j] = prediction[j] / sum(best_weights)

    return prediction

def validation_model_train(
                client              : str, 
                data_for_train      : pd.DataFrame, 
                darts_models        : List[ModelForecaster], 
                weights             : List[float], 
                validation_window   : int
            ) -> Tuple[List[float], List[float], List[float]]:

    """
    Auxiliar function to obtain optimal weights 
    and optimal alpha and beta for modified Croston tSB
    This function is auxiliar when we parallelize this procedure

    Parameters:
    -----------
    client: str
        Unique id for each series
    data_for_train: pd.DataFrame
        A data frame with data for train
    darts_models: List[ModelForecaster]
        List of models introduced by user to be considered
        in the final model
    validation_window: int
        Integer number that indicates the size of validation data

    Returns:
    --------
    validation_prediction: List[float]
        Predictions based on statistical models
        for validation period
    founded_weights: List[float]
        Optimal weights for the series with this unique id
    best_alpha_beta: List[float]
        Optimal alpha and beta for the series with this unique id
    
    How to use:
    -----------
    validationPredictions_and_weigths = Parallel(n_jobs = -1)(delayed(validation_model_train)(client, 
                                                                         data_for_train, 
                                                                         darts_models , 
                                                                         weights, 
                                                                         validation_window) for client in data_for_train.index)

    """
    series_for_train, data_for_validation = transform_data_to_timeseries(data_for_train, client, validation_window)

    founded_weights, best_alpha_beta = search_weights(series_for_train, data_for_validation, darts_models, weights, validation_window)
    
    validation_prediction = prediction_model_train(client, data_for_train, darts_models, founded_weights, best_alpha_beta, validation_window)

    founded_weights = list(founded_weights)

    return validation_prediction, founded_weights, best_alpha_beta

def validation_fit_model(data_for_train              : pd.DataFrame,
                        validation_predictions_stock : List[List[float]],
                        darts_models                 : List[ModelForecaster],
                        weights                      : List[float] = [0., 1.],
                        validation_window            : int = 4
                        )->Tuple[List[List[float]], List[float], List[float]]:
    """
    This function fit the model on the validation data to obtain 
    optimal alpha and beta for croston TSB modified and the optimal weitghts 
    for every model

    Parameters:
    -----------
    data_for_train: pd.DataFrame
        A data frame with data for train
    validation_prediction_stock: List[float]
        List of predictions for data validation based on the stock estimation
    darts_models: List[ModelForecaster]
        List of models introduced by user to be considered
        in the final model
    weights: List[float] = [0., 1.]
        List of float numbers to be estimated in each series to
        obtain the optimal model combination from modified Croston TSB
        and darts models list (pondered weights)
    validation_window: int = 4
        Integer number that indicates the size of validation data
    
    Returns:
    --------
    validationPredictions: List[List[float]]
        List of predictions (forecast and predictions
        based on stock) in the validation period
    founded_weights: List[float]
        List of optimal weights for each series
    founded_alpha_beta
        List of founded parameters alpha and beta, 
        adjusted for each series in modified Croston tSB
    
    How to use:
    ------------
     validationPredictions, founded_weights, \
        founded_alpha_beta  = validation_fit_model(data_for_train,
                                                   validationPredictionsBasedOnStock,
                                                   [
                                                        KalmanForecaster(dim_x = 1),
                                                        Croston(version = 'sba'),
                                                        #StatsForecastAutoARIMA()
                                                   ],
                                                   [0., 0.2, 0.5, 0.7, 1],
                                                   4
                                                )

    """

    # Validation and grid search process
    validationPredictions_and_weigths = Parallel(n_jobs = -1)(delayed(validation_model_train)(client, 
                                                                         data_for_train, 
                                                                         darts_models , 
                                                                         weights, 
                                                                         validation_window) for client in data_for_train.index)

    # Validation (Prediction for metrics)
    validationPredictions   = [[(validationPredictions_and_weigths[i][0][j] + validation_predictions_stock[i][j]) / 2 for j in range(validation_window)]
                                        for i in range(len(validationPredictions_and_weigths))
                                        ]

    founded_weights     = [validationPredictions_and_weigths[i][1] for i in range(len(validationPredictions_and_weigths))]
    founded_alpha_beta  = [validationPredictions_and_weigths[i][2] for i in range(len(validationPredictions_and_weigths))]

    return validationPredictions, founded_weights, founded_alpha_beta

def fit_model(data_for_train             : pd.DataFrame,
              darts_models               : List[ModelForecaster],
              params                     : dict
             ) -> (Tuple[
                List[List[float]], List[List[float]], 
                List[List[float]], List[List[float]], 
                List[List[float]], List[float]] ):
    """
    This is the main function of the model 
    and calculates prediction based on stock,
    prediction based on modified croston TSB and the models introduced by the user
    and the suggested order based in ponderated weight introduced by user

    Parameters:
    -----------
    data_for_train: pd.DataFrame
        A data frame with the data for train, 
        includes data for validation
    darts_models: List[ModelForecaster]
        A list with models from Darts library to be consider part
        of the final model introduced by user
    params: dict
        n_pred: int = 1
            Number of periods to consider for future prediction
        weights: List[float] = [0, 1]
            List of float numbers to calculate the importance
            of every model (modified Croston TSB and darts models) 
            in forecast (ponderated weight)
        pct_sales: List[float] = [0.2]
            List of float numbers to be considered to estimate what 
            is the percentage of sales for one series
        validation_window: int = 4
            The number of periods to estimate hyperparameters and optimal weights
        tendency_window: int = 4 * 7 * 6
            The period in wich we calculate the tendency to estimate 
            the suggested order and the perfect customer
        prediction_importance: float = 1.
            This is the weight for the prediction in final suggested order
        sales_increase_importance: float = 0.
            This is the risk level to increase sales in final suggested order

    Returns:
    --------
    futurePredictions: List[List[float]]
        List of predictions from statistical models, in the future
    validationPredictions: List[List[float]]
        List of predictions from statistical models, in the validation periods
    founded_weights: List[List[float]]
        List of the weights founded for every series in hyperparameter search
    model_client_behaviour: List[List[float]]
        List of the slopes of the respective perfect customer
    suggested_orders: List[List[float]]
        List of the suggest orders for every series (the final suggested order)
    pctSales: List[float]
        List with the optimal sales percentage for every series

    How to use:
    -----------
    %%time
    # Data to use
    validationWindow = 4
    complete_data = pivot_series_weekly.iloc[:, ]
    train_data    = complete_data.iloc[:, :complete_data.shape[1] - validationWindow]
    test_data     = complete_data.iloc[:,  complete_data.shape[1] - validationWindow:]

    # Fitting model
    futurePredictions, validationPredictions, \
    pesos, model_client, suggested, pct_sales = fit_model(
                                    # Data Parameter
                                    data_for_train = complete_data, 
                                    # Forecast Parameters
                                    darts_models = [
                                        KalmanForecaster(dim_x = 1),
                                        Croston(version = 'sba'),
                                        #StatsForecastAutoARIMA()
                                    ], 
                                    n_pred                      = 1,
                                    weights                     = [0., 0.2, 0.5, 0.7, 1],
                                    pct_sales                   = [0.2], 
                                    validation_window           = validationWindow,
                                    # Bussiness Parameters
                                    tendency_window             = complete_data.shape[1],
                                    prediction_importance       = 0.7,
                                    sales_increase_importance   = 0.3)
    """

    
    n_pred                    = params['n_pred']
    weights                   = params['weights']
    pct_sales                 = params['pct_sales']
    validation_window         = params['validation_window']
    tendency_window           = params['tendency_window']
    prediction_importance     = params['prediction_importance']
    sales_increase_importance = params['sales_increase_importance']
    
    print(f'> {preprocess_text()} Training the model...')                                                                      

    print(f'>> {preprocess_text()} Getting stock-based predictions...')
    # Predictions Based On Stock
    futurePredictionsBasedOnStock, validationPredictionsBasedOnStock, \
        pctSales = predictionBasedOnStock(data_for_train, validation_window, n_pred, pct_sales) 
    print(f'>> Done')

    print(f'>> {preprocess_text()} Running hyperparameter and weight search for each model...')
    # Gridsearch process and validation predictions
    validationPredictions, founded_weights, \
        founded_alpha_beta  = validation_fit_model(data_for_train,
                                                   validationPredictionsBasedOnStock,
                                                   darts_models,
                                                   weights,
                                                   validation_window
                                                )
    print('>> Done')

    print(f'>> {preprocess_text()} Getting future predictions (stock and forecast)...')
    # Final prediction
    future_model_predictions = Parallel(n_jobs = -1)(delayed(prediction_model_train)(client,
                                                                                   data_for_train,
                                                                                   darts_models,
                                                                                   founded_weights[fila],
                                                                                   founded_alpha_beta[fila],
                                                                                   n_pred) for fila, client in enumerate(data_for_train.index))
    futurePredictions   = [[(future_model_predictions[i][j] + futurePredictionsBasedOnStock[i][j]) / 2 for j in range(n_pred)]
                                        for i in range(len(future_model_predictions))
                                        ]
    print('>> Done')

    print(f'>> {preprocess_text()} Computing suggested order...')
    print(f'>>> Importancia forecast {prediction_importance} - Importancia en incremento de sales {sales_increase_importance}')
    model_client_behaviour = suggested_order(data_for_train, n_pred, tendency_window)
    suggested_orders       = [[(prediction_importance * futurePredictions[i][j] + sales_increase_importance * model_client_behaviour[i][j]) / (prediction_importance + sales_increase_importance)
                                                            for j in range(n_pred)] 
                                                            for i in range(len(futurePredictions))]
    print('> Done \n')
    return futurePredictions, validationPredictions, founded_weights, model_client_behaviour, suggested_orders, pctSales

        
