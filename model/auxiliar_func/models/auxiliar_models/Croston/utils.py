import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import List
from joblib import Parallel, delayed

def modified_Croston_TSB(data_for_train                   : pd.Series, 
                         n_pred                           : int            = 4, 
                         alpha                            : float          = 0.4, 
                         beta                             : float          = 0.4, 
                         proba_type                      = 'default', # : (str | float)   
                         interval_to_look_symmetry        : int            = 5, 
                         border_determine_zero_pred      = 'default')  -> pd.Series: # : (str |float)   
    
    """
    This is a modified version of Croston TSB
    model for intermitent time series

    Parameters:
    -----------
    data_for_train: pd.DataFrame
        A data frame with data for train
    n_pred: int
        Time interval to predict
    alpha: float
        A parameter for traditional Croston TSB
    beta: float
        A parameter for traditional Croston TSB
    proba_type: (str|float)
        Parameters that indicates what probability
        will be considerated in first instance
    interval_to_look_symmetry: int
        Interval in the past to look for symmetry
        respect with present value
    border_determine_zero_pred: (str|float)
        Parameter that indicates how to round a value
        to zero
    
    Returns:
    --------
    predictions: pd.Series
        A pandas series with de predictions calculated

    How to use:
    -----------
    predictions_client = list(modified_Croston_TSB(client_for_train, 
                                                   n_pred = 1, 
                                                   alpha  = 0.5, 
                                                   beta   = 0.3))[0]
    """
    data_for_train_array = np.array(data_for_train) # Transform the input into level numpy array
    data_train_length = len(data_for_train_array) # Historical period length
    
    proba_ceros = data_for_train_array[data_for_train_array == 0].shape[0] / data_for_train_array.shape[0]
    proba_no_ceros = 1 - proba_ceros
    
    data_for_train_array = np.append(data_for_train_array, [np.nan] * (   n_pred  + 1 ) ) # Append np.nan into the demand array to cover future periods
    
    #level (level), probability(probability) and forecast (forecast)
    level, forecast, probability = np.full((3, data_train_length +    n_pred + 1 ), np.nan)

    # Initialization
    first_occurence = np.argmax(data_for_train_array[:data_train_length]>0)
    level[0] = data_for_train_array[first_occurence]
    
    if   proba_type == 'default'       :    probability[0] = 1 / (1 + first_occurence)
    elif proba_type == 'proba_ceros'   :    probability[0] = proba_ceros
    elif proba_type == 'proba_no_ceros':    probability[0] = proba_no_ceros
    
    forecast[0] = probability[0] * level[0]
                     
    # Create all the t+1 forecasts
    for t in range(0, data_train_length + 1): 
        if data_for_train_array[t] > 0:
            level[t + 1] = alpha * data_for_train_array[t] + (1 - alpha) * level[t] 
            probability[t + 1] = beta * (1) + (1 - beta) * probability[t]  
        else:
            level[t + 1] = level[t]
            probability[t + 1] = (1 - beta) * probability[t]
        
        forecast[t] = round( probability[t] * level[t] )


    # Future Forecast
    '''
    We assume a slight symmetry between present and future data;
    this could be improved by finding the 4-element window from the past with the highest likelihood
    '''
    # data_for_train_array[data_train_length ] = data_for_train_array[data_train_length - 2 ]
    
    if border_determine_zero_pred == 'default':
        # Threshold determined by the share of zeros: 0.2 can be tuned  # train[train.shape[0] - 5 : train.shape[0] - 1][train > 0].shape[0]
        if  data_for_train_array[data_for_train_array > 0].shape[0] / data_for_train_array.shape[0] < 0.2: 
            border_determine_zero_pred = data_for_train_array.mean()
        else: 
            border_determine_zero_pred = 1 # Values below 1 are treated as 0
    
    flipped_list = list(reversed(list(data_for_train_array[:data_train_length - interval_to_look_symmetry - 1])))  # Window where a matching value is searched
    try: 
        flipped_list.index(data_for_train_array[data_train_length - 1])
        data_for_train_array[data_train_length] = flipped_list[flipped_list.index(data_for_train_array[data_train_length - 1]) - 1] 
        index_coincidence = flipped_list.index(data_for_train_array[data_train_length - 1]) 
        k = 1
        appro = True
    except:
        data_for_train_array[data_train_length ] = data_for_train_array[data_train_length - 2 ]
        i = 4
        appro = False 

    for t in range(data_train_length , data_train_length +    n_pred + 1):
        if data_for_train_array[t - 1] >= border_determine_zero_pred: 
            level[t] = alpha * data_for_train_array[t - 1] + (1 - alpha) * level[t - 1] 
            probability[t] = beta * (1) + (1 - beta) * probability[t - 1]  
        else:
            level[t] = level[t - 1]
            probability[t] = (1 - beta) * probability[t - 1]  
        
        forecast[t] = round( probability[t] * level[t], 2 )
        
        if appro:
            data_for_train_array[t] = data_for_train_array[data_train_length  - interval_to_look_symmetry - index_coincidence + k - 1]
            k += 1
            #index_coincidence += 1
        else:      
            data_for_train_array[t] =  data_for_train_array[t - i]  # Reuse past values as future predictions
            i += 2
    df = pd.DataFrame.from_dict({"Demand":data_for_train_array, "Forecast": forecast, "Probability": probability, "Level": level,"Error": data_for_train_array-forecast})
    predictions = df.iloc[len(df) - n_pred : , 1]
    return predictions

def parameters_search(client_for_train     : pd.Series,
                      validation_window    : int          = 4,
                      parameters           : dict         = {
                        'alphas' : [0.1, 0.5, 0.8],
                        'betas'  : [0.1, 0.5, 0.8]
                      }
                    ) -> List[float]:
    """
    This function makes a hyperparameters search
    to find the best alpha and beta parameters that
    adjust to validation data

    Parameters:
    ----------
    client_for_train: pd.Series
        A pandas series with historical data of the customer
        of our interes
    validation_window: int
        Size of the validation data
    parameters: Dict[str, List[float]]
        A dictionary with the grid values for alpha and beta
    
    Returns:
    best_alpha_beta: List[float]
        A list with optimal alpha and beta parameters found
    
    How to use:
    -----------
    best_alpha_beta  = parameters_search(client_for_train, 4, {
                            'alphas' : [0.1, 0.5, 0.8],
                            'betas'  : [0.1, 0.5, 0.8]
                           })
    """
    data_for_train, data_for_validation = client_for_train[ : len(client_for_train) - validation_window], \
                                          client_for_train[ len(client_for_train) - validation_window :]

    errors                 = []
    alpha_beta_combination = []
    for alpha in parameters['alphas']:
        for beta in parameters['betas']:
            predictions = modified_Croston_TSB(data_for_train = data_for_train, 
                                               n_pred = validation_window,
                                               alpha  = alpha,
                                               beta   = beta
                                               )
            errors.append( mean_absolute_error(data_for_validation, predictions) )
            alpha_beta_combination.append([alpha, beta])

    index_error_min = errors.index(min(errors))
    best_alpha_beta = alpha_beta_combination[index_error_min]
    return best_alpha_beta

def train_croston_modified(client                : str,
                           data_for_train        : pd.DataFrame,
                           validation_window     : int            = 4,
                           parameters            : dict           = {
                            'alphas' : [0.1, 0.5, 0.8],
                            'betas'  : [0.1, 0.5, 0.8]
                           })  -> float:
    
    """
    This function obtains the predictions
    for the customer of interest obtaining his optimal 
    parameters alpha and beta and is auxiliar in the 
    Parallelize process

    Parameters:
    -----------
    client: str
        Unique id of the series of interest
    data_for_train: pd.DataFrame
        A data frame with data for train
    validation_window: int
        Size of the validation data
    parameters: Dict[str, List[float]]
        A dictionary with the grid values for alpha and beta

    Returns:
    ---------
    predictions_client: float
        A value with the future prediction after
        hyperparameter search
    
    How to use:
    -----------
    final_predictions = Parallel(n_jobs = -1)(delayed(train_croston_modified)(client,
                                                                              data_for_train,
                                                                              validation_window,
                                                                              parameters) for client in data_for_train.index)
    """
    client_for_train   = data_for_train.loc[client, :]
    best_alpha_beta    = parameters_search(client_for_train, validation_window, parameters)
    predictions_client = list(modified_Croston_TSB(client_for_train, 
                                                   n_pred = 1, 
                                                   alpha = best_alpha_beta[0], 
                                                   beta = best_alpha_beta[1]))[0]

    return predictions_client


def fit_modified_croston(data_for_train      : pd.DataFrame,
                         validation_window   : int            = 4,
                         parameters          : dict           = {
                            'alphas' : [0.1, 0.5, 0.8],
                            'betas'  : [0.1, 0.5, 0.8]
                           }
                        ) -> (List[float]):
    
    """
    This function calculates the predictions based on
    modified Croston TSB

    Parameters:
    -----------
    data_for_train: pd.DataFrame
        A data frame with data for train
    validation_window: int
        Size of the validation data
    parameters: Dict[str, List[float]]
        A dictionary with the grid values for alpha and beta

    Returns:
    --------
    final_predictions: List[float]
        A list with the predictions for all the series

    How to use:
    ----------
    predictions =  modified_Croston_TSB(data_for_train, 
                                        4, 
                                        {
                            'alphas' : [0.1, 0.5, 0.8],
                            'betas'  : [0.1, 0.5, 0.8]
                           })
    """
    final_predictions = Parallel(n_jobs = -1)(delayed(train_croston_modified)(client,
                                                                              data_for_train,
                                                                              validation_window,
                                                                              parameters) for client in data_for_train.index)
    return final_predictions






