import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from typing import List, Tuple

# TODO: EFICIENTAR
def client_tendency(customer_no_returns    : pd.Series, 
                    n_pred               : int,
                    tendency_window      : int) -> Tuple[float, float, np.ndarray]:
    
    """
    This function calculates a regression line for 
    the customer selected by user

    Parameters:
    ------------
    customer_no_returns: pd.Series
        A pandas series of a customer with no returns
    n_pred: int
        Time interval to predict
    tendency_window: int
        Period to calculate the regression line
    
    Returns:
    ----------
    intercept: float
        The value for the intercept of the reg line
    coeficient: float
        The value for the slope of the reg line
    tendency: np.ndarray
        Predicted values in the regression

    How to use:
    -----------
    intercept, coeficient, _ = client_tendency(data_no_returns.loc[client, :], 1, data_no_returns.shape[0])
    """
    
    history       = customer_no_returns[len(customer_no_returns) - tendency_window: ]
    no_zero_sales = history[history != 0]

    X = np.array(range(0, len(no_zero_sales) )).reshape(-1, 1)
        
    regressor = LinearRegression()
    regressor.fit(X, no_zero_sales)
        
    intercept   = regressor.intercept_
    coeficient  = regressor.coef_[0]
    tendency    = regressor.predict(np.array(range(len(no_zero_sales), len(no_zero_sales) + n_pred  )).reshape(-1, 1)).reshape(-1, 1)

    return intercept, coeficient, tendency

def clients_classification_by_coeficient(coeficient : float) -> str:
    """
    This function classifies to the customer 
    based on his slope tendency

    Parameters:
    -----------
    coeficient: float
        Slope value of interest
    
    Returns:
    --------
    :str
        A string with customer classification
        "uptrend"   : slope > 1
        "stable"    : slope between 0 and 1
        "downtrend" : Other cases

    How to use:
    ------------
    coeficient_class = clients_classification_by_coeficient(0.5)
    """
    if (coeficient > 1)                                : return 'uptrend'
    elif (coeficient >= 0) and (coeficient <= 1)       : return 'stable'
    else                                               : return 'downtrend'

def model_client_slope(data_no_returns     : pd.DataFrame, 
                       tendency_window    : int
                       ) -> Tuple[float, float, float, float]:
    """
    With this function we obtain the perfect
    customer slope for each category:
        Starting tendency negative
        Starting tendency low
        Starting tendency mid
        Starting tendency high
    
    Parameters:
    -----------
    data_no_returns: pd.DataFrame
        A data frame with data with series with no negative values
    tendency_window: int
        Period to calculate the regression line

    How to use:
    -----------
    model_client_start_negative, model_client_start_low, \
        model_client_start_mid, model_client_start_high = model_client_slope(data_no_returns, data_no_returns.shape[0])

    """
        
    stable_start_tendency_negative, stable_start_tendency_low, stable_start_tendency_mid, stable_start_tendency_high  = [], [], [], []
    for client in data_no_returns.index:
        intercept, coeficient, _ = client_tendency(data_no_returns.loc[client, :], 1, tendency_window) # <--------------------

        coeficient_class = clients_classification_by_coeficient(coeficient)

        if (coeficient_class == 'stable') and (intercept < 0)                           : stable_start_tendency_negative.append(coeficient)
        if (coeficient_class == 'stable') and ((intercept > 0) and (intercept <= 10))   : stable_start_tendency_low.append(coeficient)
        if (coeficient_class == 'stable') and ((intercept > 10) and (intercept <= 20))  : stable_start_tendency_mid.append(coeficient)
        if (coeficient_class == 'stable') and (intercept > 20)                          : stable_start_tendency_high.append(coeficient)

    model_client_start_negative = np.mean(stable_start_tendency_negative) if len(stable_start_tendency_negative) != 0 else 0
    model_client_start_low      = np.mean(stable_start_tendency_low) if len(stable_start_tendency_low) != 0 else 0
    model_client_start_mid      = np.mean(stable_start_tendency_mid) if len(stable_start_tendency_mid) != 0 else 0
    model_client_start_high     = np.mean(stable_start_tendency_high) if len(stable_start_tendency_high) != 0 else 0

    return model_client_start_negative, model_client_start_low, model_client_start_mid, model_client_start_high

straight_line = lambda intercept, slope, X: intercept + slope * X

def suggested_by_client(client           :str, 
                        data_no_returns   : pd.DataFrame, 
                        n_pred           : int, 
                        tendency_window  : int) -> List[float]:

    """
    This function obtain the suggested order based
    on increase sales with the value obtained
    from the  straight line with the slope of the respective 
    perfect customer and its proper starting point of the tendency
    and is auxiliar for parallelization process

    Parameters:
    -----------
    client: str
        Unique id for the customer of interest
    data_no_returns: pd.DataFrame
        A data frame with data with series with no negative values
    n_pred: int
        Time interval to predict
    tendency_window: int
        Period to calculate the regression line

    Returns: 
    ---------
    : List[float]
        A list with the value obtained from the suggested order
    
    How to use:
    -----------
    prediction = suggested_by_client(client, data_no_returns, 1, )
    """
    customer_no_returns = data_no_returns.loc[client, :]

    intercept, coeficient, tendency = client_tendency(customer_no_returns, n_pred, tendency_window)
    coeficient_class = clients_classification_by_coeficient(coeficient)
    model_client_start_negative, model_client_start_low, model_client_start_mid, model_client_start_high = model_client_slope(data_no_returns, tendency_window)
   
    history       = customer_no_returns[len(customer_no_returns) - tendency_window: ]
    no_zero_sales = history[history != 0]

    X = np.array(range(0, len(no_zero_sales) + n_pred)).reshape(-1, 1)

    if (coeficient_class == 'uptrend') or (coeficient_class == 'stable'): 
        return (tendency.reshape(1, -1).tolist()[0] + np.max(no_zero_sales)) / 2
    if (coeficient_class == 'downtrend') and (intercept < 0): 
        return straight_line(intercept, model_client_start_negative, X)[len(no_zero_sales):].reshape(1, -1).tolist()[0]
    if (coeficient_class == 'downtrend') and ((intercept > 0) and (intercept <= 10)): 
        return straight_line(intercept, model_client_start_low, X)[len(no_zero_sales):].reshape(1, -1).tolist()[0]
    if (coeficient_class == 'downtrend') and ((intercept > 10) and (intercept <= 20)): 
        return straight_line(intercept, model_client_start_mid, X)[len(no_zero_sales):].reshape(1, -1).tolist()[0]
    if (coeficient_class == 'downtrend') and (intercept > 20): 
        return straight_line(intercept, model_client_start_high, X)[len(no_zero_sales):].reshape(1, -1).tolist()[0]

def suggested_order(data_no_returns              : pd.DataFrame,
                    n_pred                      : int           = 1,
                    tendency_window             : int           = 4 * 7 * 6) -> List[List[float]]:
    
    """
    This function obtains the suggeested order
    for all the series with no returns

    Parameters:
    ----------
    data_no_returns: pd.DataFrame
        A data frame with data with series with no negative values
    n_pred: int
        Time interval to predict
    tendency_window: int
        Period to calculate the regression line

    Returns:
    ---------
    suggested: List[List[float]]
        A list with the values of the suggested order
    
    How to use:
    -----------
    sugg = suggested_order(data_no_returns, 1, data_no_returns.shape[0])
    """
    suggested = Parallel(n_jobs = -1)(delayed(suggested_by_client)(client, data_no_returns, n_pred, tendency_window) for client in data_no_returns.index)
        
    return suggested
   

