import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Tuple

def unpivotDataFrame(summary          : pd.DataFrame, 
                     numberOfPeriods  : int, 
                     n_cols           : int, 
                     predictionType   : str,
                     positionInSummary: int) -> pd.DataFrame:

    """
    This function transform the pivot table
    to a normal data frame

    Parameters:
    -----------
    summary: pd.DataFrame
        A data frame to be transformed
    numberofPeriods: int
        Number of periods to be predicted
    n_cols: int
        Number of columns to be added
    predictionType: str
        A string with the option forecast prediction or predicted order prediction
    positionInSummary: int
        Position of this value (forecast or suggested)
    
    Returns:
    --------
    unpivotDF: pd.DataFrame
        The data frame transformed

    How to use:
    -----------
    suggestedOrder = unpivotDataFrame(summary_new, numberOfPeriods, n_cols, 'Suggested', n_cols - 1)
    """
    unpivotDF = pd.melt(summary, 
        id_vars    = summary.columns[summary.shape[1] - 4:],
        value_vars = summary.columns[[positionInSummary + i for i in range(0, numberOfPeriods * n_cols, n_cols)]],
        value_name = 'Sale',
        var_name   = f'Date_{predictionType}',
        )
    unpivotDF[f'Date_{predictionType}'] = unpivotDF[f'Date_{predictionType}'].apply(lambda string: string[len(string) - 10: ])
    unpivotDF.sort_values(by = ['Agency', 'Client', 'Product', f'Date_{predictionType}'], inplace = True) 
    return unpivotDF

def summary_suggested_order(data_train          : pd.DataFrame, 
                            predictions         : List[List[float]],  
                            weights             : List[List[float]], 
                            model_client        : List[List[float]], 
                            suggested           : List[List[float]],
                            pct_sales           : List[float],
                            data_test           : pd.DataFrame  = pd.DataFrame(),
                            )  -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    This function returns a summary with information 
    about suggested order, forecast predictions, slopes
    for each series, percentages sales calculated, optimal weigths
    for each series and the data test

    Parameters:
    -----------
    data_train: pd.DataFrame
        A data frame with data for train the model
    predictions: List[List[float]]
        A list with predictions obtained from model
    weights: List[List[float]]
        A list with the optimal weights calculated for
        the model
    model_client: List[List[float]]
        A list with the slopes for each series
    suggested: List[List[float]]
        A list with suggested order
    pct_sales: List[float]
        A list with percentage sales calculated
        for the model
    data_test: pd.DataFrame
        A data frame with data for test
    
    Returns:
    --------
    summary: pd.DataFrame
        A summary with all of this information 
    suggestedTable: pd.DataFrame
        A table with Agency, Rute, Customer product
        and Suggested order values
    
    How to use:
    -----------
    summary, finalTable = summary_suggested_order(complete_data, futurePredictions, pesos, model_client, suggested, pct_sales)
    """
    numberOfPeriods = len(predictions[0])
    summary = pd.DataFrame({}, index = data_train.index)

    for i in range(len(data_train)):
        for j in range(numberOfPeriods):
            deltaTime = (datetime.strptime(data_train.columns[1],  '%Y-%m-%d %H:%M:%S') - datetime.strptime(data_train.columns[0], '%Y-%m-%d %H:%M:%S')).days

            futureDate = str(datetime.strptime(data_train.columns[-1], '%Y-%m-%d %H:%M:%S') + timedelta(days = j + deltaTime))[:10]

            if len(data_test) != 0:
                summary.loc[summary.index[i], f'data_test_period_{futureDate}', ]  = data_test.iloc[i, j]
            summary.loc[summary.index[i], f'prediction_period_{futureDate}']       = predictions[i][j]
            summary.loc[summary.index[i], f'perfect_client_period_{futureDate}']   = model_client[i][j]
            summary.loc[summary.index[i], f'suggested_order_period_{futureDate}']  = suggested[i][j]
            
        for j in range(len(weights[0])):
            summary.loc[summary.index[i], f'weigth_model_{j + 1}' ] = weights[i][j] 
    summary['Percantage_Sales'] = pct_sales

    summary_new = summary.reset_index()
    summary_new['Agency']  = summary_new['unique_id'].str.split('_', expand = True)[1]
    summary_new['Rute']    = summary_new['unique_id'].str.split('_', expand = True)[2]
    summary_new['Client']  = summary_new['unique_id'].str.split('_', expand = True)[3]
    summary_new['Product'] = summary_new['unique_id'].str.split('_', expand = True)[4]
    
    summary_new.drop(columns = ['unique_id'], inplace = True)

    n_cols = 4 if len(data_test) != 0 else 3
    forecastPrediction  = unpivotDataFrame(summary_new, numberOfPeriods, n_cols, 'Forecast', n_cols - 3)
    suggestedOrder      = unpivotDataFrame(summary_new, numberOfPeriods, n_cols, 'Suggested', n_cols - 1)

    suggestedTable = pd.merge(forecastPrediction, 
                              suggestedOrder, 
                              how = 'inner', 
                              left_on = ['Rute', 'Agency', 'Client', 'Product', 'Date_Forecast'], 
                              right_on = ['Rute', 'Agency', 'Client', 'Product',  'Date_Suggested'],
                              suffixes = ['_Prediction', '_Suggested']).drop(columns = ['Date_Suggested'])

    return summary, suggestedTable
