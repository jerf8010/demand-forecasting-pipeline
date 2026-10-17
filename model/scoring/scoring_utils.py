import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error
from typing import Dict, List, Tuple

from auxiliar_func.auxiliars import preprocess_text

def tablas(
    data_train: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:

    """
    Function that classifies the customer in three categories 
    it depends on wich is the mean of their non zero sales

    Parameters:
    -----------
    data_train: pd.DataFrame
        Data Frame with data for train

    Returns:
    ----------
    (cantidades, table): Tuple[pd.DataFrame, Dict[str, List[str]]]
        cantidades: A data frame with the quantity of customers in each category
        table: A dictionary with the categories as keys and a list of 
        numbers of positions for the series in that category as values

    How to use:
    -----------
    cantidades, table = tablas(train)
    """

    # Customer segmentation for scoring
    table = {
        '(0, 5]': [],
        '(5, 8]': [],
        '(8, max]': [],
    }

    for fila in range(len(data_train)):
        lista = data_train.iloc[fila, :].values
        no_ceros = lista[lista != 0]

        if (np.mean(no_ceros) > 0) and (np.mean(no_ceros) <= 5):
            table['(0, 5]'].append(fila)
        elif (np.mean(no_ceros) > 5) and (np.mean(no_ceros) <= 8):
            table['(5, 8]'].append(fila)
        elif (np.mean(no_ceros) > 8):
            table['(8, max]'].append(fila)

    cantidades = pd.DataFrame({
        '(0, 5]': [len(table['(0, 5]'])],
        '(5, 8]': [len(table['(5, 8]'])],
        '(8, max]': [len(table['(8, max]'])]
    }, index = ['Customers'])
    
    return cantidades, table

def benchmark_score(train       : pd.DataFrame, 
                    benchmark   : int, 
                    n_pred      : int) -> List[List[float]]:

    """
    Auxiliar function for benchmark calculation for every series

    Parameters:
    ------------
    train: pd.DataFrame
         Data Frame with data for train
    benchmark: int
        Integer number with past data considered for calculation of benchmark
    n_pred: int
        Integer number of period of time to predict (lenght of data validation)

    Returns:
    --------
    predictionsBenchmark: List[List[float]]
        List with the mean of the n_pred past values for every series

    How to use:
    -----------
    predictions_bench = benchmark_score(train, benchmark, test.shape[1])
    """

    predictionsBenchmark = []
    for fila in range(len(train)):
        bench_data = train.iloc[fila, train.shape[1] - benchmark:]
        predictionsBenchmark.append([np.mean(bench_data)] * n_pred)
    return predictionsBenchmark  


def maeError(test           : pd.DataFrame, 
             predictions    : List[List[float]], 
             metricType     : str) -> float:

    """
    This function calculates the median or mean error between predictions and data test, it dependes on the metricType parameter

    Parameters:
    -----------
    test: pd.DataFrame
        Data Frame with data for test
    predictions: List[List]
        List with predictions values (Benchmark or Model predictions)
    metricType: str
        String with metric type: "median" or "mean"
    
    Returns:
    --------
    : float
        The mean or the median for the mean absolute error for every series
    
    How to use:
    -----------
    valueError = maeError(sub_test, model, 'median')
 
    """

    # metricType validation: only "mean" and "median"
    if metricType not in ['mean', 'median']:
        print('Invalid Metric')
        return
    # Mean absolute error for every series in data
    errors = [mean_absolute_error(test.iloc[i, :], predictions[i])  for i in range(len(test))]
    # Mean or median calculation
    return np.mean(errors) if metricType == 'mean' else np.median(errors)
        
    
def columna_tabla_final(interval            : str, 
                        train               : pd.DataFrame, 
                        test                : pd.DataFrame,           
                        predictions_bench   : List[List[float]], 
                        predictions         : List[List[float]], 
                        table               : Dict[str, List[int]])   -> List[float]:

    """
    This is an auxiliar function for the creation of the a column in the final table, depending on the interval introduced

    Parameters:
    ----------
    interval: str
        String with the customer category from which the information will be calculated (only ('(0, 5]', '(5, 8]', '(8, max]'))
    train: pd.DataFrame
        Data Frame with data for train
    test: pd.DataFrame
        Data Frame with data for test
    predictions_bench: List[List]
        List with benchmark predictions for every series
    predictions: List[List]
        List with model predictions for every series
    table:

    Returns:
    --------
    columnForFinalTable: List
        List with median and mean for MAE between benchmark and validation data and between prediction and validation data
        or None values in case there are no customers in this category

    How to use:
    ----------
    columna   = columna_tabla_final('(0, 5]',   train, test, predictions_bench, predictions, table)
    
    """

    # Obtaining rows: benchmark prediction, model predictions, customer in this category, sales and index 
    bench               = [predictions_bench[fila] for fila in table[interval]]
    model               = [predictions[fila] for fila in table[interval]]
    customers            = [fila for fila in table[interval]]
    sales              = [np.mean(train.iloc[fila, :][train.iloc[fila, :] != 0]) for fila in table[interval]]
    tableIntervaloIndex = [test.index[fila] for fila in table[interval]]


    # Column of the final table creation
    sub_test = test[test.index.isin(tableIntervaloIndex)]
    columnForFinalTable = []

    if len(sub_test) != 0:

        columnForFinalTable.append(round(maeError(sub_test, bench, 'median'), 2))
        columnForFinalTable.append(round(maeError(sub_test, model, 'median'), 2))

        columnForFinalTable.append(round(maeError(sub_test, bench, 'mean'), 2))
        columnForFinalTable.append(round(maeError(sub_test, model, 'mean'), 2)) 

        columnForFinalTable.append(len(customers))

        columnForFinalTable.append(round(np.mean(sales), 2))
        columnForFinalTable.append(round((median_absolute_error(sub_test, model) / np.mean(sales)) * 100, 2))

        return columnForFinalTable
    else:
        return [np.NaN, np.NaN, np.NaN, np.NaN, len(customers), round(np.mean(sales), 2), np.NaN]
        
        
    
def scoring(train                 : pd.DataFrame, 
            test                  : pd.DataFrame, 
            predictions           : List[List[float]]
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    This function creates the dataframes that summarize the metrics used to measure the model performance

    Parameters
    ----------
    train: pd.DataFrame
        Data for train
    test: pd.DataFrame
        Data for test
    predictions: List[List]
        List with predicted values

    Returns
    --------
    predictionsDtaFrame: pd.DataFrame
        A data Frame with predictions (list predictions transformed into a dataframe)
    globalErrors: pd.DataFrame
        A data frame with gloabl mean and median for benchmark and predictions
    finalTable: pd.DataFrame
        Only if tablas_info is True
        A data frame with mean and median for benchmark and predictions but with customer classification (No Zero sales mean)


    How to use
    ----------
    pred, mae, tab = scoring(train_data, 
                         test_data, 
                         validationPredictions, benchmark = 4, tablas_info = True)

    """
    benchmark   = test.shape[1]
    tablas_info = True 
    
    print(f'> {preprocess_text()} Computing scoring tables...')

    # Creation of new data frame based on test data frame
    predictionsDataFrame = test.copy()
    predictionsDataFrame.columns = test.columns
    for i, _ in enumerate(predictionsDataFrame):
        for j in range(len(predictions[i])):
            predictionsDataFrame.iloc[i, j] = predictions[i][j]
    

     # Benchmark computation
    predictions_bench = benchmark_score(train, benchmark, test.shape[1])

    # Summary tables
    
    print(f'>> {preprocess_text()}Getting summary...')
    cantidades, table = tablas(train)
    finalTable = pd.DataFrame(columns = ['(0, 5]', '(5, 8]', '(8, max]'], index = ['benchmark_error_median', 'model_error_median', 'benchmark_error_mean', 'model_error_mean', 'No Customers', 'Sale promedio', 'Error (%)'])
    finalTable['(0, 5]']   = columna_tabla_final('(0, 5]',   train, test, predictions_bench, predictions, table)
    finalTable['(5, 8]']   = columna_tabla_final('(5, 8]',   train, test, predictions_bench, predictions, table)
    finalTable['(8, max]'] = columna_tabla_final('(8, max]', train, test, predictions_bench, predictions, table)
    print('>> Done')
   
  
    # MAE for benchmark and model

    print(f'>> {preprocess_text()} Getting global errors...')
    globalErrors = pd.DataFrame(columns = ['benchmark', 'modelo'], index = ['mean_absolute_error', 'median_absolute_error'])
    globalErrors.iloc[0, 0] = maeError(test, predictions_bench, 'mean')
    globalErrors.iloc[1, 0] = maeError(test, predictions_bench, 'median')
    globalErrors.iloc[0, 1] = maeError(test, predictions, 'mean')
    globalErrors.iloc[1, 1] = maeError(test, predictions, 'median')
            
    print('> Done')
    return predictionsDataFrame, globalErrors, finalTable
