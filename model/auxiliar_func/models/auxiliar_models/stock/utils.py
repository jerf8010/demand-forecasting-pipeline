# How much stock does a customer hold when a product is delivered?
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from typing import List, Tuple



def estimationStockPerClient(data_customer : pd.Series, 
                             pct_sale: float 
                            ) -> Tuple[List[float], List[float]]:

    """
    This function estimates the stock for a customer
    before te visit and a normal stock for him (stock plus 
    the quantity of product)

    Parameters:
    -----------
    data_customer: pd.Series
        A pandas series with historical data for customer
    pct_sale: float
        Percentage of sales
    
    Returns:
    -------
    stock: List[float]
        Estimated stock 
    normalStock: List[float]
        Estimated normal  stock

    How to use:
    -----------
    stock, normalStock = estimationStockPerClient(data_customer, 0.2)
    """
    customer_diff_cero = list(data_customer[data_customer != 0])
    lista_customer = list(data_customer.values)
    sale_per_week = [ round(i * pct_sale, 2) for i in customer_diff_cero ]   
    first_sale_index =  lista_customer.index(customer_diff_cero[0])

    stock = [lista_customer[i] for i in range(len(lista_customer[ :first_sale_index + 1]))]
    
    for j in range(len(customer_diff_cero) - 1 ):
        actual_sale_index = lista_customer.index(customer_diff_cero[j] )
        next_sale_index   = lista_customer.index(customer_diff_cero[j + 1], actual_sale_index + 1, len(lista_customer)  ) 
        # Con periodo sin sales
        if next_sale_index - actual_sale_index > 1:
            sale_actualizada = customer_diff_cero[j]
            for _ in range(1, next_sale_index - actual_sale_index + 1):
                sale_actualizada -= sale_per_week[j] 
                if sale_actualizada < 0: sale_actualizada = 0
                stock.append(sale_actualizada) 
        else:
            if customer_diff_cero[j + 1] < customer_diff_cero[j] + stock[j]: 
                stock.append(customer_diff_cero[j] + stock[j] - customer_diff_cero[j + 1])
            else:
                stock.append(0)

        index_to_cut = lista_customer.index(customer_diff_cero[j + 1]) if customer_diff_cero[j + 1] != customer_diff_cero[j ] else lista_customer.index(customer_diff_cero[j + 1]) + 1
        lista_customer = lista_customer[index_to_cut: ]

    if len(stock) < len(data_customer):
        sale_actualizada = customer_diff_cero[-1]
        for _ in range(1, len(data_customer) - len(stock) + 1):
           sale_actualizada -= sale_per_week[-1]
           stock.append(sale_actualizada)
           
    normalStock = [stock[i] + list(data_customer.values)[i] for i in range(len(stock))] 
    return stock, normalStock

def estimationStock(data: pd.DataFrame, 
                    pct_sale :  float = 0.10
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    This function estimates the stock and normal stock 
    for all the data

    Parameters:
    ----------
    data: pd.DataFrame
        A data frame with all the historical data
    pct_sale: float
        Percentage of sales
    
    Returns:
    --------
    stocksDataFrame: pd.DataFrame
        A data frame with the information about
        the estimated stock
    normalStocksDataFrame: pd.DataFrame
        A data frame with the information about
        the estimated normal stock
    
    How to use:
    -----------
    stocksDataFrame, normalStocksDataFrame = estimationStockPerClient(pd.Series(dataTrainList), 0.2)
    """
    estimations = Parallel(n_jobs = -1)(delayed(estimationStockPerClient)(
                                                                    data.iloc[fila, :],
                                                                    pct_sale 
                                                                    ) for fila in range(data.shape[0]))

    stocks = [estimations[i][0] for i in range(len(estimations))]
    normalStocks = [estimations[i][1] for i in range(len(estimations))]

    stocksDataFrame = pd.DataFrame(stocks, index = data.index)
    normalStocksDataFrame = pd.DataFrame(normalStocks, index = data.index)

    return stocksDataFrame, normalStocksDataFrame

def simplePredictionBasedOnStock(fila: int,
                                 data_train: pd.DataFrame,
                                 n_pred : int,
                                 pct_sale  : float
                                 ) -> List[float]:

    """
    This function obtains a prediction of the sales
    for a customer based on a linear regression 
    on the normal stock

    Parameters:
    ----------
    fila: int
        Position in the data farame of the customer
    data_train: pd.DataFrame
        A data frame with data for train
    n_pred: int
        Lenght of the period to predict
    pct_sale: float
        PErcentage of sales for this customer
    
    Returns:
    --------
    predictedSales: List[float]
        A list of predicted sales based on normal stock
    
    How to use:
    ----------
    predSale = simplePredictionBasedOnStock(0, 
                                            dataForTrain, 
                                            validation_window = 4,
                                            pct_sale = 0.2)
    """

    dataTrainList   = list(data_train.iloc[fila, :])
    predictedSales = []


    for i in range(n_pred):
        _, normalStockList = estimationStockPerClient(pd.Series(dataTrainList), pct_sale)
        X = np.array(normalStockList[:len(normalStockList) - 1 ]).reshape(-1, 1)
        y = pd.Series(dataTrainList[1: len(dataTrainList)])

        lr = LinearRegression()
        lr.fit(X,  y)
        predictedSales.append(lr.predict(np.array(normalStockList[len(normalStockList) - 1]).reshape(-1, 1))[0])
        dataTrainList.append(predictedSales[-1])

    return predictedSales

def gridSearch(data : pd.DataFrame, 
               validation_window: int ,
               pct_sales: List[float] )  -> List[float]:
    
    """
    This function makes a search in the 
    grid of percentages sales to obtain the optimal
    percentage of sale for every customer

    Parameters:
    -----------
    data: pd.DataFrame
        Complete data 
    validation_window: int
        Lenght of the validation data to search
        optimal hyperparameter
    pct_sales: List[float]
        A grid of percentages sales to be considerated
    
    Returns:
    --------
    choicedPercentageSales: List[float]
        A list of optimal percentage sales for every customer

    How to use:
    -----------
    choicedPercentageSales  = gridSearch(data, validation_window, pct_sales)
    """
    dataForTrain = data.iloc[:, :data.shape[1] - validation_window]
    dataForValidation = data.iloc[:, data.shape[1] - validation_window:]

    def auxFunction(fila               : int, 
                    dataForTrain       : pd.DataFrame, 
                    validation_window  : int,
                    pct_sales          : List[float]) -> float:
        
        
        errors = [mean_absolute_error(dataForValidation.iloc[fila, :], simplePredictionBasedOnStock(fila, 
                                                   dataForTrain, 
                                                   validation_window,
                                                   pct_sale)) for pct_sale in pct_sales]

        return pct_sales[errors.index(np.min(errors))]

    choicedPercentageSales = Parallel(n_jobs = -1)(delayed(auxFunction)( fila,
                                                              dataForTrain,
                                                              validation_window,
                                                              pct_sales) for fila in range(len(dataForTrain)))
    return choicedPercentageSales
            
def predictionBasedOnStock(data : pd.DataFrame, 
                           validation_window: int  = 4,
                           n_pred : int = 1, 
                           pct_sales: List[float] = [0.2],
                           ) -> Tuple[List[float], List[float], List[float]]:

    """
    This function returns the predicted values based 
    on the stock estimation

    Parameters:
    -----------
    data: pd.DataFrame
        Complete data 
    n_pred: int
        Lenght of the period to predict    
    validation_window: int
        Lenght of the validation data to search
        optimal hyperparameter
    pct_sales: List[float]
        A grid of percentages sales to be considerated

    Returns:
    --------
    futurePredictions: List[float]
        A list with future predictions based on stock
    validationPredictions: List[float]
        A list with validation predictions based on stock
    choicedPercentageSales: List[float]
        A list with optimal percentage sales for every series
    --------
    """
    choicedPercentageSales  = gridSearch(data, validation_window, pct_sales)
    validationPredictions = Parallel(n_jobs = -1)(delayed(simplePredictionBasedOnStock)(fila, 
                                                   data, 
                                                   validation_window,
                                                   choicedPercentageSales[fila]) for fila in range(len(data)))

    futurePredictions  = Parallel(n_jobs = -1)(delayed(simplePredictionBasedOnStock)(fila, 
                                                   data, 
                                                   n_pred,
                                                   choicedPercentageSales[fila]) for fila in range(len(data)))
    return futurePredictions, validationPredictions, choicedPercentageSales

