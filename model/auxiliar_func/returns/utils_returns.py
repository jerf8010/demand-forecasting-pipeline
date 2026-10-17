import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def informe_returns(data_returns             : pd.DataFrame,  
                  pctReturnsAcceptable     : float, 
                  days_ini               : int  = 24, 
                  days_fin               : int  = 0, 
                  dev_mode               : bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Function that obtains the summary for data with returns

    Parameters:
    ----------
    data_returns: pd.DataFrame
        A pivot data frame with daily data and with returns data
    pctReturnsAcceptable: float
        The percentage of returns acceptable (0.05)
    days_ini: int = 24
        Starting period for  calculation
    days_fin: int = 0
        Ending period for  calculation
    dev_mode: bool = False
        If it is True two bar charts are deployed
        One for normal or anormal returns
        and oter for anormal returns classification
    
    Returns:
    ---------
    norm_anorm: pd.DataFrame
        A data frame with the quantity of customers with 
        normal returns < pctReturnsAcceptable
        and anormal returns > pctReturnsAcceptable
    summaryReturns: pd.DataFrame
        A data frame with a summary of number of clients
        with low, mid and high returns above pctReturnsAcceptable

    How to use:
    -----------
    norm_anorm, summaryReturns = informe_returns(data_returns,
                                             0.05,
                                             )
    """
    
    data_returns = data_returns.iloc[ : , data_returns.shape[1] - days_ini : data_returns.shape[1] - days_fin]
    
    # Customers with sales in the period
    customersConSales = []
    for customer in data_returns.index:
        if data_returns[data_returns > 0].T.sum()[customer] > 0:
            customersConSales.append(customer)
    
    data_returns                 = data_returns[data_returns.index.isin(customersConSales)]
    totalCustomersConReturns      = abs(data_returns[data_returns < 0].T.sum() ) 
    pctReturns                   = totalCustomersConReturns / data_returns[data_returns > 0].T.sum()
    
    # Split customers by normalised return rate in the period
    lowReturns, highReturns, cat_extra_returns = [], [], []

    for _, customer in enumerate(pctReturns.index):
        if 0 < pctReturns[customer] <= pctReturnsAcceptable:
            lowReturns.append(customer)
        elif  pctReturns[customer] > pctReturnsAcceptable:
            highReturns.append(customer)
        else:
            cat_extra_returns.append(totalCustomersConReturns[customer])

    norm_anorm = pd.DataFrame({
        'returns_norm' : [len(lowReturns)],
        'returns_anor' : [len(highReturns)],
        'no_returns_in_period' : [len(cat_extra_returns)]
    }, index = ['no_customers'])
    
    if dev_mode:
        ax = sns.barplot(x = ['pct <= ' + str(pctReturnsAcceptable), 'pct > '  + str(pctReturnsAcceptable), 'no_returns_in_period'], 
                         y = [round(len(lowReturns)/len(data_returns), 2),  round(len(highReturns)/len(data_returns), 2), round(len(cat_extra_returns)/len(data_returns), 2)],
                         palette = 'flare')
        plt.ylim([0, 1])
        plt.title(f'Return percentage (tot_returns / tot_sales) % \n over a period of {days_ini - days_fin} days')
        ax.bar_label(ax.containers[0]);
    
    # Customers with abnormal return rate (pctReturns > 0.05)
    pct_returns_extra = pctReturns - 0.05

    lowReturnsUpAcceptablePct, midReturnsUpAcceptablePct, highReturnsUpAcceptablePct = [], [], []
    for customer in highReturns:
        if pct_returns_extra[customer] <= 0.1:
            lowReturnsUpAcceptablePct.append(customer)
        elif (pct_returns_extra[customer] > 0.1) and (pct_returns_extra[customer] < 0.2):
            midReturnsUpAcceptablePct.append(customer)
        else:
            highReturnsUpAcceptablePct.append(customer)
            
    summaryReturns = pd.DataFrame({
        'low_loses':  [ len(lowReturnsUpAcceptablePct)],
        'mid_loses':  [ len(midReturnsUpAcceptablePct)],
        'high_loses': [ len(highReturnsUpAcceptablePct)]
    }, index = ['no_customers'])
    
    summaryReturns['Total'] = sum(summaryReturns.iloc[0, :])
    return norm_anorm, summaryReturns
    
   
