import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from typing import List, Tuple



def dataframe_returns(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function extract series with
    positive values and series with negative values

    Parameters:
    -----------
    data: pd.DataFrame
        Complete data
    
    Returns:
    --------
    customers_returns: pd.DataFrame
        A data frame with series with some negative value
    customers_no_returns: pd.DataFrame
        A data frame with series with positive values

    How to use:
    -----------
    customers_returns, customers_no_returns = dataframe_returns(complete_data)
    """
    returns = []
    for customer in data.index:
        for date in data.columns:
            if data.loc[customer, date] < 0:
                returns.append(customer)
                continue
                
    customers_returns     = data[data.index.isin(returns)]
    customers_no_returns = data[~data.index.isin(returns)]
    return customers_returns, customers_no_returns

def visualize(data_no_returns     : pd.DataFrame,
              intercept_neg      : List[List[float]],
              intercept_low      : List[List[float]],
              intercept_mid      : List[List[float]],
              intercept_high     : List[List[float]]
              ):

    """
    This function is auxiliar for visualize
    a summary of information for data without returns

    Parameters:
    -----------
    intercept_neg      : List[List[float]]
        A list with customer with intercept (of the tendency) negative
    intercept_low      : List[List[float]]
        A list with customer with intercept (of the tendency) low
    intercept_mid      : List[List[float]]
        A list with customer with intercept (of the tendency) mid
    intercept_high     : List[List[float]]
        A list with customer with intercept (of the tendency) high
    """
    high_dism_int_neg, mid_dism_int_neg, low_dism_int_neg, \
        stable_int_neg, high_aum_int_neg = intercept_neg[0], intercept_neg[1], intercept_neg[2], intercept_neg[3], intercept_neg[4]

    high_dism_int_low, mid_dism_int_low, low_dism_int_low, \
        stable_int_low, high_aum_int_low = intercept_low[0], intercept_low[1], intercept_low[2], intercept_low[3], intercept_low[4]

    high_dism_int_mid, mid_dism_int_mid, low_dism_int_mid, \
        stable_int_mid, high_aum_int_mid = intercept_mid[0], intercept_mid[1], intercept_mid[2], intercept_mid[3], intercept_mid[4]

    high_dism_int_high, mid_dism_int_high, low_dism_int_high, \
        stable_int_high, high_aum_int_high = intercept_high[0], intercept_high[1], intercept_high[2], intercept_high[3], intercept_high[4]

    fig = plt.figure(figsize = (10, 20))
    axs0 = plt.subplot(411)
    axs1 = plt.subplot(412)
    axs2 = plt.subplot(413)
    axs3 = plt.subplot(414)
    sns.barplot(x = ['high_dism', 'mid_dism', 'low_dism', 'stable', 'high_aum'], 
                     y = [round(len(high_dism_int_neg)/len(data_no_returns), 3), 
                          round(len(mid_dism_int_neg)/len(data_no_returns),3), 
                          round(len(low_dism_int_neg)/len(data_no_returns), 3),
                          round(len(stable_int_neg)/len(data_no_returns), 3),
                          round(len(high_aum_int_neg)/len(data_no_returns), 3)],
                    palette = 'flare', ax = axs0)
    axs0.bar_label(axs0.containers[0])
    axs0.set_title('Intercept Neg')
    sns.barplot(x = ['high_dism', 'mid_dism', 'low_dism', 'stable', 'high_aum'], 
                     y = [round(len(high_dism_int_low)/len(data_no_returns), 3), 
                          round(len(mid_dism_int_low)/len(data_no_returns),3), 
                          round(len(low_dism_int_low)/len(data_no_returns), 3),
                          round(len(stable_int_low)/len(data_no_returns), 3),
                          round(len(high_aum_int_low)/len(data_no_returns), 3)],
                    palette = 'flare', ax = axs1)
    axs1.bar_label(axs1.containers[0])
    axs1.set_title('Intercept Low Pos (< 10)')
    sns.barplot(x = ['high_dism', 'mid_dism', 'low_dism', 'stable', 'high_aum'], 
                     y = [round(len(high_dism_int_mid)/len(data_no_returns), 3), 
                          round(len(mid_dism_int_mid)/len(data_no_returns), 3), 
                          round(len(low_dism_int_mid)/len(data_no_returns), 3),
                          round(len(stable_int_mid)/len(data_no_returns),3),
                          round(len(high_aum_int_mid)/len(data_no_returns), 3)],
                    palette = 'flare', ax = axs2)
    axs2.bar_label(axs2.containers[0])
    axs2.set_title('Intercept Mid Pos (< 20)')
    sns.barplot(x = ['high_dism', 'mid_dism', 'low_dism', 'stable', 'high_aum'], 
                     y = [round(len(high_dism_int_high)/len(data_no_returns), 3), 
                          round(len(mid_dism_int_high)/len(data_no_returns), 3), 
                          round(len(low_dism_int_high)/len(data_no_returns), 3),
                          round(len(stable_int_high)/len(data_no_returns), 3),
                          round(len(high_aum_int_high)/len(data_no_returns), 3)],
                    palette = 'flare', ax = axs3)
    axs3.bar_label(axs3.containers[0])
    axs3.set_title('Intercept High Pos (> 20)');

def informe_no_returns(data_no_returns   : pd.DataFrame, 
                      days_ini         : int = 24, 
                      days_fin         : int = 0, 
                      dev_mode         : bool = False):
    
    """
    This function makes a summary report of 
    data sin returns, calculating the number of series
    in each category. These categories are made based on
    the tendency line (starting value of tendency and the
    slope of tendency)

    Parameters:
    -----------
    data_no_returns: pd.DataFrame
        A data frame with series with positive values
    days_ini: int
        The starting point of the interval for calculate tendency
    days_fin: int
        The ending point of the interval for calculate tendency
    dev_mode: bool
        A parameter that indicates if user wish to visualize charts
        of this summary
    
    Returns:
    --------
    summary_no_returns: pd.DataFrame
        A table with info about the quantity of customers for each category
    predsDf: pd.DataFrame
        A data frame with predictions using the tendency
    high_aum_slope: List[float]
        A list with series with slope > 1
    stable_slope: List[float]
        A list with series with slope between 0 and 1
    low_dism_slope: List[float]
        A list with series with slope between 0 and -1
    mid_dism_slope: List[float]
        A list with series with slope between -1 and -2.5
    high_dism_slope: List[float]
        A list with series with slope < -2.5
    
    How to use:
    -----------
    summary_no_returns, predsDf, high_aum_slope, \
        stable_slope, low_dism_slope, mid_dism_slope, high_dism_slope = informe_no_returns(data_no_returns,
                                                                                          24,
                                                                                          0,
                                                                                          True)


    """
    data_no_returns = data_no_returns.iloc[ : , data_no_returns.shape[1] - days_ini : data_no_returns.shape[1] - days_fin]

    
    intercepts = []
    
    stable_int_neg, stable_int_low, stable_int_mid, stable_int_high                           = [], [], [], []
    high_aum_int_neg, high_aum_int_low, high_aum_int_mid, high_aum_int_high                   = [], [], [], []
    low_dism_int_neg,  low_dism_int_low,  low_dism_int_mid, low_dism_int_high                 = [], [], [], []
    mid_dism_int_neg,  mid_dism_int_low,  mid_dism_int_mid, mid_dism_int_high                 = [], [], [], []
    high_dism_int_neg, high_dism_int_low, high_dism_int_mid,  high_dism_int_high              = [], [], [], []
    
    int_low_stable_trend_slope, int_low_low_trend, int_low_mid_trend, int_low_high_trend      = [], [], [], []
    int_mid_stable_trend_slope, int_mid_low_trend, int_mid_mid_trend, int_mid_high_trend      = [], [], [], []
    int_high_stable_trend_slope, int_high_low_trend, int_high_mid_trend, int_high_high_trend  = [], [], [], []


    straight_line = lambda intercept, slope, X: intercept + slope * X
    preds = []
    for fila, customer in enumerate(data_no_returns.index):
        history = data_no_returns.iloc[fila, :]
        sales = history[history != 0]
        X = np.array(range(1, len(sales) + 1)).reshape(-1, 1)
        
        lr = LinearRegression()
        lr.fit(X, sales)
        
        X_tendency = np.array(range(1, len(sales) + 7)).reshape(-1, 1)
        preds.append(straight_line(lr.intercept_, lr.coef_[0], X_tendency)[-6:])

        intercepts.append(lr.intercept_)
        if (lr.intercept_ < 0 ):
            if (lr.coef_[0] >= 0) and (lr.coef_[0] <= 1):
                stable_int_neg.append(customer)
            elif (lr.coef_[0] > 1):
                high_aum_int_neg.append(customer)
            elif (lr.coef_[0] < 0) and (lr.coef_[0] >= -1):
                low_dism_int_neg.append(customer)
            elif (lr.coef_[0] < -1) and (lr.coef_[0] >= -2.5):
                mid_dism_int_neg.append(customer)
            elif (lr.coef_[0] < -2.5):
                high_dism_int_neg.append(customer)
                
        elif (lr.intercept_ >= 0) and (lr.intercept_ < 10):
            if (lr.coef_[0] >= 0) and (lr.coef_[0] <= 1):
                stable_int_low.append(customer)
                int_low_stable_trend_slope.append(lr.coef_[0])
            elif (lr.coef_[0] > 1):
                high_aum_int_low.append(customer)
            elif (lr.coef_[0] < 0) and (lr.coef_[0] >= -1):
                low_dism_int_low.append(customer)                
                int_low_low_trend.append(lr.predict(X))
            elif (lr.coef_[0] < -1) and (lr.coef_[0] >= -2.5):
                mid_dism_int_low.append(customer)                
                int_low_mid_trend.append(lr.predict(X))
            elif (lr.coef_[0] < -2.5):
                high_dism_int_low.append(customer)                
                int_low_high_trend.append(lr.predict(X))     
        
        elif (lr.intercept_ >= 10) and (lr.intercept_ < 20):
            if (lr.coef_[0] >= 0) and (lr.coef_[0] <= 1):
                stable_int_mid.append(customer)
                int_mid_stable_trend_slope.append(lr.coef_[0])
                
            elif (lr.coef_[0] > 1):
                high_aum_int_mid.append(customer)
            elif (lr.coef_[0] < 0) and (lr.coef_[0] >= -1):
                low_dism_int_mid.append(customer)
                int_mid_low_trend.append(lr.predict(X))

            elif (lr.coef_[0] < -1) and (lr.coef_[0] >= -2.5):
                mid_dism_int_mid.append(customer)
                int_mid_mid_trend.append(lr.predict(X))

            elif (lr.coef_[0] < -2.5):
                high_dism_int_mid.append(customer)                
                int_mid_high_trend.append(lr.predict(X))
                
        else:
            if (lr.coef_[0] >= 0) and (lr.coef_[0] <= 1):
                stable_int_high.append(customer)
                int_high_stable_trend_slope.append(lr.coef_[0])
            elif (lr.coef_[0] > 1):
                high_aum_int_high.append(customer)
            elif (lr.coef_[0] < 0) and (lr.coef_[0] >= -1):
                low_dism_int_high.append(customer)                
                int_high_low_trend.append(lr.predict(X))

            elif (lr.coef_[0] < -1) and (lr.coef_[0] >= -2.5):
                mid_dism_int_high.append(customer)                
                int_high_mid_trend.append(lr.predict(X))

            elif (lr.coef_[0] < -2.5):
                high_dism_int_high.append(customer)
                int_high_high_trend.append(lr.predict(X))

    high_aum_slope =  high_aum_int_high + high_aum_int_low + high_aum_int_mid + high_aum_int_neg
    stable_slope   =  stable_int_high + stable_int_low + stable_int_mid + stable_int_neg
    low_dism_slope =  low_dism_int_high + low_dism_int_low + low_dism_int_mid + low_dism_int_neg
    mid_dism_slope =  mid_dism_int_high + mid_dism_int_low + mid_dism_int_mid + mid_dism_int_neg
    high_dism_slope =  high_dism_int_high + high_dism_int_low + high_dism_int_mid + high_dism_int_neg


    summary_no_returns = pd.DataFrame(columns = ['slopes_neg_low', 'slopes_neg_mid', 'slopes_neg_high', 'slope_stable', 'slopes_positive', 'total_inter',], 
                                     index = ['intercept low', 'intercept mid', 'intercept high', 'intercept neg', 'total slopes'])
    
    summary_no_returns.iloc[0, 0]  = len(low_dism_int_low)
    summary_no_returns.iloc[0, 1]  = len(mid_dism_int_low)
    summary_no_returns.iloc[0, 2]  = len(high_dism_int_low)
    summary_no_returns.iloc[0, 3]  = len(stable_int_low)
    summary_no_returns.iloc[0, 4]  = len(high_aum_int_low)
    
    summary_no_returns.iloc[1, 0]  = len(low_dism_int_mid)
    summary_no_returns.iloc[1, 1]  = len(mid_dism_int_mid)
    summary_no_returns.iloc[1, 2]  = len(high_dism_int_mid)
    summary_no_returns.iloc[1, 3]  = len(stable_int_mid)
    summary_no_returns.iloc[1, 4]  = len(high_aum_int_mid)
    
    summary_no_returns.iloc[2, 0]  = len(low_dism_int_high)
    summary_no_returns.iloc[2, 1]  = len(mid_dism_int_high)
    summary_no_returns.iloc[2, 2]  = len(high_dism_int_high)
    summary_no_returns.iloc[2, 3]  = len(stable_int_high)
    summary_no_returns.iloc[2, 4]  = len(high_aum_int_high)
    
    summary_no_returns.iloc[3, 0]  = len(low_dism_int_neg)
    summary_no_returns.iloc[3, 1]  = len(mid_dism_int_neg)
    summary_no_returns.iloc[3, 2]  = len(high_dism_int_neg)
    summary_no_returns.iloc[3, 3]  = len(stable_int_neg)
    summary_no_returns.iloc[3, 4]  = len(high_aum_int_neg)

    summary_no_returns.iloc[:, 5]  = np.array([sum(summary_no_returns.iloc[i, :-1]) for i in range(summary_no_returns.shape[0])])
    summary_no_returns.iloc[4, :]  = np.array([sum(summary_no_returns.iloc[:-1, i]) for i in range(summary_no_returns.shape[1])])
    
    
    preds = [preds[i].reshape(1, 6)[0] for i in range(len(preds))]
    predsDf = pd.DataFrame(index = data_no_returns.index, columns = [0, 1, 2, 3, 4, 5])
    for i in range(len(predsDf)):
        predsDf.iloc[i, :] = preds[i]

    if dev_mode:
        intercept_neg  = [high_dism_int_neg, mid_dism_int_neg, low_dism_int_neg, stable_int_neg, high_aum_int_neg]    
        intercept_low  = [high_dism_int_low, mid_dism_int_low, low_dism_int_low, stable_int_low, high_aum_int_low]    
        intercept_mid  = [high_dism_int_mid, mid_dism_int_mid, low_dism_int_mid, stable_int_mid, high_aum_int_mid]    
        intercept_high = [high_dism_int_high, mid_dism_int_high, low_dism_int_high, stable_int_high, high_aum_int_high] 

        visualize(data_no_returns, intercept_neg, intercept_low, intercept_mid, intercept_high)   

    
    return summary_no_returns, predsDf, high_aum_slope, stable_slope, low_dism_slope, mid_dism_slope, high_dism_slope
