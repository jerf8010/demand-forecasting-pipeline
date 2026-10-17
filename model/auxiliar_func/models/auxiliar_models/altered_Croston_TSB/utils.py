import math
import numpy as np
import pandas as pd

def modified_Croston_TSB(data_train,
                bias_ = False, 
                extra_periods = 1, 
                alpha_beta_default = {
                    'alpha' : 0.7,
                    'beta'  : 0.3
                }
               ):
    
    lista = data_train.values
    no_ceros = lista[lista != 0]
    # if len(no_ceros) != 0: # Avoids an error for the benchmark
    indice = list(lista).index(no_ceros[0])
    train  = data_train.values
    # test   = data_test.iloc[fila, :]
    
    d = np.array(train) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    
    ###################################################################
    c = np.array(data_train.values)
    no_ceros = c[c != 0].shape[0] / c.shape[0]
    # for i in  c[c != 0]:
    #   sales.append(i) # CORREGIR es d no f
        
    # Alpha/beta grid for each customer
    if alpha_beta_default == 'None':
        if np.mean(c[c != 0]) <= 5:
            alpha, beta = 0.7, 0.3
        elif np.mean(c[c != 0]) > 5 and np.mean(c[c != 0]) <= 8:
            alpha, beta = 0.3, 0.2
        else:
            alpha, beta = 0.3, 0.7
    else:
        alpha = alpha_beta_default['alpha']
        beta  = alpha_beta_default['beta']
        
    if bias_:
        # Probability of non-zeros in the series from the first non-zero index
        # Probabilidades:
        aux_c_n = 0
        aux_c_c = 0
        aux_n_c = 0
        aux_n_n = 0
        for i in range(c.shape[0] - 1):
            if (c[i] != 0) and (c[i + 1] != 0):
                aux_n_n += 1
            elif (c[i] == 0) and (c[i + 1] != 0):
                aux_c_n += 1
            elif (c[i] != 0) and (c[i + 1] == 0):
                aux_n_c += 1
            elif (c[i] == 0) and (c[i + 1] == 0):
                aux_c_c += 1
                
        sigue_n_n =  aux_n_n / c.shape[0]
        sigue_n_c =  aux_n_c / c.shape[0]
        sigue_c_c =  aux_c_c / c.shape[0]
        sigue_c_n =  aux_c_n / c.shape[0]
       
        if c[c.shape[0] - 1] != 0:
            bias =  - np.mean(c[c != 0]) * ( sigue_n_n + sigue_n_c  )
        else:
            bias =    np.mean(c[c != 0]) * ( sigue_c_n + sigue_c_c  )
    else:
        bias = 0
        
    #######################################################################
    
    d = np.append(d, [np.nan]*(extra_periods + 1)) # Append np.nan into the demand array to cover future periods
    
    #level (a), probability(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods + 1),np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]
                 
    # Create all the t+1 forecasts
    for t in range(0, cols + 1): 
        if d[t] > 0:
            a[t+1] = alpha * d[t] + (1 - alpha) * a[t] 
            p[t+1] = beta * (1) + (1 - beta) * p[t]  
        else:
            a[t+1] = a[t]
            p[t+1] = (1 - beta) * p[t]
        
        f[t] = p[t] * a[t] 

    f[cols+1:cols+extra_periods + 1] = math.ceil( f[cols] ) + bias 
    if f[cols+1:cols+extra_periods + 1] < 0: f[cols+1:cols+extra_periods + 1] = 0

    df = pd.DataFrame.from_dict({"Demand":d, "Forecast": f, "Probability":p, "Level":a,"Error":d-f})
    return df.iloc[-1, 1]