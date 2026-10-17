import numpy                    as np
import pandas                   as pd
import sys 
from sklearn.metrics import mean_squared_error, mean_absolute_error



def split_data(pivot_df: pd.DataFrame, test_size=None, test_frac=None) -> dict:
    pivot_df.set_index('unique_id', inplace=True)

    _, n_cols = pivot_df.iloc[:,:].values.shape 

    if test_size is None:
        if test_frac is  None:
            print('Both test_size and test_frac cannot be None')
            sys.exit()
        else:
            while np.rint(n_cols*test_frac) < 2:
                test_frac += 0.05

            n_test  = int(np.rint(n_cols*test_frac))
    else:
        n_test = test_size
   
    n_train = int(n_cols - n_test)

    train   = pivot_df.iloc[:, :n_train].reset_index()
    test    = pivot_df.iloc[:, -n_test:].reset_index()

    data    = {'train': train, 
               'test' : test}
    return data


def get_metrics(y_true, y_benchmark , y_hat):
    m =  {
        'MAE'           : mean_absolute_error(y_true, y_hat),
        'MAE Benchmark' : mean_absolute_error(y_true, y_benchmark),
        'MSE'           : mean_squared_error(y_true, y_hat),
        'MASE'          : mean_absolute_error(y_true, y_hat) / mean_absolute_error(y_true, y_benchmark),
        'FVA'           : 1 - (mean_absolute_error(y_true, y_hat) / mean_absolute_error(y_true, y_benchmark))
    }
    return m