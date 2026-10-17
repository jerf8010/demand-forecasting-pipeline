import pandas    as     pd
import numpy     as     np
from   datetime  import timedelta, datetime
from   time      import time

def return2original_id(
    df       : pd.DataFrame,
    id_cols   : list,
    drop_uid : bool = False) -> pd.DataFrame:
    
    """Returns the indices to their original values

       Parameters
    ----------
    df: pandas.DataFrame 
        Preprocess data
    params : dict 
        must contain:  
         - id_cols: Column's names of the original values (Sequence[str])
         - y_col: Column's name of interest values (str)
         - ds_col: Column's name of timestamp index (str)
    dorp_uid : bool
        Drop unique id? True/False

    Returns
    -------
    pandas.DataFrame:
        Data with the id_cols columns, instead of unique_id
    """
    
    
    for i, col in enumerate(id_cols):
        df.insert(loc = i+1, column = col, value= df['unique_id'].apply(lambda x: x.split('_')[i+1]))

    if drop_uid:
        df.drop('unique_id', axis = 1, inplace=True)
        return df
    else:
        return df


def split_text():
    """ Logg Split steps
    """
    return f'> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} Splitting:'


def gen_base_vec(n):
    vec = np.zeros(n)
    pos = np.random.randint(low = 0, high = n,)
    vec[pos] = 1
    return vec  

def fixing_sugest(x):
    if x.Remaining == 0:
        return x.Suggested
    else:
        return x.Suggested + (x.Remaining*x.Aux)

def split(
    preds       : pd.DataFrame,
    data        : pd.DataFrame,
    itinerary   : pd.DataFrame,
    calendar  : pd.DataFrame,
    params      : dict):
    
    print(split_text(), ' Getting parameters...', sep='', end='', flush=True)
    ww         = params['week']
    YYYY       = params['year']
    id_cols    = params['id_cols']
    iti_colums_dict = {"branch_code" : "Branch",
                       "route_code": "Route",
                       "customer_code": "Customer"}
    print('done \n')
    
    print(split_text(), ' Preparing Itinerary...', sep='', end='', flush=True)
    # getting  fiscal calendar
    
    # selecting only target week 
    calendar_ = calendar[(calendar["week_num"] == ww) & (calendar['year'] == YYYY)]
    last_day   = calendar_.Date.min() - timedelta(days=1)

    # selecting information of interest in itinerary 
    itinerary = itinerary[['branch_code', 'route_code', 'customer_code', 'dateVisitPlan']]
    # inner merging between calendar and itinerary to filter only target week data  
    itinerary = pd.merge(calendar_, itinerary, left_on='Date', right_on='dateVisitPlan', how = 'inner')
    itinerary = itinerary[['year', 'week_num', 'day', 'Date','branch_code','route_code', 'customer_code']]
    itinerary.rename(columns=iti_colums_dict, inplace=True)
    del calendar_

    print('done \n')
    
    print(split_text(), ' Preparing Predictions and Merging with Itinerary...', sep='', end='', flush=True)
    preds = return2original_id(df=preds,id_cols=id_cols, drop_uid=False)   
    merge_cols = list(set(preds.columns).intersection(set(itinerary.columns)))
    for col in merge_cols:
        preds[col] = preds[col].apply(str)
        itinerary[col] = itinerary[col].apply(str)
    if "Route" in merge_cols:
        itinerary["Route"] =  [('000' + route)[-4:] for route in  itinerary.Route.apply(str)]
    
    preds = pd.merge(preds, itinerary, on = merge_cols, how="inner" )
    del itinerary
    print('done \n')
    
    print(split_text(), ' Preparing Historical and Merging with Predictions...', sep='', end='', flush=True)

    for col in merge_cols:
        data[col] = data[col].apply(str)
        data      = data[data[col].isin(preds[col].unique())]

    data["Date"] = pd.to_datetime(data["Date"])
    data          = data[data.Date <= last_day]

    data          = data[data["Units_Sold"] >= 0]

    #calendar = pd.concat([get_fiscal_calendar(YYYY-1), get_fiscal_calendar(YYYY)], axis=0)
    data       = pd.merge(data, calendar, on='Date', how = "inner")
    
    data_list = []
    groups  = data.groupby(by = id_cols, as_index = False)
    for group in groups.groups:
        aux      = groups.get_group(group)
        totalvol = aux["Units_Sold"].sum()
        data_tem = aux.groupby(by = id_cols + ['day'], as_index = False).agg(
            Frac =  ('Units_Sold',  lambda x: x.sum()/totalvol ))
        data_list.append(data_tem)
        
    data       = pd.concat(data_list, axis=0)
    merge_cols = list(set(preds.columns).intersection(set(data.columns)))
    data       = pd.merge(preds, data, on = merge_cols, how="inner" )
    print('done \n')
    
    print(split_text(), ' Getting predictions on visiting days...', sep='', end='', flush=True)
    data       = pd.merge(data, 
                          data.groupby(by = 'unique_id', as_index=False).agg(NumVisitPerWeek = ('Date', 'count')),
                          on = "unique_id",
                          how = "inner")
    
    data['Suggested'] = np.rint(data["predictions"] * data.Frac)
    
    data       = pd.merge(data, 
                          data.groupby(by = 'unique_id', as_index=False).agg(Remaining = ('Suggested', 'sum')),
                          on = "unique_id",
                          how = "inner")
    data.Remaining = data["predictions"] - data.Remaining
    
    groups = data.groupby(by = "unique_id")
    data['Aux'] = groups['NumVisitPerWeek'].transform(lambda x: gen_base_vec(x.count()).tolist()).values.tolist()  
    data['SuggestedRecal'] = data.apply(fixing_sugest, axis=1)
    
    data       = pd.merge(data, 
                          data.groupby(by = 'unique_id', as_index=False).agg(RemainingRecal = ('SuggestedRecal', 'sum')),
                          on = "unique_id",
                          how = "inner")
    data.RemainingRecal = data["predictions"] - data.RemainingRecal
    print('done \n')
    
    return data