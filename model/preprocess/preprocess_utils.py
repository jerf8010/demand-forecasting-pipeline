import pandas    as pd
import numpy     as np  
from   datetime  import datetime, timedelta
from   time      import time
from   typing    import Sequence, Tuple

from auxiliar_func.auxiliars import preprocess_text

def columns2unique_id(
    df      : pd.DataFrame,
    id_cols : Sequence[str]) -> pd.DataFrame:
    
    """ Concatenates 'id_cols' into an unique_id column

    Parameters
    ----------
    df: pandas.DataFrame: 
        Data 
    id_cols: Sequence[str]: 
        Name of columns to be used for unique id
    
    Returns
    -------
    pandas.DataFrame:
        Data without the id_cols columns, instead unique_id colum

    How to use
    ----------
    id_cols  =  ["Branch", "Customer", "Product"];

    df       =  columns2unique_id(df = df, id_cols = id_cols)

    """

    df              = df.copy()
    df[id_cols]     = df[id_cols].astype(str)
    df['unique_id'] = ['uid_' + '_'.join(row) for row in df[id_cols].values.tolist()]
    df              = df.drop(id_cols, axis=1)
    return df


def get_fiscal_calendar(year=2022):
    """ Generate fiscal calendar for a given year

    Parameters
    ----------
    year: int (YYYY)
        Year for which the calendar is required

    Returns
    -------
    pandas.DataFrame

    How to use
    ----------
    calendar = get_fiscal_calendar(year = YYYY)
    """

     # The first day must be the first Thursday after the 52nd week of the past year
    first_date     = pd.to_datetime(f'{year}-01-01')
    first_date_day = ((first_date.dayofweek + 4) % 7) + 1

    if first_date_day == 1:
        n = 0
        pass
    
    elif first_date_day in [2, 3, 4]:
        n = first_date_day-1
        first_date = first_date - timedelta(n)
        
    elif first_date_day in [5,6,7]:
        n = 8 - first_date_day
        first_date = first_date + timedelta(n)
    
    # The last day must be 52 (364 days) weeks after the first day
    last_date = first_date + timedelta(52 * 7 - 1)

    date = pd.date_range(first_date, last_date) 
    ## .dayofweek returns the days of the week (Monday = 0, Tuesday = 1,..., Sunday = 6)
    ## since Thursday is 4 then we shift the days 4 units mod 7 to obtain Thursday = 0 and Wednesday = 6 and then we sum one 
    #dayofweek_dict = {1 : Thursday, 2 : Friday, 3 : Saturday, 4 : Sunday, 5 : Monday, 6 : Tuesday, 7 : Wednesday }
    day   = ((date.dayofweek + 4) % 7) + 1
    ## Since we have 52 fiscal weeks each year in 364 days, using the integer part of the 
    # indicator of the day of the year you are in between 7 gives us the week we are in
    week  = (np.arange(0,len(date)) // 7) + 1 
    calendar =  pd.DataFrame({'Date': date.values, 'week_num': week, 'day': day})
    calendar['year'] = year
    return calendar

def data_process(
    df         : pd.DataFrame, 
    calendar   : pd.DataFrame,
    itinerary  : pd.DataFrame,
    params     : dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """ Take the extracted data and do the following:
        - Adjust sales and itinerary data format 
        - Filter the data as required
        - Inner join Data, Itinerary and Calendar
        - Remove non positive sales values.
        - Group the time series to the required format.
        - Group to the desired frequency (W-THU)
        - Filter series with sales in the last 2 months
        - Remove recently added time series. (status: r0)
        - Remove series that ceased to exist (status: r1)
        - Set series assigned to Benchmark   (status: benchmark)
        - Set series assigned to PS Model    (status: model)
        

    Parameters
    ----------
    df        : pandas.DataFrame
        Historical sales data
    itinerary : pandas.DataFrame
        Visit itinerary
    params    : dict
        parameter dictionary that must contain:
            'year'           : int
                target week year, 
            'week'           : int
                target week
            'branch_name' : Sequence[str]
                list of agency names to filter, or "all" to take them all
            'channel_name'   : str
                channel name to filter, only one channel at a time
            'products'      : Sequence[str]
                list of product ids to filter, or "all" to take them all
            'customers'       : Sequence[str]
                list of client ids to filter, or "all" to take them all 
            'id_cols'        : Sequence[str]
                list of columns that will uniquely identify a series. Ex: ["Branch", "Customer", "Product"]
            'ds_col'         : str
                column name that represents the temporal index of the time series. Ex: "Date"
            'y_col'          : str
                column name that represents the target value for the time series.  Ex: "Units_Sold"
            'freq'           : str
                Frequency of the jump in time at each point of the time series. Ex: "W-THU" 
            'recently_added_threshold'  : int
                Number of weeks threshold that determines if a series is recently added one. 
                A recently added series is a series whose first historical record is after the target week 
                minus recently_added_threshold
            'ceased_to_exist_threshold' : int
                Number of weeks threshold that determines if a series ceased to exist.
                We say that a series ceased to exist if the date of its last record in the history is 
                before the target week minus ceased_to_exist_threshold
            'benchmark_threshold'       : int 
                Minimum number of records needed to train the model.
                Since some models need a number of records to be able to train, the series that do not
                have the necessary number of records will be sent directly to the benchmark.
            'benchmark':   int
                Lenght of data for validation

            
    Returns
    --------
    pivot_df    : pandas.DataFrame
        Processed data
    train_pivot : pandas.DataFrame
        Processed data for training
    test_pivot  : pandas.DataFrame
        Processed data for validation

    How to Use
    ----------
    
    pivot_series, pivot_train, pivot_vald = data_process(
        df        = df,
        itinerary = itinerary,
        params    = preprocess_args["parameters"]
        )
    """
    
    ## getting parameters from dictionary
    print(f'{preprocess_text()} Getting Parameters... ', sep=' ', end='', flush=True)
    start                    = time()
    YYYY    	             = params['year']
    ww                       = params['week']
    channel_name             = params['channel_name']
    branch_name           = params['branch_name']
    routes                    = params['route']
    products                = params['products']
    customers                 = params['customers']
    id_cols                  = params['id_cols']
    ds_col                   = params['ds_col']
    y_col                    = params['y_col']
    freq                     = params['freq']
    recently_added_threshold = params['recently_added_threshold']
    benchmark_threshold      = params['benchmark_threshold']
    ceased_2_exist_threshold = params['ceased_to_exist_threshold']
    benchmark                = params['benchmark']
    
    if ww == 1:
        validation_YYYY = YYYY - 1
        validation_ww   = 52
        last_YYYY       = YYYY - 1
        last_ww         = 51
    elif ww == 2:
        validation_YYYY = YYYY
        validation_ww   = 1
        last_YYYY       = YYYY - 1
        last_ww         = 52
    elif ww > 2:
        validation_YYYY = YYYY
        validation_ww   = ww - 1
        last_YYYY       = YYYY
        last_ww         = ww - 2
    print('done \n')
    

    ## Adjust Sales data format
    print(f'{preprocess_text()} Getting Sales... ',  sep=' ', end='', flush=True)
    for col in id_cols:
        df[col] = df[col].apply(str)
    for col in ['Channel_Name', 'Branch_Name']:
        df[col] = df[col].apply(lambda x: x.lower())
    print('done \n')

    ## Filter the data as required
    print(f'{preprocess_text()} Applying Filters... ')

    # Channel name most be unique, one channel at a time
    channel_name   = channel_name.lower()
    df = df[df.Channel_Name == channel_name]
    print(f'\t - Selected channel: {channel_name}')

    # filtering for agenciy in list of agency names, or "all" to take them all
    if branch_name == 'all':
        print(f'\t - Selected Agencies: {df.Branch_Name.nunique()} ({branch_name})')
    else:
        print(f'\t - Selected Agencies: {len(branch_name)} ({100 * len(branch_name) / df.Branch_Name.nunique():.2f} %)')
        temp_list = []
        for x in branch_name:
            temp_list.append(x.lower())
            print('\t\t > ' + x)

        branch_name = temp_list
        df = df[df.Branch_Name.isin(branch_name)]

    # filtering for routes in list of routes ids, or "all" to take them all
    if routes == 'all':
        print(f'\t - Selected Routes: {df.Route.nunique()} ({routes})')
    else:
        print(f'\t - Selected Routes: {len(routes)} ({100 * len(routes) / df.Route.nunique():.2f} %)')
        df = df[df.Route.isin(routes)]

    # filtering for client in list of client ids, or "all" to take them all
    if customers == 'all':
        print(f'\t - Selected Clients: {df.Customer.nunique()} ({customers})')
    else:
        print(f'\t - Selected Clients: {len(customers)} ({100 * len(customers) / df.Customer.nunique():.2f} %)')
        df = df[df.Customer.isin(customers)]
    
    # filtering for product in list of product ids, or "all" to take them all
    if products == 'all':
        print(f'\t - Selected Products: {df.Product.nunique()} ({products})')
    else:
        print(f'\t - Selected Products: {len(products)} ({100 * len(products) / df.Product.nunique():.2f} %)')
        df = df[df.Product.isin(products)] 
        for x in df.Product.unique():
            temp = df[df.Product == x].Product_Nombre.unique()
            print('\t\t > ' + temp[0])
    print('done \n')
    
    ## getting fiscal calendar
    print(f'{preprocess_text()} Getting Fiscal Calendar... ',  sep=' ', end='', flush=True)
    #calendar  = get_fiscal_calendar(year = YYYY)
    #calendarB = get_fiscal_calendar(year = YYYY - 1)
    #calendar = pd.concat((calendarB, calendar), axis=0)
    calendar = calendar.copy()
    print('  done \n')

    # Auxiliar Function, jost for show info about a week
    def show_week(week, year, mess = 'week'):
        first_day = calendar[(calendar['week_num'] == week) & (calendar['year'] == year)].Date.min()
        last_day  = calendar[(calendar['week_num'] == week) & (calendar['year'] == year)].Date.max()
        return f'{mess}: {week} ({first_day.date()} - {last_day.date()})'

    ## Adjust Itinerary data format
    print(f'{preprocess_text()} Getting Visit Itinerary... ')
    itinerary.rename(columns={'branch_code' : 'Branch' , 'customer_code' : 'Customer', 'route_code' : 'Route',
                         'dateVisitPlan' : ds_col}, inplace=True)
    itinerary = pd.merge(itinerary, calendar, how = 'inner', on = ds_col)
    for col in id_cols:
        if col in itinerary.columns:
            itinerary[col] = itinerary[col].apply(str)
        else:
            pass
    itinerary = itinerary[itinerary.columns[itinerary.columns.isin(id_cols)].tolist() + ['year', 'week_num', 'day'] + [ds_col]]
    print('  done \n')

    ## Matching Data with Itinerary
    print(f'{preprocess_text()} Matching Data with Itinerary...')
    # last week in historical
    mes = 'Last Week        '
    print(f'\t - {show_week(last_ww, last_YYYY, mes)}')

    # validation week
    mes = 'Validation Week  '
    print(f'\t - {show_week(validation_ww, validation_YYYY, mes)}')

    # target week
    mes = 'Target Week      '
    print(f'\t - {show_week(ww, YYYY, mes)}')

    # first and last day in validation week
    first_val_day = calendar[(calendar['week_num'] == validation_ww) & (calendar['year'] == validation_YYYY)].Date.min()
    last_val_day  = calendar[(calendar['week_num'] == validation_ww) & (calendar['year'] == validation_YYYY)].Date.max()

    # last day in training set 
    last_day  = calendar[(calendar['week_num'] == last_ww) & (calendar['year'] == last_YYYY)].Date.max()

    # itinerary in target week 
    itinerary1         = itinerary[(itinerary['week_num'] == ww) & (itinerary['year'] == YYYY)]
    itinerary1[ds_col] = pd.to_datetime(itinerary1[ds_col])

    # client on itinerary in target week
    customers_week    = itinerary1.Customer.unique().tolist()

    # client on historical Sales data
    N_customers = df.Customer.nunique()
    if N_customers == 0: 
        return None

    # customer on intersection: itineray in target week and historical Sales data
    n_customers = df[df.Customer.isin(customers_week)].Customer.nunique()
    if n_customers == 0:
        return None
    
    print(f'\t - No. Customers Sales {N_customers:,d}')
    print(f'\t - No. Customers intersection(Sales, Itinerary)  {n_customers:,d} ({100*n_customers/N_customers:.2f} %)')
    df1         = df[df.Customer.isin(customers_week)]
    df1[ds_col] = pd.to_datetime(df1[ds_col])
    print(f'\t - No. Customers Droped {N_customers - n_customers:,d} ({100*(N_customers - n_customers)/N_customers:.2f} %)')
    df1.Date = pd.to_datetime(df1.Date)
    df1 = df1[df1.Date <= last_val_day]
    print('done \n')

    # Getting dataframe for Returns analysis
    dfReturns = df1[df1[y_col] != 0].copy()

    ## Remove non positive sales values
    print(f'{preprocess_text()} Removing non positive values... ')
    print(f'\t - Negative values removed: {sum(df1[y_col] < 0):,d} ({100*sum(df1[y_col] < 0)/len(df1):.2f} %)')
    print(f'\t - Zero values removed:     {sum(df1[y_col] == 0):,d} ({100*sum(df1[y_col] == 0)/len(df1):.2f} %)')
    df1 = df1[df1[y_col] > 0]
    print('done \n')

    ## Creating a unique Id per series by combining values of id_cols
    print(f'{preprocess_text()} Creating a unique Id per series... ', sep=' ', end='', flush=True)
    dfReturns = columns2unique_id(dfReturns, id_cols)
    
    df1 = columns2unique_id(df1, id_cols)
    print('done \n')

    ## Grouping using unique_id and frequency
    print(f'{preprocess_text()} Grouping using unique_id and frequency... ')
    print(f'\t - Settled frequency: {freq}')
    
    #########################################################
    dfReturns  = dfReturns.groupby(
        by = ['unique_id'] + [pd.Grouper(key = ds_col, freq = 'D', closed = 'left', label = 'left')]
        )[y_col].sum().reset_index()

    df1  = df1.groupby(
        by = ['unique_id'] + [pd.Grouper(key = ds_col, freq = freq, closed = 'left', label = 'left')]
        )[y_col].sum().reset_index()
    print('done \n')
    df1
    
    n_series = df1['unique_id'].nunique()
    print(f'{preprocess_text()} Number of loaded series: {n_series:,d} \n')

    ## Remove recently added series
    
    # sales data with info about fiscal week from calendar
    df_copy = pd.merge(df1, calendar, on='Date', how = 'left')

    # getting date of the first record by series
    info0 = df_copy.groupby(
        by = 'unique_id', as_index=False).agg(
            s_year = ('year', 'min'),
            Date = ('Date', 'min'))
    info0 = pd.merge(info0, calendar[['week_num', 'Date']], on='Date', how= 'left')
    info0.rename(columns={'Date' : 's_date', 'week_num': 's_week'}, inplace=True)

    # getting date of last record by series
    info1 = df_copy.groupby(
        by = 'unique_id', as_index=False).agg(
            e_year = ('year', 'max'),
            Date = ('Date', 'max'))
    info1 = pd.merge(info1, calendar[['week_num', 'Date']], on='Date', how= 'left')
    info1.rename(columns={'Date' : 'e_date', 'week_num': 'e_week'}, inplace=True)

    # merging info about first record and  last record  per series
    info = pd.merge(info0, info1, on='unique_id', how = 'inner')

    ## Remove recently added time series. (status: r0)
    print(f'{preprocess_text()} Removing recently added series... ')

    if validation_ww - recently_added_threshold + 1 <= 0:
        threshold_ww0   = 52 + validation_ww-recently_added_threshold + 1
        threshold_year0 = YYYY - 1    
    else:
        threshold_ww0   = validation_ww - recently_added_threshold + 1
        threshold_year0 = YYYY
    
    mes = 'Maximum allowable start date'
    print(f'\t - {show_week(threshold_ww0, threshold_year0, mes)}')
    # first date in threshold week
    threshold_date0 = calendar[(calendar['week_num'] == threshold_ww0) & (calendar['year'] == threshold_year0)].Date.min()
    non_valid_id0 = info[info.s_date >= threshold_date0].unique_id
    n_removed0 = len(non_valid_id0)
    print(f'\t - Series that will be Removed: {n_removed0:,d} ({100*(n_removed0 / n_series):.2f} %)')
    print('done \n')

    ## Remove series that ceased to exist
    print(f'{preprocess_text()} Removing series that ceased to exist... ')
    if validation_ww - ceased_2_exist_threshold <= 0:
        threshold_ww1   = 52 + validation_ww - ceased_2_exist_threshold 
        threshold_year1 = YYYY-1
    else:
        threshold_ww1   = validation_ww - ceased_2_exist_threshold
        threshold_year1 = YYYY
    mes = 'Minimum allowable end date'
    print(f'\t - {show_week(threshold_ww1, threshold_year1, mes)}')
    threshold_date1 = calendar[(calendar['week_num'] == threshold_ww1) & (calendar['year'] == threshold_year1)].Date.max()
    non_valid_id1 = info[info.e_date <= threshold_date1].unique_id
    n_removed1 = len(non_valid_id1)

    print(f'\t - Series that will be Removed: {n_removed1:,d} ({100*(n_removed1 / n_series):.2f} %)')
    print('done \n')
    
    # Valid Series Index
    valid_index = info[(~info.unique_id.isin(non_valid_id0)) & (~info.unique_id.isin(non_valid_id1)) ].index
    n_valid     = len(valid_index)
    info = info[info.index.isin(valid_index)]
    
    ## Set series assigned to Benchmark and model
    print(f'{preprocess_text()} Setting series to benchmark... ')
    if validation_ww - benchmark_threshold <= 0:
        threshold_ww2   = 52 + validation_ww - benchmark_threshold 
        threshold_year2 = YYYY-1
    else:
        threshold_ww2   = validation_ww - benchmark_threshold
        threshold_year2 = YYYY
    
    mes = 'Minimum allowable Start Date'
    print(f'\t - {show_week(threshold_ww2, threshold_year2, mes)}')
    threshold_date2 = calendar[(calendar['week_num'] == threshold_ww2) & (calendar['year'] == threshold_year2)].Date.max()
    id2benchmark = info[info.s_date >= threshold_date2].unique_id
    id2model     = info[info.s_date < threshold_date2].unique_id
    n_benchmark  = len(id2benchmark)
    n_model      = len(id2model)
    print(f'\t - Series assigned to Benchmark: {n_benchmark:,d} ({100*(n_benchmark / n_series):.2f} %)')
    print(f'\t - Series assigned to Model: {n_model:,d} ({100*(n_model / n_series):.2f} %)')
    print('done \n')

    ## Pivoting
    print(f'{preprocess_text()} Pivotting Table... ', sep='', end=' ', flush=True)
    pivot_df_returns = dfReturns.reset_index()
    pivot_df_returns = pivot_df_returns.pivot_table(index=['unique_id'], columns=[ds_col], values=[y_col], fill_value=0)
    pivot_df_returns = pivot_df_returns.droplevel(0, axis=1)

    pivot_df = df1.reset_index()
    pivot_df = pivot_df.pivot_table(index=['unique_id'], columns=[ds_col], values=[y_col], fill_value=0)
    pivot_df = pivot_df.droplevel(0, axis=1)
    
    ## Verifying data in the last 8 weeks
    to_delete = [idx for idx, row in pivot_df.iterrows() if sum(row[len(row) - 8:]) == 0 ]
    pivot_df_returns = pivot_df_returns[~pivot_df_returns.index.isin(to_delete)]
    pivot_df = pivot_df[~pivot_df.index.isin(to_delete)]

    ## Assig Series
    #temp = np.array(pivot_df.index.values)
    #temp[pivot_df.index.isin(non_valid_id0)] = 'r0'
    #temp[pivot_df.index.isin(non_valid_id1)] = 'r1'
    #temp[pivot_df.index.isin(id2benchmark)]  = 'benchmark'
    #temp[pivot_df.index.isin(id2model)]      = 'model'
    #index = pd.MultiIndex.from_arrays([pivot_df.index.values.tolist(), temp.tolist()], names=['unique_id', 'flow_id'])
    pivot_df = pivot_df[pivot_df.index.isin(id2model)]
    pivot_df_returns = pivot_df_returns[pivot_df_returns.index.isin(id2model)]

    pivot_df.columns        = pivot_df.columns.rename(None)
    pivot_df_returns.columns  = pivot_df_returns.columns.rename(None)

    #train_pivot = pivot_df.iloc[:, pivot_df.columns <= last_day]
    #test_pivot  = pivot_df.iloc[:, (pivot_df.columns > first_val_day) & (pivot_df.columns <= last_val_day) ]

    # ELiminating series with cero sales (equal sales and returns: when they are added they are eliminated)
    to_delete      = [series for series in pivot_df_returns.index if sum(pivot_df_returns.loc[series, :]) == 0]
    pivot_df       = pivot_df[~pivot_df.index.isin(to_delete)]
    pivot_df_returns = pivot_df_returns[~pivot_df_returns.index.isin(to_delete)]
    
    #pivot_df    = pivot_df.reset_index()
    #train_pivot = train_pivot.reset_index()
    #test_pivot  = test_pivot.reset_index()

    pivot_df.columns        = [str(date) for date in pivot_df.columns]
    pivot_df_returns.columns  = [str(date) for date in pivot_df_returns.columns]
    
    train_pivot    = pivot_df.iloc[:, :pivot_df.shape[1] - benchmark]
    test_pivot     = pivot_df.iloc[:, pivot_df.shape[1] - benchmark:]
    
    print('done \n')
    print(f'{preprocess_text()} Runtime: {time() - start :.2f}')
    return pivot_df, pivot_df_returns, train_pivot, test_pivot


def data_test_process(
    df         : pd.DataFrame, 
    params     : dict,
    ) -> pd.DataFrame:

    ## getting parameters from dictionary
    print(f'{preprocess_text()} Getting Parameters... ', sep=' ', end='', flush=True)
    channel_name             = params['channel_name']
    branch_name           = params['branch_name']
    routes                    = params['route']
    products                = params['products']
    customers                 = params['customers']
    id_cols                  = params['id_cols']
    ds_col                   = params['ds_col']
    y_col                    = params['y_col']
    freq                     = params['freq']
    print('done \n')
    

    ## Adjust Sales data format
    print(f'{preprocess_text()} Getting Sales... ',  sep=' ', end='', flush=True)
    for col in id_cols:
        df[col] = df[col].apply(str)
    for col in ['Channel_Name', 'Branch_Name']:
        df[col] = df[col].apply(lambda x: x.lower())
    print('done \n')

    ## Filter the data as required
    print(f'{preprocess_text()} Applying Filters... ')

    # Channel name most be unique, one channel at a time
    channel_name   = channel_name.lower()
    df = df[df.Channel_Name == channel_name]
    print(f'\t - Selected channel: {channel_name}')

    # filtering for agenciy in list of agency names, or "all" to take them all
    if branch_name == 'all':
        print(f'\t - Selected Agencies: {df.Branch_Name.nunique()} ({branch_name})')
    else:
        print(f'\t - Selected Agencies: {len(branch_name)} ({100*len(branch_name)/df.Branch_Name.nunique():.2f} %)')
        temp_list = []
        for x in branch_name:
            temp_list.append(x.lower())
            print('\t\t > ' + x)

        branch_name = temp_list
        df = df[df.Branch_Name.isin(branch_name)]

    # filtering for routes in list of routes ids, or "all" to take them all
    if routes == 'all':
        print(f'\t - Selected Routes: {df.Route.nunique()} ({routes})')
    else:
        print(f'\t - Selected Routes: {len(routes)} ({100*len(routes)/df.Route.nunique():.2f} %)')
        df = df[df.Route.isin(routes)]

    # filtering for client in list of client ids, or "all" to take them all
    if customers == 'all':
        print(f'\t - Selected Clients: {df.Customer.nunique()} ({customers})')
    else:
        print(f'\t - Selected Clients: {len(customers)} ({100*len(customers)/df.Customer.nunique():.2f} %)')
        df = df[df.Customer.isin(customers)]
    
    # filtering for product in list of product ids, or "all" to take them all
    if products == 'all':
        print(f'\t - Selected Products: {df.Product.nunique()} ({products})')
    else:
        print(f'\t - Selected Products: {len(products)} ({100*len(products)/df.Product.nunique():.2f} %)')
        df = df[df.Product.isin(products)] 
        for x in df.Product.unique():
            temp = df[df.Product == x].Product_Nombre.unique()
            print('\t\t > ' + temp[0])
    print('done \n')

    df1 = df.copy()
    df1['Date'] = pd.to_datetime(df1['Date'])

    dfReturns = df1.copy()

    ## Creating a unique Id per series by combining values of id_cols
    print(f'{preprocess_text()} Creating a unique Id per series... ', sep=' ', end='', flush=True)
    df1 = columns2unique_id(df1, id_cols)
    dfReturns = columns2unique_id(dfReturns, id_cols)

    print('done \n')
    ## Grouping using unique_id and frequency
    print(f'{preprocess_text()} Grouping using unique_id and frequency... ')
    print(f'\t - Settled frequency: {freq}')
    dfReturns  = dfReturns.groupby(
        by = ['unique_id'] + [pd.Grouper(key = ds_col, freq = 'D', closed = 'left', label = 'left')]
        )[y_col].sum().reset_index()


    df1  = df1.groupby(
        by = ['unique_id'] + [pd.Grouper(key = ds_col, freq = freq, closed = 'left', label = 'left')]
        )[y_col].sum().reset_index()
    print('done \n')
    return df1, dfReturns
