# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def get_prices(crop,ctry):

    ##  Function to get prices available    ##

    import pandas as pd
    import pyodbc

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    qry = f"SELECT * FROM [Prices].[dbo].[prices] where cast([Country] as nvarchar) = cast('{ctry}' as nvarchar) and cast([Product] as nvarchar) = cast('{crop}' as nvarchar)"
    df_prices = pd.read_sql(qry, connStr)

    return df_prices


# %%
def get_prices_interpolated(crop,ctry):
    
    ##  Function to get all prices interpolated weekly and mean, removing the first campaign available  ##

    df_prices = get_prices(crop,ctry)
    df_prices = df_prices[df_prices.Campaign > min(df_prices.Campaign)][['Date_price', 'Price']]
    df_prices.set_index('Date_price',inplace=True)
    df_prices.sort_index(inplace=True)
    df_prices.index = df_prices.index.astype('datetime64[ns]') 
    df_prices = df_prices.resample('W-MON').mean()
    rows_null = df_prices.isnull()
    idx_null = rows_null[rows_null.any(axis=1)].index
    df_prices_all = df_prices.interpolate()

    return df_prices_all


# %%
def get_null_prices(crop,ctry):
    
    ##  Function to get weeks without prices informed -no campaign period-  ##

    import pandas as pd

    df_prices = get_prices(crop,ctry)
    df_prices = df_prices[df_prices.Campaign > min(df_prices.Campaign)][['Date_price', 'Price']]
    df_prices.set_index('Date_price',inplace=True)
    df_prices.sort_index(inplace=True)
    df_prices.index = df_prices.index.astype('datetime64[ns]') 
    df_prices = df_prices.resample('W-MON').mean()
    rows_null = df_prices.isnull()
    idx_null = rows_null[rows_null.any(axis=1)].index
    df_prices_all = df_prices.interpolate()

    df_prices_non_zero = df_prices_all[~df_prices_all.index.isin(idx_null)]
    listIndex = list(zip(df_prices_all.index, range(0,len(df_prices_all))))     # save all indexes in tuples list (index, idPosition)
    listNull = idx_null     # save all null indexes

    dfIndex = pd.DataFrame(listIndex)
    dfNull = pd.DataFrame(listNull)
    dfIndex.columns = ['Date_price','ID']
    dfNullID = dfIndex.merge(dfNull, how='inner', on='Date_price')    # this dataframe contains the null indexes with their original index id

    return dfNullID


# %%
def get_volumes(crop,ctry,trade_ctry):

    import pyodbc
    import pandas as pd

    ##  Function to get UE volumes import from Spain (Agronometrics), weekly aggregated ##
    
    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    qry = f"SELECT * FROM [Prices].[dbo].[volumes] where cast([Country] as nvarchar) = cast('{ctry}' as nvarchar) and cast([Product] as nvarchar) = cast('{crop}' as nvarchar) and cast([Trade_Country] as nvarchar) = cast('{trade_ctry}' as nvarchar)"

    df_volumes = pd.read_sql(qry, connStr)

    df_volumes = df_volumes[df_volumes.Campaign > min(df_volumes.Campaign)][['Date_volume', 'Volume']]
    df_volumes.groupby('Date_volume').agg('sum')
    df_volumes.set_index('Date_volume',inplace=True)
    df_volumes.sort_index(inplace=True)
    df_volumes.index = df_volumes.index.astype('datetime64[ns]') 
    df_volumes = df_volumes.resample('W-MON').sum()

    return df_volumes


# %%
def get_labour():

    ##  Function to get labor cost index evolution in Spain (https://www.mapa.gob.es)   ##

    import pandas as pd
    from dateutil.relativedelta import relativedelta
    import numpy as np

    # May need to lag 1 year in order to allocate campaign costs to the actual fresh produce sales  
    yr_adj = 1

    # 1985-2017 file
    df = pd.read_excel('../../data/01_raw/labor/indicesysalariosagrariosenero1985-diciembre2017_tcm30-539891.xlsx',sheet_name='IS',header=3,usecols = ['AÑO','Anual'])
    df.AÑO = pd.to_datetime(df.AÑO, format='%Y')
    df.rename(columns={'AÑO': 'year', 'Anual': 'labour_index'}, inplace=True)
    df = df[(df.year.duplicated(keep='first')==False) & df.year.notnull()]
    df.year = df.year.apply(lambda x : x + relativedelta(years=yr_adj))
    df.set_index('year', inplace=True)
    df_salaries = df

    # 2018-2020 file
    df = pd.read_excel('../../data/01_raw/labor/indicesysalariosagrariosenero2018-marzo2020_tcm30-541202.xlsx',sheet_name='IndSal2',header=3,usecols=['AÑO','Enero','Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiem.', 'Octubre', 'Noviem.', 'Diciem.'])
    df.AÑO = pd.to_datetime(df.AÑO, format='%Y')
    df.rename(columns={'AÑO': 'year'}, inplace=True)
    df = df[(df.year.duplicated(keep='first')==False) & df.year.notnull()]
    df.year = df.year.apply(lambda x : x + relativedelta(years=yr_adj))
    df.set_index('year', inplace=True)
    df = df.transpose().replace(0, np.NaN).mean(skipna=True).to_frame()
    df.rename(columns={0: 'labour_index'}, inplace=True)

    df_salaries = df_salaries.append(df)
    df_salaries.index = df_salaries.index.astype('datetime64[ns]') 
    df_salaries = df_salaries.resample('W-MON').mean().fillna(method='ffill')

    return df_salaries


