# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Modeling

# %%
def model_prediction(ctry,crop,regn,catg,pkge,crcy,msre,mdel):
    import pyodbc
    import pandas as pd
    import datetime
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from pandas import read_csv
    from pandas import datetime
    import pmdarima as pm
    from pmdarima import model_selection
    from datetime import datetime, timedelta
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.arima_model import ARIMAResults
    from datetime import datetime, timedelta

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    qry = f"SELECT * FROM [Prices].[dbo].[prices] where [Country] = '{ctry}' and [Product]='{crop}'"
    df_prices = pd.read_sql(qry, connStr)

    # Dates natural date format
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

    # load model
    ctry_lc = ctry.lower()
    crop_lc = crop.lower()
    mdel_lc = mdel.lower()
    model_name = f'Model/model_{mdel_lc}_{crop_lc}_{ctry_lc}.pkl'
    ld_model = ARIMAResults.load(model_name)

    # make predictions for last year and the following
    yr = str(datetime.now().year+1)
    prediction = ld_model.get_forecast(steps=52) #yr

    # Generate only non-zero prices from last year weeks and current year prediced weeks
    df_pred = prediction.predicted_mean.to_frame(name='Price_estimated')
    last_yr_wk_zero = dfNullID[dfNullID['Date_price'].dt.year==int(yr)-2]['Date_price'].dt.week
    df_pred = df_pred[~df_pred.index.week.isin(last_yr_wk_zero)]
    
    # Mondays and not Sundays as starting day of the week
    df_pred.reset_index(inplace=True)
    df_pred.set_index(df_pred['index'].apply(lambda x: (x - timedelta(days=x.dayofweek))),inplace=True)
    df_pred=df_pred['Price_estimated'].to_frame()

    # Import prediction data to BI
    df_price_model = df_pred.merge(df_prices_non_zero,how='outer',left_on=df_pred.index,right_on=df_prices_non_zero.index).set_index('key_0').reset_index().fillna(0)
    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    # Delete all data with price predicted for the country and crop
    qry_delete = f"DELETE FROM [Prices].[dbo].[prices_prediction] where [Country] = '{ctry}' and [Product]='{crop}'"
    cursor.execute(qry_delete)

    # Load all data with price dates greater than the N last days from today
    upd = 0

    # https://bytes.com/topic/python/answers/166025-python-mysql-insert-null
    for index,row in df_price_model.iterrows():
        if row['Price_estimated'] == 0:
            price_estimated = None 
        else: 
            price_estimated = row['Price_estimated']
        if row['Price'] == 0: 
            price_real = None 
        else: 
            price_real = row['Price']
        cursor.execute("INSERT INTO dbo.prices_prediction([Product],[Country],[Region],[Category],[Package],[Date_price],[Currency],[Measure],[Model],[Price],[Price_estimated],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?)",crop,ctry,regn,catg,pkge,row['key_0'],crcy,msre,mdel,price_real,price_estimated,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        connStr.commit()
        upd += 1

    cursor.close()
    connStr.close()    

    return (print(upd," new prices added"))


# %%



