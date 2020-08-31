# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Modeling

# %%
def get_prediction(ctry,crop,regn,catg,pkge,crcy,msre,mdel,start,end):

    ##  Function to call a model and make price predictions   ##

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
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

    df_prices = extract.get_prices_interpolated(crop,ctry)
    dfNullID = extract.get_null_prices(crop,ctry)
    df_prices_non_zero = df_prices[~df_prices.index.isin(dfNullID.set_index('Date_price').index)]

    # load model
    ctry_lc = ctry.lower()
    crop_lc = crop.lower()
    mdel_lc = mdel.lower()
    model_name = f'../../data/04_models/model_{mdel_lc}_{crop_lc}_{ctry_lc}.pkl'
    ld_model = ARIMAResults.load(model_name)

    # make predictions for last year and the following
    yr = str(datetime.now().year+1)
    pred = ld_model.get_prediction(start=start, end=end, dynamic=True)
    
    # Generate only non-zero prices for predicted weeks, based on last-year-no-price weeks 
    df_pred = pred.predicted_mean.to_frame(name='Price_estimated')
    last_yr_wk_zero = dfNullID[dfNullID['Date_price'].dt.year==int(yr)-2]['Date_price'].dt.week
    df_pred = df_pred[~df_pred.index.week.isin(last_yr_wk_zero)]
    
    # Mondays and not Sundays as starting day of the week
    df_pred.reset_index(inplace=True)
    df_pred.set_index(df_pred['index'].apply(lambda x: (x - timedelta(days=x.dayofweek))),inplace=True)
    df_pred = df_pred['Price_estimated'].to_frame()

    # Import prediction data to BI
    df_price_model = df_pred.merge(df_prices_non_zero,how='outer',left_on=df_pred.index,right_on=df_prices_non_zero.index).set_index('key_0').reset_index().fillna(0)
    df_price_model.rename(columns={'key_0' : 'Date_ref'},errors="raise",inplace=True)
    df_price_model.sort_values(by='Date_ref',inplace=True)

    return df_price_model


# %%
def load_predictions_db(df_price_model,ctry,crop,regn,catg,pkge,crcy,msre,mdel):

    ##  Function to upload prices estimated into SQL Server Database   #

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
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
    import stringing as st

    connStr = pyodbc.connect(config.db_con)
    cursor = connStr.cursor()

    #ctry = st.get_comma_values(df_price_model.Country)
    #crop = st.get_comma_values(df_price_model.Product)

    # Delete all prices predicted which are being predicted (greater than the minimum prediction date) for the country and crop
    min_pred_date = min(df_price_model[df_price_model.Price_estimated != 0].Date_ref).date()#.strftime("%Y-%m-%d")
    qry_delete = f"DELETE FROM [Prices].[dbo].[prices_prediction] where [Country] = '{ctry}' and [Product]='{crop}' and [Date_price] >= '{min_pred_date}'"
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
        if row['Date_ref'] >= min_pred_date:
            cursor.execute("INSERT INTO dbo.prices_prediction([Product],[Country],[Region],[Category],[Package],[Date_price],[Currency],[Measure],[Model],[Price],[Price_estimated],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?)",crop,ctry,regn,catg,pkge,row['Date_ref'],crcy,msre,mdel,row['Price'],row['Price_estimated'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1

    cursor.close()
    connStr.close()    

    return (print(upd," new prices added"))


# %%



