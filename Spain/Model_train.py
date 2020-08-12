# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Modeling

# %%
## First model --> https://www.youtube.com/watch?v=WjeGUs6mzXg
# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order,dfNullID):
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_error
	import warnings
	import pandas as pd
	import pyodbc
	from statsmodels.tools.sm_exceptions import ConvergenceWarning
	from pandas import read_csv
	from pandas import datetime
	from statsmodels.tsa.arima_model import ARIMA
	import numpy as np
	import pmdarima as pm
	from pmdarima import model_selection
	import datetime
	from datetime import datetime, timedelta
	import matplotlib.pyplot as plt

	# prepare training dataset
	X_clean = pd.DataFrame(X)[~pd.DataFrame(X).index.isin(dfNullID['ID'])].values # Pick only campaign weeks for measure the prediction error
	train_size = int(len(X_clean) * 0.66)
	train, test = X_clean[0:train_size], X_clean[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model_fit = SARIMAX(history, order=arima_order).fit()
		#model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_absolute_error(test, predictions) #MAE is the metric selected as price fluctuation could be up or down

	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values, crop, ctry, dfNullID):
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_error
	import warnings
	import pandas as pd
	import pyodbc
	from statsmodels.tools.sm_exceptions import ConvergenceWarning
	from pandas import read_csv
	from pandas import datetime
	from statsmodels.tsa.arima_model import ARIMA
	import numpy as np
	import pmdarima as pm
	from pmdarima import model_selection
	import datetime
	from datetime import datetime, timedelta
	import matplotlib.pyplot as plt
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mae = evaluate_arima_model(dataset, order, dfNullID)
					if mae < best_score:
						best_score, best_cfg = mae, order
					print('ARIMA%s MAE=%.3f' % (order,mae))
				except:
					continue
	print('Best ARIMA%s MAE=%.3f' % (best_cfg, best_score))
	versions_file = './Model/Model_versions.txt'
	model_data = 'Best ARIMA%s // MAE=%.3f // ' % (best_cfg, best_score)
	updated = datetime.now().strftime("%Y%m%d_%H%M%S")
	with open(versions_file, "a") as f:
		f.write("##"+crop+" "+ctry+" // "+model_data+"Updated "+updated+"##\n")
	
	return(best_cfg)


# %%
def train_arima_model(crop,ctry):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import warnings
    import pandas as pd
    import pyodbc
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from pandas import read_csv
    from pandas import datetime
    from statsmodels.tsa.arima_model import ARIMA
    import numpy as np
    import pmdarima as pm
    from pmdarima import model_selection
    import datetime
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt

    crop_lc = crop.lower()
    ctry_lc = ctry.lower()

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    qry = f"SELECT * FROM [Prices].[dbo].[prices] where cast([Country] as nvarchar) = cast('{ctry}' as nvarchar) and cast([Product] as nvarchar) = cast('{crop}' as nvarchar)"
    df_prices = pd.read_sql(qry, connStr)

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

    # Evaluate parameters
    p_values = range(0, 10)
    d_values = range(0, 5)
    q_values = range(0, 5)
    warnings.filterwarnings("ignore")
    best_model = evaluate_models(df_prices_all.values, p_values, d_values, q_values, crop, ctry, dfNullID)

    ### Our data is weekly based and the exploratory analysis has shown us that there is a clear seasonality. 
    ### So let's set up seasonal_order parameter to see if we improve the estimation.
    model = SARIMAX(df_prices_all, order = best_model, seasonal_order=(1, 1, 1, 52)).fit()

    # SAVE MODEL
    # monkey patch around bug in ARIMA class
    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    # save model
    model.save(f'Model/model_arima_{crop_lc}_{ctry_lc}.pkl')

    # save model info		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'Model/Summary_{crop_lc}_{ctry_lc}_{updated}.png'
    plt.savefig(dir_img)


# %%



