# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def evaluate_arima_model(X, exog, arima_order, dfNullID):

    ##  Function to evaluate a SARIMAX model for a given order (p,d,q) ##

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

    # Training dataset preparation
    # Pick only campaign weeks for measure the prediction error
    X_clean = pd.DataFrame(X)[~pd.DataFrame(X).index.isin(dfNullID['ID'])].values

    # Endogenous feature
    train_size = int(len(X_clean) * 0.66)
    train, test = X_clean[0:train_size], X_clean[train_size:]
    X_history = [x for x in train]

    # Exogenous features
    exog_clean = pd.DataFrame(exog)[~pd.DataFrame(exog).index.isin(dfNullID['ID'])].values.reshape(-1,1)
    train_exog, test_exog = exog_clean[0:train_size], exog_clean[train_size:]
    exog_history = [y for y in train_exog]

    # make predictions
    predictions = list()
    for t in range(len(test)):
        model_fit = SARIMAX(X_history, exog=exog_history, order=arima_order, initialization='approximate_diffuse').fit() 
        # https://github.com/statsmodels/statsmodels/issues/5459#issuecomment-480562703
        yhat = model_fit.predict()[0] #forecast
        predictions.append(yhat)
        X_history.append(test[t])
        exog_history.append(test_exog[t])
    # calculate out of sample error
    error = mean_absolute_error(test, predictions) #MAE is the metric selected as price fluctuation could be up or down

    return error


# %%
def evaluate_models(dataset, exog, p_values, d_values, q_values, crop, ctry, dfNullID):

    ##  Function to evaluate several combinations of p, d and q values in SARIMAX model and select the order with the least MAE ##

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
					mae = evaluate_arima_model(dataset, exog, order, dfNullID)
					if mae < best_score:
						best_score, best_cfg = mae, order
					print('ARIMA%s MAE=%.3f' % (order,mae))
				except:
					continue
	print('Best ARIMA%s MAE=%.3f' % (best_cfg, best_score))
	versions_file = '../../data/04_models/Model_versions.txt'
	model_data = 'Best ARIMA%s // MAE=%.3f // ' % (best_cfg, best_score)
	updated = datetime.now().strftime("%Y%m%d_%H%M%S")
	with open(versions_file, "a") as f:
		f.write("##"+crop+" "+ctry+" // "+model_data+"Updated "+updated+"##\n")
	
	return(best_cfg)


# %%
def train_sarimax_model(crop,ctry,trade_ctry):

    ##  Function to train a SARIMAX model and save it as a pickle .pkl file ## 

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
    from sklearn.model_selection import train_test_split
    from dateutil.relativedelta import relativedelta
    import extractions as extract
    import transformations as transf

    crop_lc = crop.lower()
    ctry_lc = ctry.lower()

    endog = extract.get_prices_interpolated(crop,ctry)  # this dataframe contains all prices interpolated weekly and mean
    dfNullID = extract.get_null_prices(crop,ctry)   # this dataframe contains the null indexes with their original index id

    # Obtaining exogenous features for SARIMAX model

    df_volumes = extract.get_volumes(crop,trade_ctry,ctry)
    df_salaries = extract.get_labour()

    # Evaluate parameters
    #p_values = range(0, 10)
    #d_values = range(0, 5)
    #q_values = range(0, 5)
    p_values = range(8, 9)
    d_values = range(0, 1)
    q_values = range(1, 2)
    warnings.filterwarnings("ignore")

    # Save exogenous dataframe with same shape as endogenous dataset for getting best model
    exog = endog.join(df_salaries.join(df_volumes).fillna(value=0)).drop('Price',axis=1)

    ### Data is weekly based and the exploratory analysis has shown that there is a clear seasonality between campaigns (years)
    ### So let's set up seasonal_order parameter to see if we improve the estimation and for train data (all observations except the last year)
    endog_train, endog_test = train_test_split(endog, shuffle=False, test_size=len(endog[endog.index.year==max(endog.index.year)]))
    exog_train, exog_test = train_test_split(exog, shuffle=False, test_size=len(exog[exog.index.year==max(exog.index.year)]))

    # Normalization of prices (target variable)
    endog_train_norm_inst, endog_train_norm = transf.normalize(endog_train)

    endog_train_norm = pd.DataFrame(endog_train_norm)
    endog_train_norm.index = endog_train.index
    endog_train_norm.columns = endog_train.columns.values

    # Normalization of exogenous variables
    exog_train_norm_inst, exog_train_norm = transf.normalize(exog_train)

    exog_train_norm = pd.DataFrame(exog_train_norm)
    exog_train_norm.index = exog_train.index
    exog_train_norm.columns = exog_train.columns.values

    # Get the best SARIMAX model
    best_model = evaluate_models(endog_train_norm, exog_train_norm, p_values, d_values, q_values, crop, ctry, dfNullID)

    # Generate model!
    model = SARIMAX(endog_train_norm, exog = exog_train_norm, order = best_model, seasonal_order=(1, 1, 1, 52)).fit()

    # Monkey patch around bug in ARIMA class
    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    # Save model as a pickle, .pkl file
    model.save(f'../../data/04_models/model_sarimax_{crop_lc}_{ctry_lc}.pkl')

    # Save model summary as an independent file		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'../../data/04_models/Summary_{crop_lc}_{ctry_lc}_{updated}.png'
    plt.savefig(dir_img)


