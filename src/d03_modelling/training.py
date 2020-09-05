# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def evaluate_arima_model(X, arima_order,dfNullID):

    ##  Function to evaluate a ARIMA model for a given order (p,d,q) ##

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


# %%
def evaluate_models(dataset, p_values, d_values, q_values, crop, ctry, dfNullID):

    ##  Function to evaluate several combinations of p, d and q values in ARIMA model and select the order with the least MAE ##

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
	versions_file = '../../data/04_models/Model_versions.txt'
	model_data = 'Best ARIMA%s // MAE=%.3f // ' % (best_cfg, best_score)
	updated = datetime.now().strftime("%Y%m%d_%H%M%S")
	with open(versions_file, "a") as f:
		f.write("##"+crop+" "+ctry+" // "+model_data+"Updated "+updated+"##\n")
	
	return(best_cfg)


# %%
def train_arima_model(crop,ctry,trade_ctry,ctgr):

    ##  Function to train a ARIMA model and save it as a pickle .pkl file ## 

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
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

    crop_lc = crop.lower()
    ctry_lc = ctry.lower()
    tctr_lc = trade_ctry.lower()
    ctgr_lc = ctgr.lower()

    # Get prices interpolated
    df_prices = extract.get_prices_interpolated(crop,ctry,trade_ctry,ctgr)

    # Save null indexes with their original index id
    dfNullID = extract.get_null_prices(crop,ctry,trade_ctry,ctgr)

    ### Our data is weekly based and the exploratory analysis has shown us that there is a clear seasonality. 
    ### So let's set up seasonal_order parameter to see if we improve the estimation and for train data (all observations except the last year)
    df_prices_train, df_prices_test =         train_test_split(df_prices, shuffle=False, test_size=len(df_prices[df_prices.index.year==max(df_prices.index.year)]))

    
    # Evaluate parameters
    p_values = range(0, 10)
    d_values = range(0, 5)
    q_values = range(0, 5)

    warnings.filterwarnings("ignore")

    # Get the best ARIMA model
    best_model = evaluate_models(df_prices_train.values, p_values, d_values, q_values, crop, ctry, dfNullID)

    # Generate model!
    model = SARIMAX(df_prices_train, order = best_model, seasonal_order=(1, 1, 1, 52)).fit()

    # Monkey patch around bug in ARIMA class
    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    # Save model as a pickle, .pkl file
    model.save(f'../../data/04_models/model_arima_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}.pkl')

    # Save model summary as an independent file		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'../../data/04_models/Summary_ARIMA_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}_{updated}.png'
    plt.savefig(dir_img)


# %%
def evaluate_sarimax_model(X, exog, arima_order, dfNullID):

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
def evaluate_xmodels(dataset, exog, p_values, d_values, q_values, crop, ctry, dfNullID):

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
					mae = evaluate_sarimax_model(dataset, exog, order, dfNullID)
					if mae < best_score:
						best_score, best_cfg = mae, order
					print('ARIMA%s MAE=%.3f' % (order,mae))
				except:
					continue
	print('Best SARIMAX%s MAE=%.3f' % (best_cfg, best_score))
	versions_file = '../../data/04_models/Model_versions.txt'
	model_data = 'Best SARIMAX%s // MAE=%.3f // ' % (best_cfg, best_score)
	updated = datetime.now().strftime("%Y%m%d_%H%M%S")
	with open(versions_file, "a") as f:
		f.write("##"+crop+" "+ctry+" // "+model_data+"Updated "+updated+"##\n")
	
	return(best_cfg)


# %%
def train_sarimax_model(crop,ctry,trade_ctry,ctgr,exog):

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
    tctr_lc = trade_ctry.lower()
    ctgr_lc = ctgr.lower()

    endog = extract.get_prices_interpolated(crop,ctry,trade_ctry,ctgr)  # this dataframe contains all prices interpolated weekly and mean
    dfNullID = extract.get_null_prices(crop,ctry,trade_ctry,ctgr)   # this dataframe contains the null indexes with their original index id

    # Save exogenous dataframe with same shape as endogenous dataset for getting best model
    exog = endog.join(exog.fillna(value=0)).fillna(value=0).drop('Price',axis=1)

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
    
    # Evaluate parameters
    p_values = range(0, 10)
    d_values = range(0, 5)
    q_values = range(0, 5)

    warnings.filterwarnings("ignore")

    # Get the best SARIMAX model
    best_model = evaluate_xmodels(endog_train_norm, exog_train_norm, p_values, d_values, q_values, crop, ctry, dfNullID)

    # Generate model!
    model = SARIMAX(endog_train_norm, exog = exog_train_norm, order = best_model, seasonal_order=(1, 1, 1, 52)).fit()

    # Monkey patch around bug in ARIMA class
    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    # Save model as a pickle, .pkl file
    model.save(f'../../data/04_models/model_sarimax_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}.pkl')

    # Save model summary as an independent file		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'../../data/04_models/Summary_SARIMAX_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}_{updated}.png'
    plt.savefig(dir_img)


# %%
def train_arima_model_vols(crop,ctry,trade_ctry,ctgr):

    ##  Function to train a ARIMA model and save it as a pickle .pkl file ## 

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
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

    crop_lc = crop.lower()
    ctry_lc = ctry.lower()
    tctr_lc = trade_ctry.lower()
    ctgr_lc = ctgr.lower()

    # Get prices interpolated
    df_vols = extract.get_volumes(crop,ctry,trade_ctry)

    # Save null indexes with their original index id
    dfNullID = extract.get_null_prices(crop,ctry,trade_ctry,ctgr)

    ### Our data is weekly based and the exploratory analysis has shown us that there is a clear seasonality. 
    ### So let's set up seasonal_order parameter to see if we improve the estimation and for train data (all observations except the last year)
    df_vols_train, df_vols_test =         train_test_split(df_vols, shuffle=False, test_size=len(df_vols[df_vols.index.year==max(df_vols.index.year)]))

    # Evaluate parameters
    p_values = range(0, 10)
    d_values = range(0, 5)
    q_values = range(0, 5)

    warnings.filterwarnings("ignore")

    # Get the best ARIMA model
    best_model = evaluate_models(df_vols_train.values, p_values, d_values, q_values, crop, ctry, dfNullID)

    # Generate model!
    model = SARIMAX(df_vols_train, order = best_model, seasonal_order=(1, 1, 1, 52)).fit()

    # Monkey patch around bug in ARIMA class
    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    # Save model as a pickle, .pkl file
    model.save(f'../../data/04_models/model_arima_vols_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}.pkl')

    # Save model summary as an independent file		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'../../data/04_models/Summary_ARIMA_vols_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}_{updated}.png'
    plt.savefig(dir_img)


# %%



