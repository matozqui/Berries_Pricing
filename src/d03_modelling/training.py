# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## SARIMA AND ARIMA

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
	best_score_mae, score_rmse, score_bias, best_cfg = float("inf"), float("inf"), float("inf"), None

	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mae,rmse,bias = evaluate_arima_model(dataset, order, dfNullID)
					if mae < best_score_mae: #mae is the chosen measure for selecting best order
						best_score_mae, score_rmse, score_bias, best_cfg = mae, rmse, bias, order
					print('ARIMA%s MAE=%.3f RMSE=%.3f BIAS=%.3f' % (order,mae,rmse,bias))
				except:
					continue
	print('Best ARIMA%s MAE=%.3f RMSE=%.3f BIAS=%.3f' % (best_cfg, best_score_mae, score_rmse, score_bias))
	versions_file = '../../data/04_models/Model_versions.txt'
	model_data = 'Best ARIMA%s MAE=%.3f RMSE=%.3f BIAS=%.3f' % (best_cfg, best_score_mae, score_rmse, score_bias)
	updated = datetime.now().strftime("%Y%m%d_%H%M%S")
	with open(versions_file, "a") as f:
		f.write("##"+crop+" "+ctry+" // "+model_data+"Updated "+updated+"##\n")
	
	return(best_cfg)


# %%
def evaluate_arima_model(X, arima_order, dfNullID):

    ##  Function to evaluate a ARIMA, SARIMA and SARIMAX models ##

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
    import warnings
    # https://stackoverflow.com/questions/34444607/how-to-ignore-statsmodels-maximum-likelihood-convergence-warning 
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter('ignore', ConvergenceWarning) 

    # prepare training dataset
    X_clean = pd.DataFrame(X)[~pd.DataFrame(X).index.isin(dfNullID['ID'])].values # Pick only campaign weeks for measure the prediction error

    train_size = int(len(X_clean) * 0.66)
    train, test = X_clean[0:train_size], X_clean[train_size:]
    history = [x for x in train]

    # Make predictions
    predictions = list()
    for t in range(len(test)):
        model_fit = SARIMAX(history, order=arima_order).fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mae = mean_absolute_error(test, predictions) #MAE: average of the forecast error values in absolute values
    rmse = np.sqrt(mean_squared_error(test, predictions)) # Root mean square error: average of the squared forecast error values
    bias = np.mean(predictions-test) # Bias measure, suggesting tendency of the model to over forecast (positive error) or under forecast (negative error)
    return mae,rmse,bias


# %%
def train_arima_model(crop,ctry,trade_ctry,ctgr,mdel):

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
    mdel_lc = mdel.lower()

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

    # Get the best SARIMA model
    best_model = evaluate_models(df_prices_train.values, p_values, d_values, q_values, crop, ctry, dfNullID)

    # Generate model!
    if mdel == 'SARIMA':
        model = SARIMAX(df_prices_train, order = best_model, seasonal_order=(1, 1, 1, 52)).fit()
    elif mdel == 'ARIMA':
        model = SARIMAX(df_prices_train, order = best_model).fit()

    # Monkey patch around bug in ARIMA class
    def __getnewargs__(self):
        return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
    ARIMA.__getnewargs__ = __getnewargs__

    # Save model as a pickle, .pkl file
    model.save(f'../../data/04_models/model_{mdel_lc}_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}.pkl')

    # Save model summary as an independent file		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'../../data/04_models/Summary_{mdel}_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}_{updated}.png'
    plt.savefig(dir_img)


# %%
def train_sarima_model_vols(crop,ctry,trade_ctry,ctgr):

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
    model.save(f'../../data/04_models/model_sarima_vols_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}.pkl')

    # Save model summary as an independent file		
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    updated = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_img = f'../../data/04_models/Summary_SARIMA_vols_{crop_lc}_{ctry_lc}_{tctr_lc}_{ctgr_lc}_{updated}.png'
    plt.savefig(dir_img)

# %% [markdown]
# # SARIMAX

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
def calculate_measures(crop_list):

    ##  Function to calculate measures of the different models .pkl saved ##

    import sys
    sys.path.insert(0, '../../src')
    #   https://realpython.com/python-modules-packages/
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import extractions as extract
    import transformations as transf
    import training as train
    import import_data as imp
    import inference as inf
    import pandas as pd
    import numpy as np
    import config
    import time  
    from dateutil.relativedelta import relativedelta
    from datetime import date
    import config as conf
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.arima_model import ARIMAResults
    from datetime import datetime, timedelta
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    import pyodbc
    import statsmodels.api as sm

    df_all_results = pd.DataFrame()

    # Iter through all crop models of crop_list
    for i in range(0, len(crop_list)):

        crop = crop_list[i][0]
        ctry = crop_list[i][1]
        trade_ctry = crop_list[i][2]
        ctgr = crop_list[i][3]
        mdel = crop_list[i][4]
        regn = crop_list[i][5]
        pkge = crop_list[i][6]
        crcy = crop_list[i][7]
        msre = crop_list[i][8]
        exog = crop_list[i][9]

        ctry_lc = ctry.lower()
        crop_lc = crop.lower()
        mdel_lc = mdel.lower()
        trade_ctry_lc = trade_ctry.lower()
        ctgr_lc = ctgr.lower()

        model_name = f'../../data/04_models/model_{mdel_lc}_{crop_lc}_{ctry_lc}_{trade_ctry_lc}_{ctgr_lc}.pkl'

        try:
            ld_model = ARIMAResults.load(model_name)
        except FileNotFoundError:
            print('No model found')
            break

        # FIRST save standard results obtained from model summary:
     
        results_summary = ld_model.summary()
        # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
        results_as_html = results_summary.tables[0].as_html()
        df_results_int = pd.read_html(results_as_html, index_col=0)[0]
        df_results_int['Crop'] = crop
        df_results_int['Country'] = ctry
        df_results_int['Trade_Country']  = trade_ctry
        df_results_int['Model'] = mdel
        df_results_int['Category']  = ctgr
        df_results = df_results_int.reset_index().iloc[:, 0:2].rename(columns={0: "Concept", 1: "Result"})
        df_results = df_results.append(df_results_int.reset_index().iloc[:, 2:4].rename(columns={2: "Concept", 3: "Result"})).dropna()
        df_results = df_results.join(df_results_int.reset_index().iloc[:, 4:])
        df_all_results = df_all_results.append(df_results)

        # SECOND save calculated measures. 
        # All models are fitted taking training values up to last day of previous year and inferenced the prediction for the next two years
        # Taking into account this MAE, MAPE, MSE and RMSE are measured
        start = date.today().strftime('%Y-01-01')
        end = (date.today() + relativedelta(years=1)).strftime('%Y-12-31')
        
        if mdel == 'SARIMAX':
            mdel_vols = 'SARIMA'
            df_pred_vols = inf.get_prediction_vols(ctry,crop,trade_ctry,regn,ctgr,pkge,crcy,msre,mdel_vols,start,end)
            exog = df_pred_vols[df_pred_vols.Date_ref > date.today().strftime('%Y-01-01')].drop(columns=['Volume']).set_index('Date_ref')

        df_pred = inf.get_prediction(ctry,crop,trade_ctry,regn,ctgr,pkge,crcy,msre,mdel,exog,start,end)

        df_pred = df_pred[(df_pred['Date_ref'].dt.year == date.today().year) & (df_pred['Date_ref'] < datetime.today()) & (df_pred['Price_estimated'] != 0) & (df_pred['Price'] != 0)]

        # MAE
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(df_pred.Price,df_pred.Price_estimated)
        new_row = {'Concept':'MAE', 'Result':mae, 'Crop':crop, 'Country':ctry, 'Trade_Country':trade_ctry, 'Model':mdel, 'Category':ctgr}
        df_all_results = df_all_results.append(new_row, ignore_index=True)

        # MAPE
        mape = np.mean(np.abs(df_pred.Price-df_pred.Price_estimated)/df_pred.Price_estimated)
        new_row = {'Concept':'MAPE', 'Result':mape, 'Crop':crop, 'Country':ctry, 'Trade_Country':trade_ctry, 'Model':mdel, 'Category':ctgr}
        df_all_results = df_all_results.append(new_row, ignore_index=True)

        # MSE
        from sklearn.metrics import mean_squared_error
        # Use against predictions (we must calculate the square root of the MSE)
        mse = mean_squared_error(df_pred.Price,df_pred.Price_estimated)
        new_row = {'Concept':'MSE', 'Result':mse, 'Crop':crop, 'Country':ctry, 'Trade_Country':trade_ctry, 'Model':mdel, 'Category':ctgr}
        df_all_results = df_all_results.append(new_row, ignore_index=True)

        # RMSE
        from sklearn.metrics import mean_squared_error
        # Use against predictions (we must calculate the square root of the MSE)
        rmse = np.sqrt(mean_squared_error(df_pred.Price,df_pred.Price_estimated))
        new_row = {'Concept':'RMSE', 'Result':rmse, 'Crop':crop, 'Country':ctry, 'Trade_Country':trade_ctry, 'Model':mdel, 'Category':ctgr}
        df_all_results = df_all_results.append(new_row, ignore_index=True)

        df_all_results['Result_num'] = df_all_results[df_all_results.Concept.isin(['AIC','BIC','HQIC','MAE','MAPE','MSE','RMSE'])].Result.apply(pd.to_numeric, errors='coerce')
        df_all_results['Result_num'].fillna(0, inplace = True)
    
    
    df_all_results.to_excel('../../data/04_models/results_summary.xlsx')

    return df_all_results


# %%
def load_measures_db(df_all_results):

    ##  Function to save model measures results into the database ## 

    import sys
    sys.path.insert(0, '../../src')
    #   https://realpython.com/python-modules-packages/
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import extractions as extract
    import transformations as transf
    import training as train
    import import_data as imp
    import inference as inf
    import pandas as pd
    import numpy as np
    import config
    import time  
    from dateutil.relativedelta import relativedelta
    from datetime import date
    import config as conf
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.arima_model import ARIMAResults
    from datetime import datetime, timedelta
    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    import pyodbc

    connStr = pyodbc.connect(config.db_con)
    cursor = connStr.cursor()

    # Load all data
    upd = 0

    for index,row in df_all_results.iterrows():
        if row['Result_num'] != 0: # Save only valid measures, not descriptive data
            cursor.execute("INSERT INTO dbo.models([Model],[Product],[Country],[Trade_Country],[Category],[Concept],[Result],[Updated]) values (?,?,?,?,?,?,?,?)",row['Model'],row['Crop'],row['Country'],row['Trade_Country'],row['Category'],row['Concept'],row['Result_num'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1

    cursor.close()
    connStr.close()

    return (print(upd," new prices added"))


