# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def get_prices(crop,ctry,trade_ctry,ctgr):

    ##  Function to get prices available    ##
    
    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
    import pandas as pd
    import pyodbc

    connStr = pyodbc.connect(config.db_con)
    cursor = connStr.cursor()

    qry = f"SELECT [Product],[Country],[Trade_Country],[Category],[Date_price],[Campaign],AVG([Price]) as [Price] FROM [Prices].[dbo].[prices] WHERE [Product] = '{crop}' AND [Country] = '{ctry}' AND [Trade_Country] = '{trade_ctry}' AND [Category] = '{ctgr}' GROUP BY [Product],[Country],[Trade_Country],[Category],[Date_price],[Campaign]"
    df_prices = pd.read_sql(qry, connStr)

    return df_prices


# %%
def get_prices_interpolated(crop,ctry,trade_ctry,ctgr):
    
    ##  Function to get all prices interpolated weekly and mean, removing the first campaign available  ##

    df_prices = get_prices(crop,ctry,trade_ctry,ctgr)
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
def get_null_prices(crop,ctry,trade_ctry,ctgr):
    
    ##  Function to get weeks without prices informed -no campaign period-  ##

    import pandas as pd

    df_prices = get_prices(crop,ctry,trade_ctry,ctgr)
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

    ##  Function to get UE volumes import from Spain (Agronometrics), weekly aggregated ##
    
    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
    import pandas as pd
    import pyodbc
    
    connStr = pyodbc.connect(config.db_con)
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

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    import numpy as np
    import paths

    # May need to lag 1 year in order to allocate campaign costs to the actual fresh produce sales  
    yr_adj = 1

    # 1985-2017 file
    df = pd.read_excel(paths.es_labour_file,sheet_name='IS',header=3,usecols = ['AÑO','Anual'])
    df.AÑO = pd.to_datetime(df.AÑO, format='%Y')
    df.rename(columns={'AÑO': 'year', 'Anual': 'labour_index'}, inplace=True)
    df = df[(df.year.duplicated(keep='first')==False) & df.year.notnull()]
    df.year = df.year.apply(lambda x : x + relativedelta(years=yr_adj))
    df.set_index('year', inplace=True)
    df_salaries = df

    # 2018-2020 file
    df = pd.read_excel(paths.es_labour_file2,sheet_name='IndSal2',header=3,usecols=['AÑO','Enero','Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiem.', 'Octubre', 'Noviem.', 'Diciem.'])
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


# %%
def get_prices_state(crop,ctry,trade_ctry,ctgr):

    ##  Function to get prices available    ##
    
    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
    import pandas as pd
    import pyodbc

    connStr = pyodbc.connect(config.db_con)
    cursor = connStr.cursor()

    qry = f"SELECT reg.[State], prices.[Product],prices.[Country],prices.[Trade_Country],prices.[Category],prices.[Date_price],prices.[Campaign],AVG([Price]) as [Price] FROM [Prices].[dbo].[prices] left join [Prices].[dbo].[regions] reg on reg.Region = prices.Region and reg.Country = prices.Country WHERE [Product] = '{crop}' AND prices.[Country] = '{ctry}' AND prices.[Trade_Country] = '{trade_ctry}' AND prices.[Category] = '{ctgr}' GROUP BY reg.[State], prices.[Product],prices.[Country],prices.[Trade_Country],prices.[Category],prices.[Date_price],prices.[Campaign]"
    df_prices = pd.read_sql(qry, connStr)

    return df_prices


# %%
def get_volumes_state(crop,ctry,trade_ctry):

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import extractions as extract
    import config
    import pandas as pd
    import pyodbc

    ##  Function to get UE volumes import from Spain (Agronometrics), weekly aggregated ##
    
    connStr = pyodbc.connect(config.db_con)
    cursor = connStr.cursor()

    qry = f"SELECT reg.[State], vols.[Product],vols.[Country],vols.[Trade_Country], vols.[Date_volume],vols.[Campaign],SUM([Volume]) as [Volume] FROM [Prices].[dbo].[volumes] vols left join [Prices].[dbo].[regions] reg on reg.Region = vols.Region and vols.Country = vols.Country where vols.[Country] = '{ctry}' and vols.[Product] = '{crop}' and vols.[Trade_Country] ='{trade_ctry}' group by reg.[State], vols.[Product],vols.[Country],vols.[Trade_Country], vols.[Date_volume],vols.[Campaign]"

    df_volumes = pd.read_sql(qry, connStr)

    return df_volumes


# %%
def get_plotting_analysis(df_prices, df_prices_full, desc):

    # https://www.machinelearningplus.com/time-series/time-series-analysis-python/#:~:text=Time%20series%20is%20a%20sequence,in%20Python%20%E2%80%93%20A%20Comprehensive%20Guide.

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import extractions as extract
    import transformations as transf
    import training as train
    import import_data as imp
    import inference as inf
    import time  
    from dateutil.relativedelta import relativedelta
    from datetime import date
    import config as conf
    import pandas as pd
    from statsmodels.tsa.stattools import acf
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.stattools import pacf
    from statsmodels.graphics.tsaplots import plot_pacf
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns

    pp = PdfPages(f'../../data/02_intermediate/exloratory_analysis/{desc}.pdf')

    df_prices['Week'] = df_prices.index.week
    df_prices['Year'] = df_prices.index.year

    # Evolution (line graph)
    plot1 = plt.figure()
    for i in list(df_prices.index.year.drop_duplicates()):
        data_graph = df_prices[df_prices.index.year == i]
        ax = sns.lineplot(x='Week',y='Price',data=data_graph, label=i)
    plt.title('Evolution of market prices', fontsize=14)
    plt.xlabel('Week', fontsize=10)
    plt.ylabel('Producer Price (Local currency/Kg)', fontsize=10)
    plt.savefig(f'../../data/02_intermediate/exloratory_analysis/{desc}_evolution.png')
    pp.savefig(plot1)

    # Distribution univariate kernel density estimate)
    plot2 = plt.figure()
    for i in list(df_prices.index.year.drop_duplicates()):
        data_graph = df_prices[df_prices.index.year == i]
        ax = sns.kdeplot(data_graph['Price'].dropna(),label=i, shade=True)
    plt.title('Distribution market prices', fontsize=14)
    plt.xlabel('Producer Price (Local currency/Kg)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.savefig(f'../../data/02_intermediate/exloratory_analysis/{desc}_distribution.png')
    pp.savefig(plot2)

    # Boxplot for years
    plot3 = plt.figure()
    ax = sns.boxplot(x="Year", y="Price",data=df_prices, showmeans=True)
    plt.title('Year-wise boxplot for market prices (trend)', fontsize=14)
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Producer Price (Local currency/Kg)', fontsize=10)
    plt.savefig(f'../../data/02_intermediate/exloratory_analysis/{desc}_boxplot_years.png')
    pp.savefig(plot3)

    # Boxplot for weeks
    plot4 = plt.figure(figsize=(14,6))
    ax = sns.boxplot(x="Week", y="Price",data=df_prices, showmeans=True)
    plt.title('Week-wise boxplot for market prices (seasonality)', fontsize=14)
    plt.xlabel('Week', fontsize=1)
    plt.ylabel('Producer Price (Local currency/Kg)', fontsize=10)
    plt.savefig(f'../../data/02_intermediate/exloratory_analysis/{desc}_boxplot_weeks.png')
    pp.savefig(plot4)

    # ACF correlation
    from statsmodels.graphics.tsaplots import plot_acf
    acf(df_prices_full)
    plot5 = plot_acf(df_prices_full,lags=100,unbiased=True)
    plot5.savefig(f'../../data/02_intermediate/exloratory_analysis/{desc}_acf.png')
    pp.savefig(plot5)
    
    from statsmodels.tsa.stattools import pacf
    from statsmodels.graphics.tsaplots import plot_pacf
    plot6 = plot_pacf(df_prices_full, lags = 30)
    plot6.savefig(f'../../data/02_intermediate/exloratory_analysis/{desc}_pacf.png')
    pp.savefig(plot6)

    # Correlation between related campaign weeks
    for y in range(2017,2020):
        c = df_prices_full[df_prices_full.index.year==y]['Price'].reset_index()['Price'][0:21].corr(df_prices_full[df_prices_full.index.year==y+1].reset_index()['Price'][0:21])
        print(f'Correlations between campaigns {y} and {y+1}: {c}')


    pp.close()


