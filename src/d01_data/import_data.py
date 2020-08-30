# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Prices
# %% [markdown]
# ## Spain
# %% [markdown]
# ### Source
# %% [markdown]
# Market Data Source: 
# 
#     Junta de Andalucía (warehouse output prices paid to farmers)
# 
#     http://www.juntadeandalucia.es/agriculturaypesca/observatorio/servlet/FrontController?action=Static&subsector=19&url=subsector.jsp

# %%
from IPython.display import Image
Image("../../data/01_raw/prices/website_junta.png")

# %% [markdown]
# ### Import scripts

# %%
def get_prices_junta():

    ##  Function to get spanish prices from external files downloaded from official spanish region government website  #

    import pandas as pd
    import datetime
    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import config
    import pandas as pd
    import numpy as np
    import pyodbc
    import re

    pd.set_option('display.max_columns',None) ## Display all columns in pandas dataframe  
    pd.set_option('display.max_rows',None) ## Display all rows in pandas dataframe 

    sheet_name = 'Observatorio de Precios'
    rows_skip_13 = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    rows_skip_14 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    rows_skip_15 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    cols = [0,1]
    file_names_year = [  
                        ['../../data/01_raw/prices/ArandanoPreciosAgricultor.xls',sheet_name,rows_skip_15,cols,'PriceProducer','BLUEBERRIES',0,25],\
                        ['../../data/01_raw/prices/FrambuesaPreciosAgricultor.xls',sheet_name,rows_skip_15,cols,'PriceProducer','RASPBERRIES',35,25],\
                        ['../../data/01_raw/prices/FresaPreciosAgricultor.xls',sheet_name,rows_skip_15,cols,'PriceProducer','STRAWBERRIES',48,22]\
                    ]

    price = pd.DataFrame()
    price = pd.read_excel(file_names_year[0][0]                                ,sheet_name = file_names_year[0][1]                                ,header = 6                                ,skiprows = file_names_year[0][2]                                ,usecols = file_names_year[0][3])
    price.columns = ['Week',file_names_year[0][4]]
    price[['Week_No','Year']] = price.Week.str.split(pat='-',expand=True)
    price['Week_No'] = price['Week_No'].astype('int32')
    price['Year'] = price['Year'].astype('int32')
    price.drop_duplicates(inplace=True)
    price['Crop'] = file_names_year[0][5]
    price['Week_Campaign'] = price['Week_No'].apply(lambda x: (x - file_names_year[0][6] + 53) % 53 )
    price['Week_Campaign'] = price['Week_Campaign'].astype('int32')
    price['Year_Campaign'] = price['Week_No'].apply(lambda x : 0 if 0 == file_names_year[0][6] else 0 if x < file_names_year[0][6] else 1)
    price['Year_Campaign'] = price['Year_Campaign'].astype('int32')
    price.to_excel(f'../../data/02_intermediate/Price_Excel{file_names_year[0][5]}.xlsx')
    price['Year_Campaign'] = price['Year_Campaign'] + price['Year']

    for i in range(1,len(file_names_year)):
        price_excel=pd.read_excel(file_names_year[i][0]                                ,sheet_name=file_names_year[i][1]                                ,header=6                                ,skiprows=file_names_year[i][2]                                ,usecols=file_names_year[i][3])
        price_excel.columns = ['Week',file_names_year[i][4]]
        price_excel[['Week_No','Year']] = price_excel.Week.str.split(pat='-',expand=True)
        price_excel['Week_No'] = price_excel['Week_No'].astype('int32')
        price_excel['Year'] = price_excel['Year'].astype('int32')
        price_excel.drop_duplicates(inplace=True)
        price_excel['Crop'] = file_names_year[i][5]
        # Common campaign dates:
            # Strawberry campaign from week 49 to 22
            # Blueberry campaign from week 1 to 25
            # Raspberry campaign from week 36 to 25
        price_excel['Week_Campaign'] = price_excel['Week_No'].apply(lambda x: (x - file_names_year[i][6] + 53) % 53 )
        price_excel['Week_Campaign'] = price_excel['Week_Campaign'].astype('int32')
        price_excel['Year_Campaign'] = price_excel['Week_No'].apply(lambda x : 0 if 0 == file_names_year[i][6] else 0 if x < file_names_year[i][6] else 1)
        price_excel['Year_Campaign'] = price_excel['Year_Campaign'].astype('int32')
        price_excel['Year_Campaign'] = price_excel['Year_Campaign'] + price_excel['Year'] 
        price = price.append(price_excel)
        price_excel.to_excel(f'../../data/02_intermediate/Price_Excel{file_names_year[i][5]}.xlsx')
        price_excel[price_excel['Crop']==file_names_year[i][5]].groupby(['Year_Campaign'])['PriceProducer'].describe().transpose().to_excel(f'./../../data/02_intermediate/summary_{file_names_year[i][5]}.xlsx')
            
    price['Date_Ref']=price['Week'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%V-%G-%u') )
    price.drop(range(0,price.PriceProducer.notna().idxmax()),inplace=True)
    price = price [['Crop',                    'PriceProducer',                    'Week',                    'Week_No',                    'Year',                    'Date_Ref',                    'Week_Campaign',                    'Year_Campaign']]
    price.dropna(subset=['PriceProducer'], inplace=True)
    price['Country']='ES'
    price['Currency']='EUR'
    price['Measure']='KG'
    price['Region']='ANDALUSIA'
    price['Trade_Country']='ES'
    price['Category']='std'
    price['Package']='bulk'

    price.rename(columns={'PriceProducer' : 'Price', 'Crop' : 'Product'},errors="raise",inplace=True)

    # Remove all prices previous to ndays config parameter
    fdate = datetime.datetime.today() - datetime.timedelta(days=config.ndays)
    price.drop(price[price.Date_Ref < fdate].index,inplace=True)

    return price

# %% [markdown]
# ## United States
# %% [markdown]
# Market Data Source: 
# 
#     USDA (warehouse output prices paid to farmers)
# 
#     https://www.ams.usda.gov/market-news/fruits-vegetables

# %%
from IPython.display import Image
Image("../../data/01_raw/prices/website_usda.png")

# %% [markdown]
# ### Import scripts

# %%
def get_prices_usda(crop,crop_abb):

    ##  Function to get US prices directly from from official USDA website  #

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import config
    import pandas as pd
    from datetime import date, datetime, timedelta
    import numpy as np
    import pyodbc

    # Read format conversions to KG
    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    qry_formats = "SELECT * FROM [Prices].[dbo].[formats]"
    df_formats = pd.read_sql(qry_formats, connStr)

    cursor.close()
    connStr.close()

    # Setting dates
    # Date from
    fdate = datetime.today() - timedelta(days=config.ndays)
    fday = fdate.strftime('%d')
    fmonth = fdate.strftime('%m')
    fyear = fdate.strftime('%Y')

    # Date to : current date data to collect updated information
    tday = date.today().strftime('%d')
    tmonth = date.today().strftime('%m')
    tyear = date.today().strftime('%Y')

    # URL for accessing prices
    USprices = f"https://www.marketnews.usda.gov/mnp/fv-report-top-filters?&commAbr={crop_abb}&varName=&locAbr=&repType=shipPriceDaily&navType=byComm&locName=&navClass=&type=shipPrice&dr=1&volume=&commName={crop}&navClass,=&portal=fv&region=&repDate={fmonth}%2F{fday}%2F{fyear}&endDate={tmonth}%2F{tday}%2F{tyear}&format=excel&rebuild=false"
    
    # Assign the table data in html format to a Pandas dataframe
    table =  pd.read_html(USprices,header=0,parse_dates=['Date'])[0]

    # Read the table in new dataframe with the main info
    prices = table[['Commodity Name',                        'City Name',                        'Package',                        'Type',                        'Item Size',                        'Date',                        'Low Price',                        'High Price',                        'Mostly Low',                        'Mostly High',                        'Season']]

    #################### Cleaning data ######################

    # Delete rows if no price available
    prices.dropna(axis='index',how='all',subset=['High Price','Low Price','Mostly Low','Mostly High'],inplace=True)

    # New Category field based on item size + type
    prices['Item Size'].fillna(value='std',inplace=True)
    prices['Type'].fillna(value='',inplace=True)
    prices['Category'] = (prices['Item Size'] + ' ' + prices['Type'].str.lower()).str.strip()

    # Average price based on fields Low Price -	High Price - Mostly Low - Mostly High
    prices['Mostly Avg'] = prices[['Mostly High','Mostly Low']].mean(axis=1)
    prices['Avg Price'] = prices[['Low Price','High Price']].mean(axis=1)
    for i in enumerate(prices[~prices['Mostly Avg'].isnull()].index):
        prices.at[[i[1]], "Avg Price"] = prices['Mostly Avg']

    # Campaign dates
    campaign_dates = prices.groupby('Season')['Date'].agg('min').reset_index()
    campaign_dates.columns = ['Season','First Date']
    prices = pd.merge(left=prices, right=campaign_dates, how='left', left_on='Season', right_on='Season')
    prices['Week_num_campaign'] = np.ceil((prices['Date'] - prices['First Date']).dt.days.astype('int16')/7).astype(int)
    prices['Week_num_campaign'] = prices['Week_num_campaign'].apply(lambda x: x + 1 if x == 0 else x)

    # Convert price to kg
    prices = prices.merge(df_formats, left_on='Package', right_on='Format',how='left')
    prices['Avg Price'] = prices['Avg Price'] / prices['Weight']

    # New column for origin of imports
    prices['Origin'] = prices['City Name'].apply(lambda x : transf.label_origin(x))

    # Naming of Region
    prices['City Name'] = prices['City Name'].apply(lambda x : transf.label_region(x))

    # Some City Name renaming
    prices['City Name'].replace({'ARGENTINA IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT','BRITISH COLUMBIA CROSSINGS THROUGH NORTHWEST WASHINGTON' : 'NORTHWEST WASHINGTON','MEXICO CROSSINGS THROUGH OTAY MESA' : 'OTAY MESA (SAN DIEGO)','CHILE IMPORTS - PORT OF ENTRY MIAMI AREA' : 'MIAMI','OREGON AND WASHINGTON' : 'OREGON AND WASHINGTON','URUGUAY IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT','ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTH FLORIDA' : 'SOUTH FLORIDA','PERU IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'SOUTHERN CALIFORNIA','CHILE IMPORTS - PORT OF ENTRY PHILADELPHIA AREA' : 'PHILADELPHIA','SOUTH GEORGIA' : 'PHILADELPHIA','MEXICO CROSSINGS THROUGH ARIZONA, CALIFORNIA AND TEXAS' : 'ARIZONA, CALIFORNIA AND TEXAS','CHILE IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT','ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'SOUTHERN CALIFORNIA','ARGENTINA IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT','PERU IMPORTS - PORTS OF ENTRY PHILADELPHIA AREA AND NEW YORK CITY AREA' : 'PHILADELPHIA','CHILE IMPORTS - PORT OF ENTRY LOS ANGELES AREA' : 'LOS ANGELES','CHILE IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT','MEXICO CROSSINGS THROUGH TEXAS' : 'TEXAS'}, inplace=True)

    # Informative fields
    prices['Crop']=crop
    prices['Country']='US'
    prices['Currency']='USD'
    prices['Measure']='KG'

    # Taking only relevant columns
    prices = prices[['Crop','Country','City Name','Origin','Category','Package','Season','Week_num_campaign','Date','Currency','Measure','Avg Price']]
    prices.rename(columns={'Crop' : 'Product','Country' : 'Country','City Name' : 'Region','Origin' : 'Trade_Country', 'Category' : 'Category','Package' : 'Package','Season' : 'Year_Campaign','Week_num_campaign' : 'Week_Campaign','Date' : 'Date_Ref','Currency' : 'Currency','Measure' : 'Measure','Avg Price' : 'Price'},errors="raise",inplace=True)

    # Delete duplicates
    prices.drop_duplicates(inplace=True)

    return prices

# %% [markdown]
# ## Common scripts

# %%
def load_prices_bbdd(df_prices):

    ##  Function to upload prices retrieved from different sources to SQL Server Database   #

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d00_utils')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import config
    import pandas as pd
    from datetime import date, datetime, timedelta
    import numpy as np
    import pyodbc
    import re

    ctry = re.sub('[^A-Za-z0-9]+', "','", str(df_prices.Country.unique()))[2:-2]
    crop = re.sub('[^A-Za-z0-9]+', "','", str(df_prices.Product.unique()))[2:-2]

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    # Setting dates
    # Date from
    fdate = datetime.today() - timedelta(days=config.ndays)
    fday = fdate.strftime('%d')
    fmonth = fdate.strftime('%m')
    fyear = fdate.strftime('%Y')

    # Date to : current date data to collect updated information
    tday = date.today().strftime('%d')
    tmonth = date.today().strftime('%m')
    tyear = date.today().strftime('%Y')

    # Delete all data with price dates greater than the ndays parameter last days from today
# Delete all data with price dates greater than the ndays parameter last days from today
    qry_delete = f"DELETE FROM [Prices].[dbo].[prices] where [Country] = {ctry} and [Product] IN ({crop}) and Date_price > '{fdate}'"
    cursor.execute(qry_delete)
    connStr.commit()

    # Load all data with price dates greater than the ndays global parameter last days from today
    upd = 0
    try:
        for index,row in df_prices.iterrows():
            if row['Date_Ref'] > fdate: # Python price line date must be greater than the max date in SQL table
                cursor.execute("INSERT INTO dbo.prices([Product],[Country],[Region],[Trade_Country],[Category],[Package],[Campaign],[Campaign_wk],[Date_price],[Currency],[Measure],[Price],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Country'],row['Category'],row['Package'],row['Year_Campaign'],row['Week_Campaign'],row['Date_Ref'],row['Currency'],row['Measure'],row['Price'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                connStr.commit()
                upd += 1
    except TypeError: # If there price is null no posibility to compare operands
        for index,row in df_prices.iterrows(): # When there are no prices in SQL
            cursor.execute("INSERT INTO dbo.prices([Product],[Country],[Region],[Trade_Country],[Category],[Package],[Campaign],[Campaign_wk],[Date_price],[Currency],[Measure],[Price],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Country'],row['Category'],row['Package'],row['Year_Campaign'],row['Week_Campaign'],row['Date_Ref'],row['Currency'],row['Measure'],row['Price'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1
    cursor.close()
    connStr.close()
    print(upd," new prices added")

# %% [markdown]
# # Volumes
# %% [markdown]
# ## European Union
# %% [markdown]
# ### Source
# %% [markdown]
# Market Data Source: 
# 
#     IBO (International Blueberry Association)
# 
#     https://www.internationalblueberry.org/

# %%
from IPython.display import Image
Image("../../data/01_raw/volumes/website_ibo.png")

# %% [markdown]
# ### Import scripts

# %%
def get_volumes_ibo():

    ##  Function to get UE import volumes from external files downloaded from international blueberry organization website  #

    import os
    import pandas as pd
    import datetime

    directory = '../../data/01_raw/volumes/'
    volume = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".xls"):
            #print(os.path.join(directory, filename))
            curr = pd.DataFrame(pd.read_excel(os.path.join(directory, filename)))
            curr['Filename'] = filename
            volume = volume.append(curr)
        else:
            continue
    volume.drop('Week.1', axis=1, inplace=True)

    # Validation
    df_all = pd.DataFrame(volume[volume.Filename=='BLUEBERRIES-SUM.xls'].agg('sum')).drop(['Week','Filename'])
    df_countries = volume[volume.Filename!='BLUEBERRIES-SUM.xls'].agg('sum').drop(['Week','Filename'])
    if (df_countries.subtract(df_all, fill_value=0).sum()[0] != 0.):
        raise Exception('Volume files unbalanced')

    # Cleanance

    # new data frame with split value columns 
    new = volume.Filename.str.split("-", n = 1, expand = True) 
    # making separate first name column from new data frame 
    volume['Product']= new[0] 
    # making separate last name column from new data frame 
    volume['Trade_country']= new[1].str.rstrip('.xls')
    # Dropping old Name columns 
    volume.drop(columns =['Filename'], inplace = True) 
    # Fill NaNs with 0 values
    volume.fillna(0, inplace=True)
    # Convert columns in rows
    volume = volume.melt(id_vars=['Week','Product','Trade_country'], 
            var_name="Year", 
            value_name="Volume")
    # Drop 'Sum' lines which include all countries groupped
    volume.drop(volume[volume['Trade_country']=='SUM'].index, inplace=True)
    # Set date column
    volume.rename(columns={"Week": "Week_num"}, inplace = True)
    volume['Week_desc'] = volume['Week_num'].astype('str').str.cat(volume['Year'].astype('str'), sep ="-")
    volume['Date_ref'] = volume['Week_desc'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%V-%G-%u') )
    # Add fixed values
    volume['Measure'] = 'KG'
    volume['Country'] = 'EU'
    volume['Region'] = ''
    volume['Trade_Type'] = 'Import'
    volume['Category'] = 'std'
    volume['Package'] = 'std'
    volume['Transport'] = 'na'

    return volume


# %%
def load_volumes_bbdd(df_volumes):

    ##  Function to upload prices retrieved to SQL Server Database   #

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')
    import transformations as transf
    import pyodbc
    from datetime import datetime, timedelta

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    # Delete all data with volumes dates greater than the ndays parameter last days from today
    N = 60 
    rep_date = datetime.now().date() - timedelta(days=N)
    qry_delete = f"DELETE FROM [Prices].[dbo].[volumes] where cast([Country] as nvarchar) = cast('EU' as nvarchar) and Date_volume > '{rep_date}'"
    cursor.execute(qry_delete)

    # Load all data with volumes dates greater than the ndays parameter last days from today
    upd = 0

    try:
        for index,row in df_volumes.iterrows():
            if row['Date_ref'] > rep_date: # Python volumes line date must be greater than the max date in SQL table
                cursor.execute("INSERT INTO dbo.volumes([Product],[Country],[Region],[Trade_Type],[Trade_Country],[Category],[Package],[Transport],[Campaign],[Campaign_wk],[Date_volume],[Measure],[Volume],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Type'],transf.label_trade_countries(row['Trade_country']),row['Category'],row['Package'],row['Transport'],row['Year'],row['Week_num'],row['Date_ref'],row['Measure'],row['Volume'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                connStr.commit()
                upd += 1
    except TypeError: # If there volume is null no posibility to compare operands
        for index,row in df_volumes.iterrows(): # When there are no volumes in SQL
            cursor.execute("INSERT INTO dbo.volumes([Product],[Country],[Region],[Trade_Type],[Trade_Country],[Category],[Package],[Transport],[Campaign],[Campaign_wk],[Date_volume],[Measure],[Volume],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Type'],transf.label_trade_countries(row['Trade_country']),row['Category'],row['Package'],row['Transport'],row['Year'],row['Week_num'],row['Date_ref'],row['Measure'],row['Volume'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1
    print(upd," new volumes added")

    cursor.close()
    connStr.close()


