# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # US market analysis
# %% [markdown]
# ## Prices

# %%
def label_origin(row):
    switcher = {
        'ARGENTINA IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'AR',
        'BRITISH COLUMBIA CROSSINGS THROUGH NORTHWEST WASHINGTON' : 'CA',
        'MEXICO CROSSINGS THROUGH OTAY MESA' : 'MX',
        'CHILE IMPORTS - PORT OF ENTRY MIAMI AREA' : 'CL',
        'OREGON AND WASHINGTON' : 'US',
        'ORANGE-SAN DIEGO COUNTIES & COACHELLA DISTRICT, CALIFORNIA' : 'US',
        'OXNARD DISTRICT CALIFORNIA' : 'US',
        'URUGUAY IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'UY',
        'ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTH FLORIDA' : 'UY',
        'PERU IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'PE',
        'CHILE IMPORTS - PORT OF ENTRY PHILADELPHIA AREA' : 'CL',
        'SOUTH GEORGIA' : 'US',
        'MEXICO CROSSINGS THROUGH ARIZONA, CALIFORNIA AND TEXAS' : 'MX',
        'EASTERN NORTH CAROLINA' : 'US',
        'CENTRAL AND SOUTHERN SAN JOAQUIN VALLEY CALIFORNIA' : 'US',
        'SOUTH & CENTRAL DISTRICT CALIFORNIA' : 'US',
        'CHILE IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'CL',
        'CENTRAL & NORTH FLORIDA' : 'US',
        'ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'UY',
        'MICHIGAN' : 'US',
        'ARGENTINA IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'AR',
        'PERU IMPORTS - PORTS OF ENTRY PHILADELPHIA AREA AND NEW YORK CITY AREA' : 'PE',
        'SOUTH NEW JERSEY' : 'US',
        'CHILE IMPORTS - PORT OF ENTRY LOS ANGELES AREA' : 'CL',
        'CHILE IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'CL',
        'CENTRAL FLORIDA' : 'US',
        'SALINAS-WATSONVILLE CALIFORNIA' : 'US',
        'SANTA MARIA CALIFORNIA' : 'US',
        'MEXICO CROSSINGS THROUGH TEXAS' : 'MX',
        'ANDALUSIA' : 'ES'
    }
    return switcher.get(row, "nan")


# %%
def label_region(row):
    switcher = {
        'ARGENTINA IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT',
        'BRITISH COLUMBIA CROSSINGS THROUGH NORTHWEST WASHINGTON' : 'NORTHWEST WASHINGTON',
        'MEXICO CROSSINGS THROUGH OTAY MESA' : 'OTAY MESA (SAN DIEGO)',
        'CHILE IMPORTS - PORT OF ENTRY MIAMI AREA' : 'MIAMI',
        'OREGON AND WASHINGTON' : 'OREGON AND WASHINGTON',
        'ORANGE-SAN DIEGO COUNTIES & COACHELLA DISTRICT, CALIFORNIA' : 'ORANGE-SAN DIEGO COUNTIES & COACHELLA DISTRICT, CALIFORNIA',
        'OXNARD DISTRICT CALIFORNIA' : 'OXNARD DISTRICT CALIFORNIA',
        'URUGUAY IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT',
        'ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTH FLORIDA' : 'SOUTH FLORIDA',
        'PERU IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'SOUTHERN CALIFORNIA',
        'CHILE IMPORTS - PORT OF ENTRY PHILADELPHIA AREA' : 'PHILADELPHIA',
        'SOUTH GEORGIA' : 'PHILADELPHIA',
        'MEXICO CROSSINGS THROUGH ARIZONA, CALIFORNIA AND TEXAS' : 'ARIZONA, CALIFORNIA AND TEXAS',
        'EASTERN NORTH CAROLINA' : 'EASTERN NORTH CAROLINA',
        'CENTRAL AND SOUTHERN SAN JOAQUIN VALLEY CALIFORNIA' : 'CENTRAL AND SOUTHERN SAN JOAQUIN VALLEY CALIFORNIA',
        'SOUTH & CENTRAL DISTRICT CALIFORNIA' : 'SOUTH & CENTRAL DISTRICT CALIFORNIA',
        'CHILE IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT',
        'CENTRAL & NORTH FLORIDA' : 'CENTRAL & NORTH FLORIDA',
        'ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'SOUTHERN CALIFORNIA',
        'MICHIGAN' : 'MICHIGAN',
        'ARGENTINA IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT',
        'PERU IMPORTS - PORTS OF ENTRY PHILADELPHIA AREA AND NEW YORK CITY AREA' : 'PHILADELPHIA',
        'SOUTH NEW JERSEY' : 'SOUTH NEW JERSEY',
        'CHILE IMPORTS - PORT OF ENTRY LOS ANGELES AREA' : 'LOS ANGELES',
        'CHILE IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT',
        'CENTRAL FLORIDA' : 'CENTRAL FLORIDA',
        'SALINAS-WATSONVILLE CALIFORNIA' : 'SALINAS-WATSONVILLE CALIFORNIA',
        'SANTA MARIA CALIFORNIA' : 'SANTA MARIA CALIFORNIA',
        'MEXICO CROSSINGS THROUGH TEXAS' : 'TEXAS'
    }
    return switcher.get(row, "nan")


# %%
def load_prices(pcrop,pcrop_abb,ndays):

    # Importing libraries 
    import pandas as pd
    from datetime import date, datetime, timedelta
    import numpy as np
    import pyodbc

    # We read the format conversion to KG (more than 20 different)
    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    qry_formats = "SELECT * FROM [Prices].[dbo].[formats]"
    df_formats = pd.read_sql(qry_formats, connStr)

    cursor.close()
    connStr.close()

    # Setting dates
    # Date from
    fdate = datetime.today() - timedelta(days=ndays)
    fday = fdate.strftime('%d')
    fmonth = fdate.strftime('%m')
    fyear = fdate.strftime('%Y')

    # Date to : current date data to collect updated information
    tday = date.today().strftime('%d')
    tmonth = date.today().strftime('%m')
    tyear = date.today().strftime('%Y')

    # URL for accessing prices
    USprices = f"https://www.marketnews.usda.gov/mnp/fv-report-top-filters?&commAbr={pcrop_abb}&varName=&locAbr=&repType=shipPriceDaily&navType=byComm&locName=&navClass=&type=shipPrice&dr=1&volume=&commName={pcrop}&navClass,=&portal=fv&region=&repDate={fmonth}%2F{fday}%2F{fyear}&endDate={tmonth}%2F{tday}%2F{tyear}&format=excel&rebuild=false"
    
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
    #prices['Year'] = prices['Date'].dt.year
    #prices['Week'] = prices['Date'].dt.week
    #prices['Week_desc'] = prices['Week'].apply(str) + "-" + prices['Year'].apply(str)

    # Convert price to kg
    prices = prices.merge(df_formats, left_on='Package', right_on='Format',how='left')
    prices['Avg Price'] = prices['Avg Price'] / prices['Weight']

    # New column for origin of imports
    prices['Origin'] = prices['City Name'].apply(lambda x : label_origin(x))

    # Naming of Region
    prices['City Name'] = prices['City Name'].apply(lambda x : label_region(x))

    # Some City Name renaming
    prices['City Name'].replace({'ARGENTINA IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT','BRITISH COLUMBIA CROSSINGS THROUGH NORTHWEST WASHINGTON' : 'NORTHWEST WASHINGTON','MEXICO CROSSINGS THROUGH OTAY MESA' : 'OTAY MESA (SAN DIEGO)','CHILE IMPORTS - PORT OF ENTRY MIAMI AREA' : 'MIAMI','OREGON AND WASHINGTON' : 'OREGON AND WASHINGTON','URUGUAY IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT','ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTH FLORIDA' : 'SOUTH FLORIDA','PERU IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'SOUTHERN CALIFORNIA','CHILE IMPORTS - PORT OF ENTRY PHILADELPHIA AREA' : 'PHILADELPHIA','SOUTH GEORGIA' : 'PHILADELPHIA','MEXICO CROSSINGS THROUGH ARIZONA, CALIFORNIA AND TEXAS' : 'ARIZONA, CALIFORNIA AND TEXAS','CHILE IMPORTS - PORT OF ENTRY LOS ANGELES INTERNATIONAL AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT','ARGENTINA/URUGUAY IMPORTS - PORTS OF ENTRY SOUTHERN CALIFORNIA' : 'SOUTHERN CALIFORNIA','ARGENTINA IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT','PERU IMPORTS - PORTS OF ENTRY PHILADELPHIA AREA AND NEW YORK CITY AREA' : 'PHILADELPHIA','CHILE IMPORTS - PORT OF ENTRY LOS ANGELES AREA' : 'LOS ANGELES','CHILE IMPORTS - PORT OF ENTRY MIAMI INTERNATIONAL AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT','MEXICO CROSSINGS THROUGH TEXAS' : 'TEXAS'}, inplace=True)

    # Informative fields
    prices['Crop']=pcrop
    prices['Country']='US'
    prices['Currency']='USD'
    prices['Measure']='KG'

    # Taking only relevant columns
    prices = prices[['Crop','Country','City Name','Origin','Category','Package','Season','Week_num_campaign','Date','Currency','Measure','Avg Price']]
    prices.rename(columns={'Crop' : 'Product','Country' : 'Country','City Name' : 'Region','Origin' : 'Trade_Country', 'Category' : 'Category','Package' : 'Package','Season' : 'Campaign','Week_num_campaign' : 'Campaign_wk','Date' : 'Date_price','Currency' : 'Currency','Measure' : 'Measure','Avg Price' : 'Price'},errors="raise",inplace=True)

    # Delete duplicates
    prices.drop_duplicates(inplace=True)

    # Checks
    #   list(set(prices['Category']))
    #   prices[prices['Avg Price'].isnull()].shape
    #   prices[~prices['Mostly Avg'].isnull()].shape
    #   prices[~prices['Price Avg'].isnull()].shape
    #   prices[prices['Price Avg'].isnull()].shape

    #################### SQL statements ######################

    import pyodbc
    from datetime import datetime, timedelta

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    # Delete all data with price dates greater than the ndays parameter last days from today
    qry_delete = f"DELETE FROM [Prices].[dbo].[prices] where cast([Country] as nvarchar) = cast('US' as nvarchar) and cast([Product] as nvarchar) = cast('{pcrop}' as nvarchar) and Date_price > '{fdate}'"
    cursor.execute(qry_delete)

    # Load all data with price dates greater than the ndays parameter last days from today
    upd = 0

    try:
        for index,row in prices.iterrows():
            if row['Date_price'] > fdate: # Python price line date must be greater than the max date in SQL table
                cursor.execute("INSERT INTO dbo.prices([Product],[Country],[Region],[Trade_Country],[Category],[Package],[Campaign],[Campaign_wk],[Date_price],[Currency],[Measure],[Price],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Country'],row['Category'],row['Package'],row['Campaign'],row['Campaign_wk'],row['Date_price'],row['Currency'],row['Measure'],row['Price'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                connStr.commit()
                upd += 1
    except TypeError: # If there price is null no posibility to compare operands
        for index,row in prices.iterrows(): # When there are no prices in SQL
            cursor.execute("INSERT INTO dbo.prices([Product],[Country],[Region],[Trade_Country],[Category],[Package],[Campaign],[Campaign_wk],[Date_price],[Currency],[Measure],[Price],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Country'],row['Category'],row['Package'],row['Campaign'],row['Campaign_wk'],row['Date_price'],row['Currency'],row['Measure'],row['Price'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1
    print(upd," new prices added")

    cursor.close()
    connStr.close()

# %% [markdown]
# ## Volumes

# %%
def label_region_volumes(row):
    switcher = {
        'ALL AREAS' : 'ALL AREAS',
        'CANADA CROSSINGS THRU WASHINGTON' : 'NORTHWEST WASHINGTON',
        'CENTRAL CALIFORNIA DISTRICT' : 'SOUTH & CENTRAL DISTRICT CALIFORNIA',
        'FLORIDA CENTRAL DISTRICT' : 'CENTRAL FLORIDA',
        'FLORIDA DISTRICTS' : 'CENTRAL FLORIDA',
        'GEORGIA DISTRICT' : 'SOUTH GEORGIA',
        'IMPORTS THROUGH ALEXANDRIA BAY (NY)' : 'ALEXANDRIA BAY (NY)',
        'IMPORTS THROUGH ATLANTA' : 'ATLANTA',
        'IMPORTS THROUGH BALTIMORE' : 'BALTIMORE',
        'IMPORTS THROUGH BLAINE (WASHINGTON)' : 'BLAINE (WASHINGTON)',
        'IMPORTS THROUGH BOSTON/LOGAN AIRPORT' : 'BOSTON/LOGAN AIRPORT',
        'IMPORTS THROUGH BROOKLYN (NYC)' : 'BROOKLYN (NYC)',
        'IMPORTS THROUGH BUFFALO' : 'BUFFALO',
        'IMPORTS THROUGH CALAIS (MAINE)' : 'CALAIS (MAINE)',
        'IMPORTS THROUGH CAMDEN' : 'CAMDEN',
        'IMPORTS THROUGH CHAMPLAIN (NEW YORK)' : 'CHAMPLAIN (NEW YORK)',
        'IMPORTS THROUGH CHARLESTOWN' : 'CHARLESTOWN',
        'IMPORTS THROUGH CHICAGO' : 'CHICAGO',
        'IMPORTS THROUGH DALLAS-FORT WORTH' : 'DALLAS-FORT WORTH',
        'IMPORTS THROUGH DERBY LINE (VERMONT)' : 'DERBY LINE (VERMONT)',
        'IMPORTS THROUGH DETROIT' : 'DETROIT',
        'IMPORTS THROUGH DULLES (VIRGINIA)' : 'DULLES (VIRGINIA)',
        'IMPORTS THROUGH ELIZABETH (NEW JERSEY)' : 'ELIZABETH (NEW JERSEY)',
        'IMPORTS THROUGH HONOLULU (HAWAII)' : 'HONOLULU (HAWAII)',
        'IMPORTS THROUGH HOULTON (MAINE)' : 'HOULTON (MAINE)',
        'IMPORTS THROUGH HOUSTON' : 'HOUSTON',
        'IMPORTS THROUGH HUNTSVILLE (ALABAMA)' : 'HUNTSVILLE (ALABAMA)',
        'IMPORTS THROUGH JACKMAN (MAINE)' : 'JACKMAN (MAINE)',
        'IMPORTS THROUGH JACKSONVILLE (FLORIDA)' : 'CENTRAL & NORTH FLORIDA',
        'IMPORTS THROUGH LOS ANGELES AIRPORT' : 'LOS ANGELES INTERNATIONAL AIRPORT',
        'IMPORTS THROUGH LOS ANGELES-LONG BEACH' : 'LOS ANGELES-LONG BEACH',
        'IMPORTS THROUGH MIAMI AIRPORT' : 'MIAMI INTERNATIONAL AIRPORT',
        'IMPORTS THROUGH NEW YORK JFK AIRPORT' : 'NEW YORK JFK AIRPORT',
        'IMPORTS THROUGH NEWARK AIRPORT' : 'NEWARK AIRPORT',
        'IMPORTS THROUGH NEWARK-ELIZABETH (NJ)' : 'NEWARK-ELIZABETH (NJ)',
        'IMPORTS THROUGH NORFOLK' : 'NORFOLK',
        'IMPORTS THROUGH OAKLAND' : 'OAKLAND',
        'IMPORTS THROUGH ORLANDO' : 'ORLANDO',
        'IMPORTS THROUGH OROVILLE (WASHINGTON)' : 'OROVILLE (WASHINGTON)',
        'IMPORTS THROUGH PEMBINA (NORTH DAKOTA)' : 'PEMBINA (NORTH DAKOTA)',
        'IMPORTS THROUGH PHARR (TEXAS)' : 'TEXAS',
        'IMPORTS THROUGH PHILADELPHIA' : 'PHILADELPHIA',
        'IMPORTS THROUGH PHILADELPHIA-CAMDEN' : 'PHILADELPHIA',
        'IMPORTS THROUGH PORT HUENEME (CALIF)' : 'PORT HUENEME (CALIF)',
        'IMPORTS THROUGH PORT HURON (MICHIGAN)' : 'PORT HURON (MICHIGAN)',
        'IMPORTS THROUGH PORTAL (NORTH DAKOTA)' : 'PORTAL (NORTH DAKOTA)',
        'IMPORTS THROUGH PORTLAND (OREGON)' : 'PORTLAND (OREGON)',
        'IMPORTS THROUGH ROMULUS (MICH)' : 'ROMULUS (MICH)',
        'IMPORTS THROUGH SAN DIEGO' : 'SAN DIEGO',
        'IMPORTS THROUGH SAN FRANCISCO' : 'SAN FRANCISCO',
        'IMPORTS THROUGH SAN JUAN (PUERTO RICO)' : 'SAN JUAN (PUERTO RICO)',
        'IMPORTS THROUGH SAVANNAH' : 'SAVANNAH',
        'IMPORTS THROUGH SEATTLE-TACOMA' : 'SEATTLE-TACOMA',
        'IMPORTS THROUGH SOUTH FLORIDA/TAMPA' : 'SOUTH FLORIDA',
        'IMPORTS THROUGH ST. LOUIS' : 'ST. LOUIS',
        'IMPORTS THROUGH SUMAS (WASHINGTON)' : 'SUMAS (WASHINGTON)',
        'IMPORTS THROUGH SWEETGRASS (MONTANA)' : 'SWEETGRASS (MONTANA)',
        'IMPORTS THROUGH WILMINGTON (DELAWARE)' : 'WILMINGTON (DELAWARE)',
        'MEXICO CROSSINGS THROUGH BROWNSVILLE, TX' : 'BROWNSVILLE (TX)',
        'MEXICO CROSSINGS THROUGH CALEXICO' : 'CALEXICO',
        'MEXICO CROSSINGS THROUGH LAREDO, TX' : 'LAREDO (TX)',
        'MEXICO CROSSINGS THROUGH OTAY MESA' : 'OTAY MESA',
        'MEXICO CROSSINGS THROUGH PHARR, TX' : 'PHARR (TX)',
        'MEXICO CROSSINGS THROUGH PROGRESO, TX' : 'PROGRESO (TX)',
        'MEXICO CROSSINGS THROUGH SAN LUIS' : 'SAN LUIS',
        'MEXICO CROSSINGS THRU NOGALES' : 'NOGALES',
        'MEXICO CROSSINGS THRU TEXAS' : 'TEXAS',
        'MICHIGAN DISTRICT' : 'MICHIGAN',
        'NEW JERSEY DISTRICT' : 'NEW JERSEY',
        'NORTH CAROLINA DISTRICT' : 'NORTH CAROLINA',
        'ORANGE-SAN DIEGO-COACHELLA DISTRICT' : 'ORANGE-SAN DIEGO COUNTIES & COACHELLA DISTRICT, CALIFORNIA',
        'OREGON AND WASHINGTON' : 'OREGON AND WASHINGTON',
        'OREGON COAST DISTRICT' : 'OREGON COAST',
        'OREGON DISTRICT' : 'OREGON',
        'OXNARD DISTRICT' : 'OXNARD DISTRICT CALIFORNIA',
        'SALINAS-WATSONVILLE' : 'SALINAS-WATSONVILLE CALIFORNIA',
        'SAN FRANCISCO/OAKLAND' : 'SAN FRANCISCO/OAKLAND',
        'SAN JOAQUIN VALLEY DISTRICT' : 'CENTRAL AND SOUTHERN SAN JOAQUIN VALLEY CALIFORNIA',
        'SANTA MARIA' : 'SANTA MARIA',
        'SOUTH GEORGIA' : 'SOUTH GEORGIA',
        'SOUTH NEW JERSEY' : 'SOUTH NEW JERSEY',
        'SOUTHERN CALIFORNIA DISTRICT' : 'SOUTHERN CALIFORNIA',
        'WASHINGTON COAST DISTRICT' : 'WASHINGTON COAST',
        'WASHINGTON DISTRICT' : 'WASHINGTON'
    }
    return switcher.get(row, "nan")


# %%
def label_trade_countries(row):
    switcher = {
        'ANDORRA' : 'AD',
        'UNITED ARAB EMIRATES' : 'AE',
        'AFGHANISTAN' : 'AF',
        'ANTIGUA AND BARBUDA' : 'AG',
        'ANGUILLA' : 'AI',
        'ALBANIA' : 'AL',
        'ARMENIA' : 'AM',
        'ANGOLA' : 'AO',
        'ANTARCTICA' : 'AQ',
        'ARGENTINA' : 'AR',
        'AMERICAN SAMOA' : 'AS',
        'AUSTRIA' : 'AT',
        'AUSTRALIA' : 'AU',
        'ARUBA' : 'AW',
        'ÅLAND ISLANDS' : 'AX',
        'AZERBAIJAN' : 'AZ',
        'BOSNIA AND HERZEGOVINA' : 'BA',
        'BARBADOS' : 'BB',
        'BANGLADESH' : 'BD',
        'BELGIUM' : 'BE',
        'BURKINA FASO' : 'BF',
        'BULGARIA' : 'BG',
        'BAHRAIN' : 'BH',
        'BURUNDI' : 'BI',
        'BENIN' : 'BJ',
        'SAINT BARTHÉLEMY' : 'BL',
        'BERMUDA' : 'BM',
        'BRUNEI DARUSSALAM' : 'BN',
        'BOLIVIA (PLURINATIONAL STATE OF)' : 'BO',
        'BONAIRE, SINT EUSTATIUS AND SABA' : 'BQ',
        'BRAZIL' : 'BR',
        'BAHAMAS' : 'BS',
        'BHUTAN' : 'BT',
        'BOUVET ISLAND' : 'BV',
        'BOTSWANA' : 'BW',
        'BELARUS' : 'BY',
        'BELIZE' : 'BZ',
        'CANADA' : 'CA',
        'COCOS (KEELING) ISLANDS' : 'CC',
        'CONGO, DEMOCRATIC REPUBLIC OF THE' : 'CD',
        'CENTRAL AFRICAN REPUBLIC' : 'CF',
        'CONGO' : 'CG',
        'SWITZERLAND' : 'CH',
        'CÔTE D''IVOIRE' : 'CI',
        'COOK ISLANDS' : 'CK',
        'CHILE' : 'CL',
        'CAMEROON' : 'CM',
        'CHINA' : 'CN',
        'COLOMBIA' : 'CO',
        'COSTA RICA' : 'CR',
        'CUBA' : 'CU',
        'CABO VERDE' : 'CV',
        'CURAÇAO' : 'CW',
        'CHRISTMAS ISLAND' : 'CX',
        'CYPRUS' : 'CY',
        'CZECHIA' : 'CZ',
        'GERMANY' : 'DE',
        'DJIBOUTI' : 'DJ',
        'DENMARK' : 'DK',
        'DOMINICA' : 'DM',
        'DOMINICAN REPUBLIC' : 'DO',
        'ALGERIA' : 'DZ',
        'ECUADOR' : 'EC',
        'ESTONIA' : 'EE',
        'EGYPT' : 'EG',
        'WESTERN SAHARA' : 'EH',
        'ERITREA' : 'ER',
        'SPAIN' : 'ES',
        'ETHIOPIA' : 'ET',
        'FINLAND' : 'FI',
        'FIJI' : 'FJ',
        'FALKLAND ISLANDS (MALVINAS)' : 'FK',
        'MICRONESIA (FEDERATED STATES OF)' : 'FM',
        'FAROE ISLANDS' : 'FO',
        'FRANCE' : 'FR',
        'GABON' : 'GA',
        'UNITED KINGDOM' : 'GB',
        'GRENADA' : 'GD',
        'GEORGIA' : 'GE',
        'FRENCH GUIANA' : 'GF',
        'GUERNSEY' : 'GG',
        'GHANA' : 'GH',
        'GIBRALTAR' : 'GI',
        'GREENLAND' : 'GL',
        'GAMBIA' : 'GM',
        'GUINEA' : 'GN',
        'GUADELOUPE' : 'GP',
        'EQUATORIAL GUINEA' : 'GQ',
        'GREECE' : 'GR',
        'SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS' : 'GS',
        'GUATEMALA' : 'GT',
        'GUAM' : 'GU',
        'GUINEA-BISSAU' : 'GW',
        'GUYANA' : 'GY',
        'HONG KONG' : 'HK',
        'HEARD ISLAND AND MCDONALD ISLANDS' : 'HM',
        'HONDURAS' : 'HN',
        'CROATIA' : 'HR',
        'HAITI' : 'HT',
        'HUNGARY' : 'HU',
        'INDONESIA' : 'ID',
        'IRELAND' : 'IE',
        'ISRAEL' : 'IL',
        'ISLE OF MAN' : 'IM',
        'INDIA' : 'IN',
        'BRITISH INDIAN OCEAN TERRITORY' : 'IO',
        'IRAQ' : 'IQ',
        'IRAN (ISLAMIC REPUBLIC OF)' : 'IR',
        'ICELAND' : 'IS',
        'ITALY' : 'IT',
        'JERSEY' : 'JE',
        'JAMAICA' : 'JM',
        'JORDAN' : 'JO',
        'JAPAN' : 'JP',
        'KENYA' : 'KE',
        'KYRGYZSTAN' : 'KG',
        'CAMBODIA' : 'KH',
        'KIRIBATI' : 'KI',
        'COMOROS' : 'KM',
        'SAINT KITTS AND NEVIS' : 'KN',
        'KOREA (DEMOCRATIC PEOPLES REPUBLIC OF)' : 'KP',
        'KOREA, REPUBLIC OF' : 'KR',
        'KUWAIT' : 'KW',
        'CAYMAN ISLANDS' : 'KY',
        'KAZAKHSTAN' : 'KZ',
        'LAO PEOPLES DEMOCRATIC REPUBLIC' : 'LA',
        'LEBANON' : 'LB',
        'SAINT LUCIA' : 'LC',
        'LIECHTENSTEIN' : 'LI',
        'SRI LANKA' : 'LK',
        'LIBERIA' : 'LR',
        'LESOTHO' : 'LS',
        'LITHUANIA' : 'LT',
        'LUXEMBOURG' : 'LU',
        'LATVIA' : 'LV',
        'LIBYA' : 'LY',
        'MOROCCO' : 'MA',
        'MONACO' : 'MC',
        'MOLDOVA, REPUBLIC OF' : 'MD',
        'MONTENEGRO' : 'ME',
        'SAINT MARTIN (FRENCH PART)' : 'MF',
        'MADAGASCAR' : 'MG',
        'MARSHALL ISLANDS' : 'MH',
        'NORTH MACEDONIA' : 'MK',
        'MALI' : 'ML',
        'MYANMAR' : 'MM',
        'MONGOLIA' : 'MN',
        'MACAO' : 'MO',
        'NORTHERN MARIANA ISLANDS' : 'MP',
        'MARTINIQUE' : 'MQ',
        'MAURITANIA' : 'MR',
        'MONTSERRAT' : 'MS',
        'MALTA' : 'MT',
        'MAURITIUS' : 'MU',
        'MALDIVES' : 'MV',
        'MALAWI' : 'MW',
        'MEXICO' : 'MX',
        'MALAYSIA' : 'MY',
        'MOZAMBIQUE' : 'MZ',
        'NAMIBIA' : 'NA',
        'NEW CALEDONIA' : 'NC',
        'NIGER' : 'NE',
        'NORFOLK ISLAND' : 'NF',
        'NIGERIA' : 'NG',
        'NICARAGUA' : 'NI',
        'NETHERLANDS' : 'NL',
        'NORWAY' : 'NO',
        'NEPAL' : 'NP',
        'NAURU' : 'NR',
        'NIUE' : 'NU',
        'NEW ZEALAND' : 'NZ',
        'OMAN' : 'OM',
        'PANAMA' : 'PA',
        'PERU' : 'PE',
        'FRENCH POLYNESIA' : 'PF',
        'PAPUA NEW GUINEA' : 'PG',
        'PHILIPPINES' : 'PH',
        'PAKISTAN' : 'PK',
        'POLAND' : 'PL',
        'SAINT PIERRE AND MIQUELON' : 'PM',
        'PITCAIRN' : 'PN',
        'PUERTO RICO' : 'PR',
        'PALESTINE, STATE OF' : 'PS',
        'PORTUGAL' : 'PT',
        'PALAU' : 'PW',
        'PARAGUAY' : 'PY',
        'QATAR' : 'QA',
        'RÉUNION' : 'RE',
        'ROMANIA' : 'RO',
        'SERBIA' : 'RS',
        'RUSSIAN FEDERATION' : 'RU',
        'RWANDA' : 'RW',
        'SAUDI ARABIA' : 'SA',
        'SOLOMON ISLANDS' : 'SB',
        'SEYCHELLES' : 'SC',
        'SUDAN' : 'SD',
        'SWEDEN' : 'SE',
        'SINGAPORE' : 'SG',
        'SAINT HELENA, ASCENSION AND TRISTAN DA CUNHA' : 'SH',
        'SLOVENIA' : 'SI',
        'SVALBARD AND JAN MAYEN' : 'SJ',
        'SLOVAKIA' : 'SK',
        'SIERRA LEONE' : 'SL',
        'SAN MARINO' : 'SM',
        'SENEGAL' : 'SN',
        'SOMALIA' : 'SO',
        'SURINAME' : 'SR',
        'SOUTH SUDAN' : 'SS',
        'SAO TOME AND PRINCIPE' : 'ST',
        'EL SALVADOR' : 'SV',
        'SINT MAARTEN (DUTCH PART)' : 'SX',
        'SYRIAN ARAB REPUBLIC' : 'SY',
        'ESWATINI' : 'SZ',
        'TURKS AND CAICOS ISLANDS' : 'TC',
        'CHAD' : 'TD',
        'FRENCH SOUTHERN TERRITORIES' : 'TF',
        'TOGO' : 'TG',
        'THAILAND' : 'TH',
        'TAJIKISTAN' : 'TJ',
        'TOKELAU' : 'TK',
        'TIMOR-LESTE' : 'TL',
        'TURKMENISTAN' : 'TM',
        'TUNISIA' : 'TN',
        'TONGA' : 'TO',
        'TURKEY' : 'TR',
        'TRINIDAD AND TOBAGO' : 'TT',
        'TUVALU' : 'TV',
        'TAIWAN, PROVINCE OF CHINA' : 'TW',
        'TANZANIA, UNITED REPUBLIC OF' : 'TZ',
        'UKRAINE' : 'UA',
        'UGANDA' : 'UG',
        'UNITED STATES MINOR OUTLYING ISLANDS' : 'UM',
        'UNITED STATES OF AMERICA' : 'US',
        'URUGUAY' : 'UY',
        'UZBEKISTAN' : 'UZ',
        'HOLY SEE' : 'VA',
        'SAINT VINCENT AND THE GRENADINES' : 'VC',
        'VENEZUELA (BOLIVARIAN REPUBLIC OF)' : 'VE',
        'VIRGIN ISLANDS (BRITISH)' : 'VG',
        'VIRGIN ISLANDS (U.S.)' : 'VI',
        'VIET NAM' : 'VN',
        'VANUATU' : 'VU',
        'WALLIS AND FUTUNA' : 'WF',
        'SAMOA' : 'WS',
        'YEMEN' : 'YE',
        'MAYOTTE' : 'YT',
        'SOUTHAFRICA' : 'ZA',
        'ZAMBIA' : 'ZM',
        'ZIMBABWE' : 'ZW',
        'SOUTH KOREA' : 'KR'
    }
    return switcher.get(row, row)


# %%
def load_volumes(pcrop,pcrop_abb,ndays):
    
    # Importing libraries 
    import pandas as pd
    from datetime import date, datetime, timedelta
    import numpy as np
    import pyodbc

    # Setting dates
    # Date from
    fdate = datetime.today() - timedelta(days=ndays)
    fday = fdate.strftime('%d')
    fmonth = fdate.strftime('%m')
    fyear = fdate.strftime('%Y')

    # Date to : current date data to collect updated information
    tday = date.today().strftime('%d')
    tmonth = date.today().strftime('%m')
    tyear = date.today().strftime('%Y')

    # URL for accessing quantities
    USquantity =f"https://www.marketnews.usda.gov/mnp/fv-report-top-filters?&commAbr={pcrop_abb}&varName=&locAbr=&repType=movementDaily&navType=byComm&locName=&navClass=&navClass=&type=movement&dr=1&volume=&commName={pcrop}&portal=fv&region=&repDate={fmonth}%2F{fday}%2F{fyear}&endDate={tmonth}%2F{tday}%2F{tyear}&format=excel&rebuild=false"
        
    
    # Assign the table data in html format to a Pandas dataframe
    table =  pd.read_html(USquantity,header=0,parse_dates=['Date'])[0]

    # Read the table in new dataframe with the main info
    volumes = table[['Commodity Name',
                    'Origin Name',
                    'Type',
                    'Package',
                    'Date',
                    'District',
                    '10000lb units',
                    'Trans Mode',
                    'Season',
                    'Import/Export']]

    ########## Cleaning data ###########

    # Delete rows if no volume available
    volumes.dropna(axis='index',how='all',subset=['10000lb units'],inplace=True)

    # New Category field based on type
    volumes['Category'] = 'std ' + volumes['Type'].str.lower().str.strip()
    volumes['Category'].fillna('std', inplace = True)
    volumes["Import/Export"].fillna('internal', inplace = True)

    # Convert 10K lb units to KG
    volumes['Volume'] = volumes['10000lb units'] * 4535.9237

    # Campaign dates
    campaign_dates = volumes.groupby('Season')['Date'].agg('min').reset_index()
    campaign_dates.columns = ['Season','First Date']
    volumes = pd.merge(left=volumes, right=campaign_dates, how='left', left_on='Season', right_on='Season')
    volumes['Week_num_campaign'] = np.ceil((volumes['Date'] - volumes['First Date']).dt.days.astype('int16')/7).astype(int)
    volumes['Week_num_campaign'] = volumes['Week_num_campaign'].apply(lambda x: x + 1 if x == 0 else x)

    # Informative fields
    volumes['Crop']=pcrop
    volumes['Country']='US'
    volumes['Measure']='KG'

    # Same package naming as prices
    volumes['Package'].replace({'CTNS 8 18-OZ CNTRS W/LID' : 'cartons 8 18-oz containers with lids', 'FLTS 12 6-OZ CUPS W/LIDS' : 'flats 12 6-oz cups with lids'}, inplace=True)

    # Formatting
    volumes['Crop'] = volumes['Crop'].astype('str')
    volumes['Country'] = volumes['Country'].astype('str')
    volumes['District'] = volumes['District'].astype('str')
    volumes["Import/Export"] = volumes["Import/Export"].astype('str')
    volumes['Origin Name'] = volumes['Origin Name'].astype('str')
    volumes['Category'] = volumes['Category'].astype('str')
    volumes['Package'] = volumes['Package'].astype('str')
    volumes['Trans Mode'] = volumes['Trans Mode'].astype('str')

    # Naming of Region and Trade Countries
    volumes['District'] = volumes['District'].apply(lambda x : label_region_volumes(x))
    volumes['Origin Name'] = volumes['Origin Name'].apply(lambda x : label_trade_countries(x))

    # Taking only relevant columns
    volumes = volumes[['Crop', 'Country', 'District', 'Import/Export', 'Origin Name', 'Category', 'Package', 'Trans Mode', 'Season','Week_num_campaign','Date','Measure','Volume']]
    volumes.rename(columns={'Crop' : 'Product', 'Country' : 'Country', 'District' : 'Region', 'Import/Export' : 'Trade_Type', 'Origin Name' : 'Trade_Country', 'Category' : 'Category', 'Package' : 'Package', 'Trans Mode' : 'Transport', 'Season' : 'Campaign', 'Week_num_campaign' : 'Campaign_wk', 'Date' : 'Date_volume', 'Measure' : 'Measure','Volume' : 'Volume'},errors="raise",inplace=True)

    #################### SQL statements ######################

    import pyodbc
    from datetime import datetime, timedelta

    connStr = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes')
    cursor = connStr.cursor()

    # Delete all data with volumes dates greater than the ndays parameter last days from today
    #N = 60 #3864
    #rep_date = datetime.now().date() - timedelta(days=N)
    qry_delete = f"DELETE FROM [Prices].[dbo].[volumes] where cast([Country] as nvarchar) = cast('US' as nvarchar) and cast([Product] as nvarchar) = cast('{pcrop}' as nvarchar) and Date_volume > '{fdate}'"
    cursor.execute(qry_delete)

    # Load all data with volumes dates greater than the ndays parameter last days from today
    upd = 0

    try:
        for index,row in volumes.iterrows():
            if row['Date_volume'] > fdate: # Python volumes line date must be greater than the max date in SQL table
                cursor.execute("INSERT INTO dbo.volumes([Product],[Country],[Region],[Trade_Type],[Trade_Country],[Category],[Package],[Transport],[Campaign],[Campaign_wk],[Date_volume],[Measure],[Volume],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Type'],row['Trade_Country'],row['Category'],row['Package'],row['Transport'],row['Campaign'],row['Campaign_wk'],row['Date_volume'],row['Measure'],row['Volume'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                connStr.commit()
                upd += 1
    except TypeError: # If there volume is null no posibility to compare operands
        for index,row in volumes.iterrows(): # When there are no volumes in SQL
            cursor.execute("INSERT INTO dbo.volumes([Product],[Country],[Region],[Trade_Type],[Trade_Country],[Category],[Package],[Transport],[Campaign],[Campaign_wk],[Date_volume],[Measure],[Volume],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Region'],row['Trade_Type'],row['Trade_Country'],row['Category'],row['Package'],row['Transport'],row['Campaign'],row['Campaign_wk'],row['Date_volume'],row['Measure'],row['Volume'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1
    print(upd," new volumes added")

    cursor.close()
    connStr.close()


