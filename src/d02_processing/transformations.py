# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def normalize(df):

    ##  Function to normalize continous variables into 0 to 1 range ##

    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    norm_inst = MinMaxScaler(feature_range=(0.01, 0.99))
    norm = norm_inst.fit_transform(df)

    return norm_inst, norm


# %%
def denormalize(norm_inst,norm):

    ##  Function to denormalize continous variables into the real continous variable ##

    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    descaling_input = norm_inst.inverse_transform(norm)
    
    descaling_input =pd.DataFrame(descaling_input)
    descaling_input.index = norm.index
    descaling_input.columns = norm.columns.values

    return descaling_input


# %%
def label_trade_countries(row):

    ## Function to convert the desctiption of a country in capital letters into country ISO3166-1 alpha-2   ##

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
def label_origin(row):

    ## Function to convert the desctiption of origin from USDA website into country ISO3166-1 alpha-2   ##

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

    ## Function to convert the desctiption of region from USDA website into a standard region description   ##

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
def label_region_volumes(row):

    ## Function to convert the desctiption of volumes region from USDA website into a standard region description   ##

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
def get_prices_volumes(ctry,crop,trade_ctry,ctgr):

    #   Function to combine prices and volumes in one dataframe #

    import sys
    sys.path.insert(0, '../../src')
    sys.path.append('../../src/d01_data')
    sys.path.append('../../src/d02_processing')
    sys.path.append('../../src/d03_modelling')

    import extractions as extract
    import transformations as transf
    import training as train
    import import_data as imp
    import inference as inf
    import datetime

    df_prices = extract.get_prices(crop, ctry, trade_ctry, ctgr)
    df_prices.set_index('Date_price',inplace=True)

    df_volumes = extract.get_volumes(crop, ctry, trade_ctry)

    df_volumes_prices = df_volumes.join(df_prices,how='outer')
    df_volumes_prices.Volume.fillna(value=0, inplace=True)
    df_volumes_prices.fillna(method='pad', inplace=True)

    # First date with price
    first_date = max(df_volumes_prices[(df_volumes_prices.Price).isnull()].index) + datetime.timedelta(days=1)

    df_volumes_prices.drop(df_volumes_prices[df_volumes_prices.index < first_date].index, inplace=True)
    df_volumes_prices.reset_index(inplace=True)
    df_volumes_prices.rename(columns={'index' : 'Date_ref'},errors="raise",inplace=True)

    return df_volumes_prices


# %%
def load_volumes_prices_bbdd(df_volumes_prices):

    ##  Function to upload the combination of prices and volumes retrieved from different sources to SQL Server Database   #

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
    import stringing as st

    ctry = st.get_comma_values(df_volumes_prices.Country)
    crop = st.get_comma_values(df_volumes_prices.Product)
    trade_ctry = st.get_comma_values(df_volumes_prices.Trade_Country)
    ctgr = st.get_comma_values(df_volumes_prices.Category)

    connStr = pyodbc.connect(config.db_con)
    cursor = connStr.cursor()

    # Setting dates
    # Date from
    fdate = datetime.today() - timedelta(days=config.ndays)

    # Delete all data with price dates greater than the ndays parameter last days from today
# Delete all data with price dates greater than the ndays parameter last days from today
    qry_delete = f"DELETE FROM [Prices].[dbo].[volumes_prices] where [Country] = {ctry} and [Product] IN ({crop}) and [Trade_Country] = {trade_ctry} and [Category] = {ctgr} and Date_ref > '{fdate}'"
    cursor.execute(qry_delete)
    connStr.commit()

    # Load all data with price dates greater than the ndays global parameter last days from today
    upd = 0
    try:
        for index,row in df_volumes_prices.iterrows():
            if row['Date_ref'] > fdate: # Python price line date must be greater than the max date in SQL table
                cursor.execute("INSERT INTO [Prices].[dbo].[volumes_prices]([Product],[Country],[Trade_Country],[Category],[Campaign],[Campaign_wk],[Date_ref],[Currency],[Measure],[Price],[Volume],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Trade_Country'],row['Category'],row['Campaign'],'',row['Date_ref'],'USD','KG',row['Price'],row['Volume'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                connStr.commit()
                upd += 1
    except TypeError: # If there price is null no posibility to compare operands
        for index,row in df_volumes_prices.iterrows(): # When there are no prices in SQL
            cursor.execute("INSERT INTO [Prices].[dbo].[volumes_prices]([Product],[Country],[Trade_Country],[Category],[Campaign],[Campaign_wk],[Date_ref],[Currency],[Measure],[Price],[Volume],[Updated]) values (?,?,?,?,?,?,?,?,?,?,?,?)",row['Product'],row['Country'],row['Trade_Country'],row['Category'],row['Campaign'],'',row['Date_ref'],'USD','KG',row['Price'],row['Volume'],datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            connStr.commit()
            upd += 1
    cursor.close()
    connStr.close()
    print(upd," new prices added")


