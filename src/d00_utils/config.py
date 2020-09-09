ndays = 90
db_con = 'DRIVER={ODBC Driver 13 for SQL Server};SERVER=bipro02\\adminbi;DATABASE=Prices;Trusted_Connection=yes'
crop_list_arima = [['BLUEBERRIES','US','MX','std','ARIMA','','','USD','KG',None],\
    ['RASPBERRIES','US','MX','std','ARIMA','','','USD','KG',None],\
    ['STRAWBERRIES','US','MX','med','ARIMA','','','USD','KG',None],\
    ['BLUEBERRIES','ES','ES','std','ARIMA','','','EUR','KG',None],\
    ['RASPBERRIES','ES','ES','std','ARIMA','','','EUR','KG',None],\
    ['STRAWBERRIES','ES','ES','std','ARIMA','','','EUR','KG',None]]
crop_list_sarima = [['BLUEBERRIES','US','MX','std','SARIMA','','','USD','KG',None],\
    ['RASPBERRIES','US','MX','std','SARIMA','','','USD','KG',None],\
    ['STRAWBERRIES','US','MX','med','SARIMA','','','USD','KG',None],\
    ['BLUEBERRIES','ES','ES','std','SARIMA','','','EUR','KG',None],\
    ['RASPBERRIES','ES','ES','std','SARIMA','','','EUR','KG',None],\
    ['STRAWBERRIES','ES','ES','std','SARIMA','','','EUR','KG',None]]
crop_list_sarimax = [['BLUEBERRIES','US','MX','std','SARIMAX','','','USD','KG',None],\
    ['RASPBERRIES','US','MX','std','SARIMAX','','','USD','KG',None],\
    ['STRAWBERRIES','US','MX','med','SARIMAX','','','USD','KG',None]]
crop_list_usa = [['BLUEBERRIES','BLUBY'],\
    ['RASPBERRIES','RASP'],\
    ['STRAWBERRIES','STRBY']]