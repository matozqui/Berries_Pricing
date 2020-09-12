path_price_es_blue = '../../data/01_raw/prices/ArandanoPreciosAgricultor.xls'
path_price_es_rasp = '../../data/01_raw/prices/FrambuesaPreciosAgricultor.xls'
path_price_es_strw = '../../data/01_raw/prices/FresaPreciosAgricultor.xls'
es_labour_file1 = '../../data/01_raw/labor/indicesysalariosagrariosenero1985-diciembre2017_tcm30-539891.xlsx'
es_labour_file2 = '../../data/01_raw/labor/indicesysalariosagrariosenero2018-marzo2020_tcm30-541202.xlsx'

def usda_web_prices(crop_abb,crop,fmonth,fday,fyear,tmonth,tday,tyear):
    return(f"https://www.marketnews.usda.gov/mnp/fv-report-top-filters?&commAbr={crop_abb}&varName=&locAbr=&repType=shipPriceDaily&navType=byComm&locName=&navClass=&type=shipPrice&dr=1&volume=&commName={crop}&navClass,=&portal=fv&region=&repDate={fmonth}%2F{fday}%2F{fyear}&endDate={tmonth}%2F{tday}%2F{tyear}&format=excel&rebuild=false") 

def usda_web_quantities(crop_abb,crop,fmonth,fday,fyear,tmonth,tday,tyear):
    return(f"https://www.marketnews.usda.gov/mnp/fv-report-top-filters?&commAbr={crop_abb}&varName=&locAbr=&repType=movementDaily&navType=byComm&locName=&navClass=&navClass=&type=movement&dr=1&volume=&commName={crop}&portal=fv&region=&repDate={fmonth}%2F{fday}%2F{fyear}&endDate={tmonth}%2F{tday}%2F{tyear}&format=excel&rebuild=false")