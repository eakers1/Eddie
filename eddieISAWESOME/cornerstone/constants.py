

depth_dict = {
    "kraken": 500,
    "coinbaseexchange": 2000,
    "okx":3000,
    "okx_perp":3000
}

#fetch from Talos API
fees = {
    "kraken": 0.001,
    "coinbaseexchange": 0.000045,
    "okx":0.0004,
    "okx_perp": 0.00025
}


exchange_credentials = {}
# all exchanges we have access to 
full_exchange_list = [ "coinbaseexchange", "okx","kraken", "okx_perp","binance","bybit"]

# calculation for fx rate, this one can be edited to choose the exchange to use
fx_exchange_list = ["kraken", "coinbaseexchange",'okx']

USDT_EXCHANGES = ["okx","okx_perp","binance","bybit"]

USD_EXCHANGES = ["kraken","coinbaseexchange"]