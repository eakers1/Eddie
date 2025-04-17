import logging 
import ccxt
import pandas as pd
import numpy as np
import datetime as dt
import constants
### by using 24h volume and quantity, calculating the estimated time to trade
def get_estimated_time_to_trade(base_ccy,quote_ccy,exchange_list,full_exchanges,quantity,is_quantity,participating_rate):
    participating_rate = participating_rate/100
    
    exchanges = {key: full_exchanges[key] for key in exchange_list if key in full_exchanges}
    # coinbase does not have 24h quote volume, so in this case, we use the tob mid price to convert notional by using current tob to quantity
    daily_volume = []
    current_mid_price = []
    for exchange_name in exchanges: 
        if quote_ccy in ['USD','USDC'] and exchange_name in constants.USDT_EXCHANGES:
            temp_quote_ccy = 'USDT'
        elif quote_ccy in ['USDT','USDC'] and exchange_name in constants.USD_EXCHANGES:
            temp_quote_ccy = 'USD'
        else:
            temp_quote_ccy = quote_ccy
            
        try:
            if exchange_name == 'okx_perp': #okx perp has a special symbol format 
                symbol = base_ccy + '/' + temp_quote_ccy + ':USDT'
            else:
                symbol = base_ccy + '/' + temp_quote_ccy
            ticker_info = exchanges[exchange_name].fetchTicker(symbol)
            base_volume = ticker_info['baseVolume'] 
            mid_price = (ticker_info['bid'] + ticker_info['ask'])/2
            daily_volume.append(base_volume)
            current_mid_price.append(mid_price)
        except Exception as e:
            logging.warning('Error in getting daily volume for exchange %s: %s',exchange_name,e)
            
    if len(daily_volume) <= 0 or len(current_mid_price) <= 0:
        logging.warning(f'Not enough data for calculating estimated time to trade, number of exchange volume data is {len(daily_volume)}, and number of exchange mid price data is {len(current_mid_price)}')
        return 0
    else:   
        if is_quantity:
            return 24*quantity/(participating_rate*np.sum(daily_volume))
        else:
            return 24*quantity/(np.average(current_mid_price)*participating_rate*np.sum(daily_volume))



### fucntion ot calculate volatility premium based on total time to hedge
def get_volatility_premuim(base_ccy,quote_ccy,exchange_list,full_exchanges,quantity,is_quantity,participating_rate,estimated_delivery_time):
    estimated_time_to_trade = get_estimated_time_to_trade(base_ccy,quote_ccy,exchange_list,full_exchanges,quantity,is_quantity,participating_rate) 
    total_time = estimated_time_to_trade + estimated_delivery_time
    # for short term trading, we use the 1 minute data to predict the return

    if total_time <= 0.5:
        timeframe = '1m'
        end_date = dt.datetime.now(dt.timezone.utc)
        start_date = end_date - dt.timedelta(minutes=220)

        # Convert timestamps to milliseconds (required by CCXT)
        since = int(start_date.timestamp() * 1000)
        #until = int(end_date.timestamp() * 1000)
        
        for exchange in full_exchanges:
            if exchange in constants.USD_EXCHANGES:
                symbol = base_ccy + '/USD'
            else:
                symbol = base_ccy + '/USDT'
                
            try:
                ohlcv = full_exchanges[exchange].fetch_ohlcv(symbol, timeframe, since, limit=None, params={})
                return_series = data_processing(ohlcv)
                return estimated_time_to_trade,ewma_volatility_model(return_series,0.2,total_time) 
                #print(ohlcv)
            except: 
                logging.warning(f'Error in fetching ohlcv data for exchange {exchange}, trying next exchange')
                continue
        
        return estimated_time_to_trade, 0                            
            
        
    else:
        timeframe = '1h'
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(hours=12)

        # Convert timestamps to milliseconds (required by CCXT)
        since = int(start_date.timestamp() * 1000)
        #until = int(end_date.timestamp() * 1000)
        
        for exchange in full_exchanges:
            if exchange in constants.USD_EXCHANGES:
                symbol = base_ccy + '/USD'
            else:
                symbol = base_ccy + '/USDT'
                
            try:
                ohlcv = full_exchanges[exchange].fetch_ohlcv(symbol, timeframe, since, limit=None, params={})
                return_series = data_processing(ohlcv)
                return estimated_time_to_trade,ewma_volatility_model(return_series,0.2,total_time) 
                #print(ohlcv)
            except: 
                logging.warning(f'Error in fetching ohlcv data for exchange {exchange}, trying next exchange')
                continue
        
        return estimated_time_to_trade, 0    
            
### generating return list by ohlcv data            
def data_processing(ohlcv):
    # ohlcv: a list of open, high, low, close, volume data
    # output: a list of returns based on the close price
    price_df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    ### error handling 
    price_df['Close'] = price_df['Close'].replace(0,np.nan)
    price_df['Close'] = price_df['Close'].ffill()
    price_df['return'] = np.log(price_df['Close']/price_df['Close'].shift(1))
    price_df= price_df.dropna()
    return_series = pd.Series(price_df['return'][-200:])
    return return_series


### simple ewma model to predict volatility based on total time to hedge          
def ewma_volatility_model(return_series,alpha_config, total_time):
    # half_life_ms: half life of the ewma model 
    # output: a list of predicted volatilities based on the ewma model 
    squared_return_series = return_series.apply(lambda x: x**2)
    ewma_variance = squared_return_series.ewm(alpha = alpha_config).mean().iloc[-1]
    #print(ewma_variance)
    if ewma_variance == 0:
        logging.warning('The ewma variance is 0, please check the input data')
        return 0
    else:
        if total_time <=1/60: 
            return round(np.sqrt(ewma_variance)*10000,2)
        elif total_time > 1/60 and total_time <= 0.5:
            return round(np.sqrt(ewma_variance*total_time*60)*10000,2) #in bps term and round to 1
        else:
            return round(np.sqrt(ewma_variance*total_time)*10000,2)  # in bps term 


### instructions for hedging actions based on total quantity and total time to hedge 
def hedging_action(base_ccy,quote_ccy,exchanges,quantity,is_quantity,total_time): 
    exchange_output = ','.join(exchanges)
    symbol = base_ccy + '/' + quote_ccy
    time_multiplier = [1,1/2,1/12,1/60]
    quantity_slice = [round(num*quantity/total_time) for num in time_multiplier]
    print(f'Total time to hedge: {round(total_time,2)} hours')
    if is_quantity:
        print("Hedging actions:")
        print(f' {quantity_slice[0]} {base_ccy} per hour')
        print(f' {quantity_slice[1]} {base_ccy} per 30 mins')
        print(f' {quantity_slice[2]} {base_ccy} per 5 mins')
        print(f' {quantity_slice[3]} {base_ccy} per 1 min')
    else:
        print("Hedging actions:")
        print(f' ${quantity_slice[0]} {quote_ccy} per hour')
        print(f' ${quantity_slice[1]} {quote_ccy} per 30 mins')
        print(f' ${quantity_slice[2]} {quote_ccy} per 5 mins')
        print(f' ${quantity_slice[3]} {quote_ccy} per 1 min')
    
    
        