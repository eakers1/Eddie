import logging
import os
import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.backend_bases import MouseButton
import constants
### quote currency price calculation 
"""
two methods: 
- method 1: direct market price. for example: there is a USDT/USD order book in coinbaseexchange and kraken. 
- method 2: synthetic price calulcating from a symbol. for example: by using the price of BTC/USDT and BTC/USD ---> synthetic price of USDT/USD 
"""
def ccxt_ticker_mid_price(exchanges,base_ccy, quote_ccy, exchange_name):
    try:
        tob = exchanges[exchange_name].fetch_ticker(base_ccy + '/' + quote_ccy)
        return (tob['bid'] + tob['ask'])/2
    except Exception as e:
        logging.warning(f"Failed to fetch ticker for {base_ccy}/{quote_ccy} on {exchange_name}: {e}")


def sythetic_average_rate(exchanges,base_ccy, quote_ccy):
    sythetic_rate = []
    if quote_ccy == 'USDC':
        try: 
            kraken_usdc_mid = ccxt_ticker_mid_price(exchanges,base_ccy,quote_ccy ,'kraken')
            kraken_usd_mid = ccxt_ticker_mid_price(exchanges,base_ccy,'USD','kraken')
            okx_usdc_mid = ccxt_ticker_mid_price(exchanges,base_ccy,quote_ccy ,'okx')
            coinbaseexchange_usd_mid = ccxt_ticker_mid_price(exchanges,base_ccy,'USD' ,'coinbaseexchange')
            sythetic_rate.extend([kraken_usd_mid/kraken_usdc_mid, coinbaseexchange_usd_mid/okx_usdc_mid])
            return sythetic_rate
        except Exception as e:
            logging.warning(f"issue calculating USDC/USD rate by {base_ccy}")
            return sythetic_rate
    elif quote_ccy == 'USDT':
        try: 
            kraken_usdt_mid = ccxt_ticker_mid_price(exchanges,base_ccy,quote_ccy ,'kraken')
            kraken_usd_mid = ccxt_ticker_mid_price(exchanges,base_ccy,'USD','kraken')
            okx_usdt_mid = ccxt_ticker_mid_price(exchanges,base_ccy,quote_ccy ,'okx')
            coinbaseexchange_usdt_mid = ccxt_ticker_mid_price(exchanges,base_ccy,'USDT' ,'coinbaseexchange')
            coinbaseexchange_usd_mid = ccxt_ticker_mid_price(exchanges,base_ccy,'USD' ,'coinbaseexchange')
            sythetic_rate.extend([kraken_usd_mid/kraken_usdt_mid, coinbaseexchange_usd_mid/okx_usdt_mid,coinbaseexchange_usdt_mid/coinbaseexchange_usdt_mid])
            return sythetic_rate
        except Exception as e:
            logging.warning(f"issue calculating USDT/USD rate by {base_ccy}")
            return sythetic_rate
                    
# calculating stable coin price (USDC,USDT to USD) by two methods, considering we always provide USD price to clients 
# 2 methods here: direct market order book or sythetic market calculation 
def fx_rate(exchanges,method, usdc_parity):
    #print(base_ccy, quote_ccy, exchange_list, method)
   
    quote_ccy_list = ['USD','USDC','USDT']
    quote_ccy_dict = {}
    for quote_ccy in quote_ccy_list:
        if quote_ccy == 'USD' or (quote_ccy == 'USDC' and usdc_parity):
            quote_ccy_dict[quote_ccy] = 1.0
        else:
            ### if we want to use the direct market price
            if method == 'direct':
                if quote_ccy == 'USDC': # kraken and usdc_parity = False
                    quote_ccy_dict[quote_ccy] = ccxt_ticker_mid_price(exchanges,quote_ccy,'USD' ,'kraken')
                    
                elif quote_ccy == 'USDT': # kraken and coinbaseexchange 
                    kraken_mid = ccxt_ticker_mid_price(exchanges,quote_ccy,'USD' ,'kraken')
                    coinbaseexchange_mid = ccxt_ticker_mid_price(exchanges,quote_ccy,'USD' ,'coinbaseexchange')
                    quote_ccy_dict[quote_ccy] = (kraken_mid + coinbaseexchange_mid)/2
                        
            elif method =='synthetic':
                base_symbol_list = ['BTC','ETH']
                sythetic_usdc_rate = []
                
                for i in base_symbol_list:
                    sythetic_usdc_rate.extend(sythetic_average_rate(exchanges,i,quote_ccy))
                    
                if len(sythetic_usdc_rate) != 0:
                    quote_ccy_dict[quote_ccy] = np.mean(sythetic_usdc_rate)
                
            else:
                logging.warning(f"Unknown method: {method}")
    
    rounded_quote_ccy_dict = {key: round(value,4) for key, value in quote_ccy_dict.items()}
    return rounded_quote_ccy_dict

### calculating the fair price by an aggreagted order book 
#----------------------------------------------------------------------------------------------------------------#
def aggreagted_order_book(depth_dict,fees,exchanges,base_ccy,quote_ccy,enable_fx_rate,method,fx_exchanges,usdc_parity):
    quote_currency_rate = {}
    if enable_fx_rate:
        quote_currency_rate = fx_rate(fx_exchanges,method,usdc_parity)
        print(f"current quote fx rate is: {quote_currency_rate}")
    else:
        pass 
    
    bid_order_book = pd.DataFrame()
    ask_order_book = pd.DataFrame()
    
    base_ccy = base_ccy.upper()
    quote_ccy = quote_ccy.upper()
    for exchange_name in exchanges:
        ### we always calculate the best liquidity book in each exchange 
        bid_temp = {}
        ask_temp = {}
        if quote_ccy in ['USD','USDC'] and exchange_name in constants.USDT_EXCHANGES:
            temp_quote_ccy = 'USDT'
        elif quote_ccy in ['USDT','USDC'] and exchange_name in constants.USD_EXCHANGES:
            temp_quote_ccy = 'USD'
        else:
            temp_quote_ccy = quote_ccy
        
        try:
            if exchange_name == 'okx_perp':  # special handling of okx_perp symbol name in ccxt 
                symbol = base_ccy + '/' + temp_quote_ccy + ':USDT'
            else:
                symbol = base_ccy + '/' + temp_quote_ccy
            depth = depth_dict[exchange_name]
            order_book = exchanges[exchange_name].fetch_order_book(symbol,depth)
            bid_temp = pd.DataFrame(np.array(order_book['bids'][:depth])[:,:2]).rename(columns = {0: 'price', 1: 'quantity'})
            ask_temp = pd.DataFrame(np.array(order_book['asks'][:depth])[:,:2]).rename(columns = {0: 'price', 1: 'quantity'})
            bid_temp['price'] = bid_temp['price']* quote_currency_rate.get(temp_quote_ccy,1.0)
            ask_temp['price'] = ask_temp['price']* quote_currency_rate.get(temp_quote_ccy,1.0)
            bid_temp['price'] = bid_temp['price']* (1-fees.get(exchange_name,0.0))
            ask_temp['price'] = ask_temp['price']* (1+fees.get(exchange_name,0.0))
            
            bid_order_book = pd.concat([bid_order_book,bid_temp],axis=0)
            ask_order_book = pd.concat([ask_order_book,ask_temp],axis=0)
            
        except Exception as e: 
            logging.warning(f"Failed to fetch ticker or order book info for {symbol} on {exchange_name}: {e}")    
    
    
    if len(bid_order_book) != 0 and len(ask_order_book) != 0:
        return bid_order_book, ask_order_book, quote_currency_rate
    else:
        logging.warning(f"Failed to fetch any order book info for {base_ccy}/{quote_ccy}")
        return None, None, quote_currency_rate
        
### to get a precise vwap price by rather using next row price, but a volume weighted price based on the extra quantity for the next level   
def get_precise_vwap_price(orderbook,quantity,is_quantity,index):
         
    if index == 0:
        return orderbook.loc[index,'vwap']
    elif index >= len(orderbook): ### should not have case > 
        print("Not enough liquidity for calculating fair price")
        return None
    else:
        first_row_vwap = orderbook.loc[index-1,'vwap']
        first_row_size = orderbook.loc[index-1,'cumulative_quantity']
        second_row_price = orderbook.loc[index,'price']
        if is_quantity:
            second_row_size = quantity - first_row_size
            current_vwap = (first_row_vwap*first_row_size + second_row_price*second_row_size)/quantity
            return current_vwap
        else:
            second_row_size =(quantity - first_row_vwap*first_row_size)/second_row_price
            current_vwap = quantity/(first_row_size+second_row_size)
            return current_vwap        
    
     

def fair_value_calculation(depth_dict,fees,exchange_list,full_exchanges,base_ccy,quote_ccy,quantity,is_quantity,enable_fx_rate,method,fx_exchanges,usdc_parity):
    
    exchanges = {key: full_exchanges[key] for key in exchange_list if key in full_exchanges}
    
    bid_order_book, ask_order_book, quote_ccy_rate = aggreagted_order_book(depth_dict,fees,exchanges,base_ccy,quote_ccy,enable_fx_rate,method,fx_exchanges,usdc_parity)
    print(f"exchanges: {','.join(list(exchanges.keys()))}")
    print()
    if bid_order_book is None or ask_order_book is None:
        return None,None,None
    else:
        bid_order_book = bid_order_book.groupby('price').agg({'quantity':'sum'}).sort_values(by='price',ascending=False).reset_index()
        ask_order_book = ask_order_book.groupby('price').agg({'quantity':'sum'}).sort_values(by='price',ascending=True).reset_index()
        
        bid_order_book['cumulative_notional'] = (bid_order_book['price']*bid_order_book['quantity']).cumsum()
        ask_order_book['cumulative_notional'] = (ask_order_book['price']*ask_order_book['quantity']).cumsum()
        
        bid_order_book['cumulative_quantity'] = bid_order_book['quantity'].cumsum()
        ask_order_book['cumulative_quantity'] = ask_order_book['quantity'].cumsum()
        
        bid_order_book['vwap'] = bid_order_book['cumulative_notional']/bid_order_book['cumulative_quantity']
        ask_order_book['vwap'] = ask_order_book['cumulative_notional']/ask_order_book['cumulative_quantity']
        bid_list = bid_order_book[['vwap','cumulative_quantity']].values
        ask_list = ask_order_book[['vwap','cumulative_quantity']].values
        top_bid = bid_order_book['vwap'].iloc[0]/quote_ccy_rate.get(quote_ccy,1.0)
        top_ask = ask_order_book['vwap'].iloc[0]/quote_ccy_rate.get(quote_ccy,1.0)
        
        ### calculating fair price
        
        if is_quantity:
            bid_index = np.searchsorted(bid_order_book['cumulative_quantity'],quantity)
            ask_index = np.searchsorted(ask_order_book['cumulative_quantity'],quantity)
            bid_fair_price = get_precise_vwap_price(bid_order_book,quantity,is_quantity,bid_index) 
            ask_fair_price = get_precise_vwap_price(ask_order_book,quantity,is_quantity,ask_index)
            bid_fair_price_bps = 0.0 
            ask_fair_price_bps = 0.0
            if bid_fair_price != None: 
                bid_fair_price = round(bid_fair_price/quote_ccy_rate.get(quote_ccy,1.0),4) 
                bid_fair_price_bps = round((top_bid - bid_fair_price)/top_bid*10000,2)
            if ask_fair_price != None: 
                ask_fair_price = round(ask_fair_price/quote_ccy_rate.get(quote_ccy,1.0),4)
                ask_fair_price_bps = round((ask_fair_price - top_ask)/top_ask*10000,2)

            print(f"The top of the Bid/Ask price for {base_ccy}/{quote_ccy} is: {round(top_bid,4)}/{round(top_ask,4)}")
            print(f"The Bid/Ask fair price for {quantity} {base_ccy}/{quote_ccy} is: {bid_fair_price}/{ask_fair_price}")
            #print(f"The Bid/Ask fair price for {quantity} {base_ccy}/{quote_ccy} in bps term is: {bid_fair_price_bps}/{ask_fair_price_bps} ")   
              
            return top_bid, top_ask,bid_fair_price,ask_fair_price,bid_list,ask_list
        else:
            bid_index = np.searchsorted(bid_order_book['cumulative_notional'],quantity)
            ask_index = np.searchsorted(ask_order_book['cumulative_notional'],quantity)
            bid_fair_price = get_precise_vwap_price(bid_order_book,quantity,is_quantity,bid_index) 
            ask_fair_price = get_precise_vwap_price(ask_order_book,quantity,is_quantity,ask_index)
            bid_fair_price_bps = 0.0 
            ask_fair_price_bps = 0.0
            if bid_fair_price != None: 
                bid_fair_price = round(bid_fair_price/quote_ccy_rate.get(quote_ccy,1.0),4) 
                bid_fair_price_bps = round((top_bid - bid_fair_price)/top_bid*10000,2)
            if ask_fair_price != None: 
                ask_fair_price = round(ask_fair_price/quote_ccy_rate.get(quote_ccy,1.0),4)
                ask_fair_price_bps = round((ask_fair_price - top_ask)/top_ask*10000,2)

            print(f"The top of the Bid/Ask price for {base_ccy}/{quote_ccy} is: {round(top_bid,4)}/{round(top_ask,4)}")    
            print(f"The Bid/Ask fair price for ${quantity} {base_ccy}/{quote_ccy} is: {bid_fair_price}/{ask_fair_price}")
            #print(f"The Bid/Ask fair price for ${quantity} {base_ccy}/{quote_ccy} in bps term is: {bid_fair_price_bps}/{ask_fair_price_bps} ")   
              
            return top_bid,top_ask,bid_fair_price,ask_fair_price,bid_list,ask_list



### interactive graph showing aggreagting orderbook info (quantity, price) for bid orderbook  and ask orderbook
def interactive_fair_price(x,y,z,h):
    
    fig, ax = plt.subplots(nrows=2, ncols=1,figsize =(12,6))
    ax[0].plot(x, y, label='fair price curve')
    ax[0].set_xlabel('bid quantity')
    ax[0].set_ylabel('bid price')
    ax[0].set_title('bid curve')

    # Plotting the second curve on the second subplot (ax[1])
    ax[1].plot(z, h, label='fair value curve')
    ax[1].set_xlabel('ask quantity')
    ax[1].set_ylabel('ask price')
    ax[1].set_title('ask curve')

   
    # Adjust layout
    plt.tight_layout()
    
    def on_move(event):
        if event.inaxes:
            print(f'data coords {event.xdata} {event.ydata},',
              f'pixel coords {event.x} {event.y}')

    def on_click(event):
        if event.button is MouseButton.LEFT:
            print('disconnecting callback')
            plt.disconnect(binding_id)


    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    # Display the plot
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.show()
    


### below are functions initilization in ccxt and get the exchange list  
def load_exchange(exchange_name, credentials):
    logging.info(f"Loading exchange {exchange_name}")
    try:
        if exchange_name == 'okx_perp':
            exchange_name = 'okx'
        exchange_init = getattr(ccxt, exchange_name)
        if credentials:
            exchange = exchange_init(credentials)
        else:
            exchange = exchange_init()
        exchange.load_markets()
    except:
        logging.warning(f"Exchange {exchange_name} not found")
        return None
    return exchange

def get_exchanges(selected_exchanges, credentials):
    if not selected_exchanges:
        return None
    exchanges = {key: load_exchange(key.lower(), credentials.get(key, None)) for key in selected_exchanges}
    return exchanges