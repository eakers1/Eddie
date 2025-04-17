from risk.dealerArb import create_connection
import os
import datetime
import requests
import hmac
import hashlib
import base64
import json
import uuid
import pprint
import argparse
import sys
import logging
import time
import pandas as pd
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter, Retry


class Talos:
    _TALOS_REST_MARKET_DATA = "https://talostrading.com"

    def __init__(self, api_key, api_secret, host, version="v1"):
        self._path_ws = "/ws/" + version
        self._url_rest = "https://" + host
        self._path_rest = "/" + version
        self._host = host
        self._api_key = api_key
        self._api_secret = api_secret
        self._session = requests.Session()

        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[401, 500, 502, 503, 504])
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def _date_now(self):
        utc_now = datetime.datetime.utcnow()
        return utc_now.strftime("%Y-%m-%dT%H:%M:%S.000000Z")

    def _signature(self, path, query=None, body=None, method="GET"):
        utc_datetime = self._date_now()
        #print(utc_datetime+"\n---------")
        param_list = [method, utc_datetime, self._host, path]
        if query is not None:
            param_list.append(query)
        if body is not None:
            param_list.append(body)
        params = "\n".join(param_list)
        # print(params)
        # print("\n")
        hash = hmac.new(self._api_secret.encode("ascii"), params.encode("ascii"), hashlib.sha256)
        signature = base64.urlsafe_b64encode(hash.digest()).decode()
        return {
            "TALOS-KEY": self._api_key,
            "TALOS-SIGN": signature,
            "TALOS-TS": utc_datetime,
        }

    def _connect_ws(self):
        sign = self._signature(self._path_ws)
        ws = create_connection("wss://" + self._host + self._path_ws, header=sign)
        ws.recv()  # Read hello
        return ws
    
    def _new_uuid(self):
        return str(uuid.uuid4())

    def get_pairs(self):
        """Retrieve all supported trading pairs"""
        ws = self._connect_ws()
        ws.send("""{"reqid": 1, "type": "subscribe", "streams": [{"name": "Security"}]}""")
        reply = ws.recv()
        ws.close()
        return json.loads(reply)

    def get_customer_quotes_feed(self, rfq_id, symbol, time_delta):
        """Listen to customer quotes feed"""
        now = datetime.datetime.now()
        ws = self._connect_ws()
        payload = {"reqid": 13, "type": "subscribe", "streams": [{"name": "Quote"}]}

        if rfq_id:
            payload["streams"][0]["RFQID"] = rfq_id

        if symbol:
            payload["streams"][0]["Symbol"] = symbol

        delta = 10
        if time_delta:
            delta = time_delta

        time_ago = now - datetime.timedelta(minutes=int(delta))
        payload["streams"][0]["StartDate"] = time_ago.strftime("%Y-%m-%dT%H:%M:%SZ")  # ISO-8601 UTC

        ws.send(json.dumps(payload))

        try:
            while True:
                reply = ws.recv()
                response = json.loads(reply)
                print(json.dumps(response, indent=4))
        except KeyboardInterrupt:
            ws.close()

    def get_customer_execution_report_feed(self):
        """Listen to customer execution report feed"""
        now = datetime.datetime.now()
        ws = self._connect_ws()
        payload = {
            "reqid": 8,
            "type": "subscribe",
            "streams": [{"name": "CustomerExecutionReport"}],
        }

        ws.send(json.dumps(payload))

        try:
            while True:
                reply = ws.recv()
                response = json.loads(reply)
                print(json.dumps(response, indent=4))
        except KeyboardInterrupt:
            ws.close()

    def get_customer_trades_feed(self):
        """Listen to customer trades feed"""
        now = datetime.datetime.now()
        ws = self._connect_ws()
        payload = {
            "reqid": 3,
            "type": "subscribe",
            "streams": [{"name": "CustomerTrade"}],
        }

        ws.send(json.dumps(payload))

        try:
            while True:
                reply = ws.recv()
                response = json.loads(reply)
                print(json.dumps(response, indent=4))
        except KeyboardInterrupt:
            ws.close()

    def request_quote(self, symbol, currency, quantity):
        """Request a customer quote"""
        req = {
            "Counterparty": "Anchorage",
            "QuoteReqID": str(uuid.uuid4()),
            "Symbol": symbol,
            "OrderQty": quantity,
            "Currency": currency,
            "TransactTime": self._date_now(),
            "Parameters": {
                "RFQTTL": "6s",
                "QuoteTTL": "3s",
                "RequestTimeout": "6s",
                "Spread": "0",
                "PricesTimeout": "3s",
            },
        }

        path = self._path_rest + "/customer/quotes"
        query = "wait-for-status=confirmed&timeout=5s"
        body = json.dumps(req)
        sign = self._signature(path, query, body, "POST")
        result = requests.post(self._url_rest + path, data=body, params=query, headers=sign)
        result.raise_for_status()
        return result.json()["data"][0]

    def order(self, quote_id, rfq_id, side, symbol, quantity, price):
        """Request a customer order"""
        req = {
            "Counterparty": "Anchorage",
            "ClOrdID": str(uuid.uuid4()),
            "Symbol": symbol,
            "Side": side,
            "QuoteID": quote_id,
            "RFQID": rfq_id,
            "OrderQty": quantity,
            "Price": price,
            "TransactTime": self._date_now(),
            "OrdType": "RFQ",
        }

        path = self._path_rest + "/customer/orders"
        query = "wait-for-status=completed&timeout=5s"
        body = json.dumps(req)
        sign = self._signature(path, query, body, "POST")
        result = requests.post(self._url_rest + path, data=body, params=query, headers=sign)
        result.raise_for_status()
        return result.json()["data"][0]

    def create_order(self, side, symbol, currency, quantity):
        """Request a customer market order"""
        req = {
            "Counterparty": "Anchorage",
            "ClOrdID": str(uuid.uuid4()),
            "Symbol": symbol,
            "Currency": currency,
            "Side": side,
            "OrderQty": quantity,
            "TransactTime": self._date_now(),
            "OrdType": "Market",
            "TimeInForce": "FillOrKill",
        }

        path = self._path_rest + "/customer/orders"
        query = "wait-for-status=completed&timeout=12s"
        body = json.dumps(req)
        sign = self._signature(path, query, body, "POST")
        result = requests.post(self._url_rest + path, data=body, params=query, headers=sign)
        result.raise_for_status()
        return result.json()["data"][0]

    def get_market_data_feed(self, symbol, markets, **kwargs):
        """Listen to market data feed"""
        ws = self._connect_ws()
        payload = {
            "reqid": 5,
            "type": "subscribe",
            "streams": [
                {
                    "name": "MarketDataSnapshot",
                    "Symbol": symbol,
                    "DepthType": "Price",  # L2 style price depth; alternative is "VWAP"
                    "Markets": markets,
                    "Throttle": "1s",
                }
            ],
        }
        if kwargs:
            payload["streams"][0].update(kwargs)

        ws.send(json.dumps(payload))

        try:
            while True:
                reply = ws.recv()
                response = json.loads(reply)
                print(json.dumps(response, indent=4))
                return response
        except KeyboardInterrupt:
            ws.close()

    def get_orders_feed(self):
        ws = self._connect_ws()
        payload = {
            "reqid": 9,
            "type": "subscribe",
            "streams": [{"name": "Order", "StartDate": "2023-01-17T17:46:16.000000Z"}],
        }
        try:
            while True:
                print("here")
                reply = ws.recv()
                response = json.loads(reply)
                print(json.dumps(response, indent=4))

        except KeyboardInterrupt:
            ws.close()

    def get_market_data_feed_v2(self, symbol, dealers, exchanges=None, **kwargs):
        """Listen to market data feed"""

        logging.info(f"Fetching for {symbol} on {exchanges} and {dealers} | {kwargs}")

        ws = self._connect_ws()
        payload = {
            "reqid": 5,
            "type": "subscribe",
            "streams": [
                {
                    "name": "MarketDataSnapshot",
                    "Symbol": symbol,
                    "DepthType": "VWAP",  # L2 style price depth; alternative is "VWAP"
                    "Markets": dealers,
                    "Throttle": "5s",
                }
            ],
        }

        if exchanges != None:
            exchange_streams = {
                "name": "MarketDataSnapshot",
                "Symbol": symbol,
                "DepthType": "VWAP",  # L2 style price depth; alternative is "VWAP"
                "Markets": "exchange",
                "Throttle": "5s",
            }

            for exchange in exchanges:
                if (exchange in ["binance", "kucoin"]) and symbol[-3:] == "USD":
                    exchange_streams["Symbol"] = symbol.replace("USD", "USDT")
                exchange_streams["Markets"] = [exchange]
                payload["streams"].append(exchange_streams.copy())

        if kwargs:
            for x in range(0, len(payload["streams"])):
                payload["streams"][x].update(kwargs)

        ws.send(json.dumps(payload))

        try:
            data_dict = {}
            i = 0
            while True:
                reply = ws.recv()
                response = json.loads(reply)
                data_dict[i] = response["data"]
                # print(json.dumps(response, indent=4))
                i += 1
                if len(data_dict) == len(payload["streams"]):
                    return data_dict

        except KeyboardInterrupt:
            ws.close()

    def get_currency_conversion(self, symbol):
        """Listen to currency conversion feed"""
        ws = self._connect_ws()
        payload = {
            "reqid": 6,
            "type": "subscribe",
            "streams": [
                {
                    "name": "CurrencyConversion",
                    "Currencies": [symbol],
                    "EquivalentCurrency": "USD",
                    "Throttle": "5s",
                }
            ],
        }
        ws.send(json.dumps(payload))

        try:
            while True:
                reply = ws.recv()
                response = json.loads(reply)
                print(json.dumps(response, indent=4))
        except KeyboardInterrupt:
            ws.close()

    def get_customer_security(self, symbol):
        ws = self._connect_ws()
        payload = {
            "reqid": 1,
            "type": "subscribe",
            "streams": [
                {
                "name": "CustomerSecurity"
                }
            ]
        }

        ws.send(json.dumps(payload))
        
    def get_trade_history(self, StartDate):
        ws = self._connect_ws()
        payload = {
            "reqid": 3,
            "type": "subscribe",
            "streams": [
                {
                "name": "TradeHistory",
                "StartDate": StartDate
                }
            ]
            }

        ws.send(json.dumps(payload))

    def get_order_analytics(self, order_id, resolution):
        ws = self._connect_ws()

        payload = {
            "reqid": 1,
            "type": "subscribe",
            "streams": [{"name": "OrderAnalytic", "OrderID": order_id, "Resolution": resolution}],
        }

        ws.send(json.dumps(payload))
        order_analytics = pd.DataFrame()
        while True:
            try:
                result = json.loads(ws.recv())
                # json_formatted_str = json.dumps(result, indent = 4)
                # print(json_formatted_str)
                if result["type"] == "OrderAnalytic":
                    order_analytics = pd.concat([order_analytics, pd.DataFrame(result["data"])], ignore_index=True)
                    # print(result.get('next'))
                    if not result.get("next"):
                        logging.info("Finished pagination")
                        break

                    payload["type"] = "page"
                    payload["streams"][0]["after"] = result["next"]

                    ws.send(json.dumps(payload))

            except Exception as e:
                logging.critical(e)
                break

        order_analytics.StartTime = pd.to_datetime(order_analytics.StartTime)
        order_analytics.set_index("StartTime", inplace=True)  # set the Date column as the index
        order_analytics.index = order_analytics.index.tz_convert("US/Eastern").tz_localize(None)
        order_analytics = order_analytics.reset_index()
        order_analytics = order_analytics[order_analytics.TradedAvgPx != "0"]  # this might be a bad assumption
        order_analytics["ArrivalPx"] = order_analytics["ArrivalPx"].iloc[0]

        return order_analytics

    # def get_trade_analytics(self, order_id):
    #         ws = self._connect_ws()
    #         payload = {
    #             "reqid": 1,
    #             "type": "subscribe",
    #             "streams": [{"name": "TradeAnalytics", "OrderID": order_id}],
    #         }
    #         ws.send(json.dumps(payload))
    #         trade_analytics = pd.DataFrame()
    #         trade_analytics_live = pd.DataFrame()
    #         combined_data = pd.DataFrame()
    #         last_next = None
    #         while True:
    #             try:
    #                 result = json.loads(ws.recv())
    #                 #print(payload)
    #                 if result["type"] == "TradeAnalytics":
    #                     if result.get("next"):
    #                         next_id = result["next"]
    #                         logging.info(f"Received paginated data, next_id = {next_id}, page = {result.get('page', False)}")
    #                         print(payload)

    #                         trade_analytics = pd.concat([trade_analytics, pd.DataFrame(result["data"])], ignore_index=True)
    #                         if not result.get("page") and result.get("next"):
    #                             json_formatted_str = json.dumps(result, indent = 4)
    #                             print(json_formatted_str)
    #                             logging.info("Finished pagination")
    #                             break
    #                         payload["type"] = "page"
    #                         payload["streams"][0]["after"] = next_id
    #                         ws.send(json.dumps(payload))
    #                         logging.info("Sent paginated request")
    #                     else:
    #                         logging.info("Live Data")
    #                         trade_analytics_live = pd.concat([trade_analytics_live, pd.DataFrame(result["data"])], ignore_index=True)
    #                 logging.info("Combining data")
    #                 combined_data = pd.concat([trade_analytics, trade_analytics_live], ignore_index=True)
    #                 combined_data = combined_data.drop_duplicates(subset='TradeID', keep='last')

    #             except Exception as e:
    #                 logging.critical(e)
    #                 break
    #         return combined_data

    def get_trade_analytics(self, order_id):
            ws = self._connect_ws()
            payload = {
                "reqid": 1,
                "type": "subscribe",
                "streams": [{"name": "TradeAnalytics", "OrderID": order_id}],
            }
            ws.send(json.dumps(payload))
            trade_analytics = pd.DataFrame()
            trade_analytics_live = pd.DataFrame()
            combined_data = pd.DataFrame()
            
            while True:
                try:
                    result = json.loads(ws.recv())
                    #print(payload)
                    if result["type"] == "TradeAnalytics":
                        if result.get("next"):
                            next_id = result["next"]
                            logging.info(f"Received paginated data, page = {result.get('page', False)}, next_id = {next_id}")
                            trade_analytics = pd.concat([trade_analytics, pd.DataFrame(result["data"])], ignore_index=True)
                            payload["type"] = "page"
                            payload["streams"][0]["after"] = next_id
                            ws.send(json.dumps(payload))
                            print(payload)
                            
                        elif result.get("page") and not result.get("next"):
                            json_formatted_str = json.dumps(result, indent = 4)
                            print(json_formatted_str)
                            logging.info("Finished pagination")
                            break
                        else:
                            #logging.info("Passing live data")
                            continue
                    #         logging.info("Live Data")
                    #         trade_analytics_live = pd.concat([trade_analytics_live, pd.DataFrame(result["data"])], ignore_index=True)

                    # combined_data = pd.concat([trade_analytics, trade_analytics_live], ignore_index=True)
                    # combined_data = combined_data.drop_duplicates(subset='TradeID', keep='last')
                    trade_analytics = trade_analytics.drop_duplicates(subset='TradeID', keep='last')
                    print(f"Combined data length: {len(trade_analytics)}")
                    #print(f"Last row: {trade_analytics.iloc[-1]}")

                except Exception as e:
                    logging.critical(e)
                    break
            return trade_analytics

    def get_quote_by_rfqid(self, id):
        """Fetch an existing quote by RFQID"""
        path = self._path_rest + "/customer/quotes/" + id
        sign = self._signature(path)
        result = requests.get(self._url_rest + path, headers=sign)
        result.raise_for_status()
        return result.json()["data"][0]

    def get_quote_summary(self, id):
        """Fetch an existing quote summary by RFQID"""
        path = self._path_rest + "/customer/quotes/" + id + "/summary"
        sign = self._signature(path)
        result = requests.get(self._url_rest + path, headers=sign)
        result.raise_for_status()
        return result.json()["data"][0]

    def get_order_by_clordid(self, id):
        """Fetch an existing order by ClOrdID"""
        path = self._path_rest + "/customer/orders"
        query = "ClOrdID=" + id
        sign = self._signature(path, query)
        result = requests.get(self._url_rest + path, params=query, headers=sign)
        result.raise_for_status()
        return result.json()["data"][0]

    def get_trade_by_id(self, trade_id):
        """Fetch an existing trade by trade id"""
        path = self._path_rest + "/trades"
        query = "TradeID=" + trade_id
        sign = self._signature(path, query)
        result = requests.get(self._url_rest + path, params=query, headers=sign)
        result.raise_for_status()
        return result.json()["data"]

    def get_order_by_order_id(self, order_id):
        """Fetch an existing order by order id"""
        path = self._path_rest + "/trade-history"
        query = "OrderID=" + order_id
        sign = self._signature(path, query)
        result = requests.get(self._url_rest + path, params=query, headers=sign)
        result.raise_for_status()
        return result.json()

    # def get_orders(self, start_date, end_date):
    #     path = self._path_rest + "/orders"
    #     query = {"StartDate": start_date, "EndDate": end_date}
    #     sign = self._signature(path, query)
    #     result = requests.get(self._url_rest + path, params=query, headers=sign)
    #     result.raise_for_status()
    #     return result.json()["data"][0]

    def _counterparty_type(self, x):
        key = "Type"
        if x.get(key, -1) != -1:
            if x[key] == "Exchange" or x[key] == "Dealer":
                return True
        return False

    def get_counterparties(self):
        """Fetch dealer and exchange counterparties enabled for Anchorage"""
        path = self._path_rest + "/market-accounts"
        sign = self._signature(path)
        result = requests.get(self._url_rest + path, headers=sign)
        result.raise_for_status()

        active_markets = filter(lambda x: x["Type"] == "Trading", result.json()["data"])
        filter_dict = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

        return map(
            lambda x: filter_dict(x, ("Market", "DisplayName", "Status")),
            active_markets,
        )

    # This function follows an improved format for REST requests
    # as we can add any parameters available on talos docs and get paginated results from the getgo
    def get_trade_by_order_id(self, order_id, clordid=None, start_date=None, end_date=None, statuses=None, limit=1000):
        """Fetch an existing trade by trade id"""
        path = self._path_rest + "/trades"
        params = {
            "OrderID": order_id,
            "ClOrdID": clordid,
            "StartDate": start_date,
            "EndDate": end_date,
            "Statuses": statuses,
            "limit": limit,
        }
        return self._paginate_request(path, params)

    def get_orders(
        self,
        order_id=None,
        cl_order_id=None,
        start_date=None,
        end_date=None,
        statuses=None,
        rfqid=None,
        group=None,
        limit=None,
    ):
        path = self._path_rest + "/orders"

        params = {
            "OrderID": order_id,
            "ClOrdID": cl_order_id,
            "StartDate": start_date,
            "EndDate": end_date,
            "Statuses": statuses,
            "RFQID": rfqid,
            "Group": group,
            "limit": limit,
        }

        return self._paginate_request(path, params)

    def get_ohlcv(self, symbol, markets, resolution, start_date, end_date, limit=10000):
        path = f"/v1/symbols/{symbol}/markets/{markets}/ohlcv/{resolution}"

        params = {"startDate": start_date, "endDate": end_date, "limit": limit}

        return self._paginate_request(path, params, self._TALOS_REST_MARKET_DATA)

    def _paginate_request(self, path, params=None, url=None):
        all_results = []

        if url:
            url += path
        else:
            url = self._url_rest + path

        #retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[401, 500, 502, 503, 504])

        #self._session.mount("https://", HTTPAdapter(max_retries=retries))

        while True:
            if params:
                query = urlencode({k: v for k, v in params.items() if v is not None})
            else:
                query = None

            try:
                sign = self._signature(path, query)
                response = self._session.get(url, params=query, headers=sign)
                response.raise_for_status()
                result = response.json()
                all_results.extend(result["data"])

                if not result.get("next"):
                    break

                next_token = result["next"]
                params["after"] = next_token

            except requests.exceptions.RequestException as e:
                print(f"An exception was raised: {e}")
                return

        return all_results

    def get_markets(self):
        path = self._path_rest + "/markets"

        return self._paginate_request(path)

    def get_execution_reports(self,
        order_id=None,
        cl_order_id=None,
        start_date=None,
        end_date=None,
        statuses=None,
        rfqid=None,
        group=None,
        sub_accounts=None,
        limit=None,
    ):

        path = self._path_rest + "/execution-reports"

        params = {
            "OrderID": order_id,
            "ClOrdID": cl_order_id,
            "StartDate": start_date,
            "EndDate": end_date,
            "Statuses": statuses,
            "RFQID": rfqid,
            "Group": group,
            "SubAccounts": sub_accounts,
            "limit": limit,
        }

        return self._paginate_request(path, params)

    def get_trade_fills(self, order_id):
        trades_list = self.get_trade_by_order_id(order_id)
        all_trades = pd.DataFrame(trades_list)
        all_trades.Timestamp = pd.to_datetime(all_trades.Timestamp)
        all_trades.set_index("Timestamp", inplace=True)  # set the Date column as the index
        all_trades.index = all_trades.index.tz_convert("US/Eastern").tz_localize(None)
        all_trades = all_trades.reset_index()

        all_trades.Amount = pd.to_numeric(all_trades.Amount)

        return all_trades

    def send_rfq(self, symbol: str, amount: float, markets: list, currency: str = None, side: str = "", post_params: dict = None, wait_for_status: str = "confirmed", timeout: str = "5s"):
        """Send an RFQ to the Talos RFQ API"""     
        utc_datetime = self._date_now()
        if not currency:
            currency = symbol.split("-")[0]
        
        logging.info(f"Sending RFQ for {amount} {symbol} {side} on {markets}")

        path = self._path_rest + "/quotes"
        print(path)

        body = json.dumps({
            "Symbol": symbol,
            "QuoteReqID": self._new_uuid(),
            "Side": side,
            "OrderQty": amount,
            "Currency": currency,
            "Markets": markets,
            "TransactTime": utc_datetime,
            "Parameters": {
                "PricesTimeout": "5s"
            }

        })

        params = {
            "wait-for-status": wait_for_status,
            "timeout": timeout,
        }

        if post_params:
            body["Parameters"] = post_params

        try:
            sign = self._signature(path ,body=body, method="POST")
            response = self._session.post(self._url_rest + path, headers=sign, data=body)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(e)
            return e
        
    def retrieve_quote(self, quote_id: str):
        """Retrieve a quote from the Talos RFQ API"""
        path = self._path_rest + f"/quotes/{quote_id}"

        params = {"RFQID": quote_id}

        params = urlencode({k: v for k, v in params.items() if v is not None})

        try:
            sign = self._signature(path)
            response = self._session.get(self._url_rest + path, headers=sign)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(e)
            return e