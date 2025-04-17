import pandas as pd
import numpy as np
import datetime as dt
import logging
import pytz
import ccxt  # type: ignore
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression


class MarketData:
    def __init__(
        self,
        exchange: str,
        symbol: str = None,
        future_symbol: str = None,
        intervals: list = ["1s", "1m", "5m", "1h", "1d"],
        date_range: dict = {"1s": 1, "1m": 48, "5m": 168, "1h": 720, "1d": 8760},
        credentials: dict = None,
        debug: bool = False,
    ):
        self._configure_logging(debug)
        self._validate_symbol(symbol, future_symbol)
        self._init_exchange(exchange, credentials)
        self._init_data()

        self.symbol = symbol
        self.future_symbol = future_symbol
        self.interval = intervals
        self.date_range = date_range
        self.last_run = None

        self._fetch_data()

    def _configure_logging(self, debug):
        if debug:
            logging.basicConfig(
                format="%(levelname)s (%(asctime)s): %(message)s (Line:%(lineno)d in %(filename)s))",
                datefmt="%Y/%m/%d %I:%M:%S %p",
                level=logging.INFO,
            )

    def _validate_symbol(self, symbol, future_symbol):
        if not symbol and not future_symbol:
            raise ValueError("Either symbol or future_symbol must be provided")

    def _init_exchange(self, exchange, credentials):
        exchange = getattr(ccxt, exchange)
        if credentials:
            self.exchange = exchange(credentials)
        else:
            self.exchange = exchange()

        logging.info("Loading markets")
        self.markets = self.exchange.load_markets()

    def _init_data(self):
        self.ob_data_spot = None
        #self.ohlcv_data_spot = {}
        self.ob_data_future = None
        self.ohlcv_data_future = {}
        self.ohlcv_data = {"spot": {}, "future": {}}

        self.spot_mid_price = None
        self.future_mid_price = None
        self.top_bid_ask = {"spot": {}, "future": {}}

        self.historical_funding_rate = None
        self.funding_rate_summary = None
        self.next_funding_rate = None
        self.funding_datetime = None
        self.next_funding_seconds = None

    def _calculate_interval_hours(self, interval):
        # Calculate the number of hours to fetch data for

        return self.date_range.get(interval, 168)

    def _fetch_data(self):
        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            if self.symbol:
                logging.info(f"Fetching order book data for {self.symbol}")
                executor.submit(self.fetch_spot_order_book)
                executor.submit(self.fetch_ohlcv_spot)

            if self.future_symbol:
                logging.info(f"Fetching funding rate data for {self.future_symbol}")
                executor.submit(self.fetch_future_order_book)
                executor.submit(self.fetch_ohlcv_future)
                logging.info(f"Fetching funding rate data")
                executor.submit(self.fetch_funding_rate)
                executor.submit(self.fetch_funding_rate_hist)

        self.last_run = dt.datetime.utcnow()

    def reload_data(self):
        self._fetch_data()
        return

    def fetch_spot_order_book(self):
        self.ob_data_spot = self._fetch_ob_data(self.symbol)
        self.spot_mid_price = self._calculate_mid_price(self.ob_data_spot)
        self.top_bid_ask["spot"] = self._get_top_ob_prices(self.ob_data_spot)
        logging.info(f"Fetched spot order book successfuly")
        return self.ob_data_spot

    def fetch_future_order_book(self):
        self.ob_data_future = self._fetch_ob_data(self.future_symbol)
        self.future_mid_price = self._calculate_mid_price(self.ob_data_future)
        self.top_bid_ask["future"] = self._get_top_ob_prices(self.ob_data_future)
        logging.info(f"Fetched future successfuly")
        return self.ob_data_future
    
    def fetch_ohlcv_spot(self):
        if self.last_run is None:
            for interval in self.interval:
                logging.info(f"Fetching OHLCV data for {self.symbol} ({interval})")
                # self.ohlcv_data_spot[interval] = self._fetch_ohlcv_data(
                #     self.symbol, interval
                # )
                self.ohlcv_data["spot"][interval] = self._fetch_ohlcv_data(
                    self.symbol, interval
                )
            self.ohlcv_data["spot"] = {k:v for (k,v) in self.ohlcv_data["spot"].items() if not v.empty}

        else:
            now = dt.datetime.utcnow()
            for interval in self.ohlcv_data["spot"].keys():
                # hours = (now - self.ohlcv_data_spot[interval].iloc[-1].name) / dt.timedelta(hours=1)
                hours = (now - self.ohlcv_data["spot"][interval].iloc[-1].name) / dt.timedelta(hours=1)
                logging.info(f"Updating OHLCV data for {self.symbol} ({interval})")
                df = self._fetch_ohlcv_data(
                    self.symbol, interval, hours=hours, end_time=now
                )
                #df_concat = pd.concat([self.ohlcv_data_spot[interval], df])
                df_concat = pd.concat([self.ohlcv_data["spot"][interval], df])
                df_concat = df_concat.loc[~df_concat.index.duplicated(keep='last')]
                #self.ohlcv_data_spot[interval] = df_concat
                self.ohlcv_data["spot"][interval] = df_concat

        

    def fetch_ohlcv_future(self):
        if self.last_run is None:
            if self.future_symbol:
                for interval in self.interval:
                    logging.info(
                        f"Fetching OHLCV data for {self.future_symbol} ({interval})")
                    # self.ohlcv_data_future[interval] = self._fetch_ohlcv_data(
                    #     self.future_symbol, interval
                    # )
                    self.ohlcv_data["future"][interval] = self._fetch_ohlcv_data(
                        self.future_symbol, interval
                    )

                self.ohlcv_data["future"] = {k:v for (k,v) in self.ohlcv_data["future"].items() if not v.empty}
        else:
            now = dt.datetime.utcnow()
            for interval in self.ohlcv_data["future"].keys():
                #hours = (now - self.ohlcv_data_future[interval].iloc[-1].name) / dt.timedelta(hours=1)
                hours = (now - self.ohlcv_data["future"][interval].iloc[-1].name) / dt.timedelta(hours=1)
                logging.info(f"Updating OHLCV data for {self.future_symbol} ({interval})")
                df = self._fetch_ohlcv_data(
                    self.future_symbol, interval, hours=hours, end_time=now
                )
                #df_concat = pd.concat([self.ohlcv_data_future[interval], df])
                df_concat = pd.concat([self.ohlcv_data["future"][interval], df])
                df_concat = df_concat.loc[~df_concat.index.duplicated(keep='last')]
                #self.ohlcv_data_future[interval] = df_concat
                self.ohlcv_data["future"][interval] = df_concat

    def fetch_funding_rate(self, symbol=None):
        symbol = symbol or self.future_symbol
        funding_rate_info = self._fetch_funding_rate(self.future_symbol)
        self.next_funding_rate = funding_rate_info["fundingRate"]
        self.funding_datetime = funding_rate_info["fundingDatetime"]
        self.next_funding_seconds = funding_rate_info["nextFundingSeconds"]

    def fetch_funding_rate_hist(self, limit=30): #symbol=None, 
        #symbol = symbol if symbol else self.future_symbol
        # Removing the ability to request another symbol so we don't change the class attribute
        symbol = self.future_symbol
        self.historical_funding_rate = self._fetch_funding_rate_hist(symbol, limit)
        self.funding_rate_summary = self.historical_funding_rate.describe().to_dict()

        return self.historical_funding_rate

    def _fetch_ob_data(self, symbol, limit=1000):
        ob_data = self.exchange.fetch_order_book(symbol, limit)
        ob_data = pd.DataFrame(ob_data)

        bids_df = pd.DataFrame(
            ob_data["bids"].tolist(), columns=["price_bids", "size_bids"]
        )
        asks_df = pd.DataFrame(
            ob_data["asks"].tolist(), columns=["price_asks", "size_asks"]
        )

        df = pd.concat(
            [ob_data.drop(["bids", "asks"], axis=1), bids_df, asks_df], axis=1
        )

        logging.info(f"Order Book Data for {symbol} fetched successfully")

        return df

    def _fetch_ohlcv_data(self, symbol, interval, hours=None, end_time = dt.datetime.now(), limit=500):
        hours = hours or self._calculate_interval_hours(interval)
        start_time = end_time - dt.timedelta(hours=hours)
        since = int(start_time.timestamp() * 1000)
        all_data = []

        while since < int(end_time.timestamp() * 1000):
            try:
                logging.info(
                    f"Fetching OHLCV data for {symbol} ({interval} since {since})"
                )
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe=interval, since=since, limit=limit
                )
            except Exception as e:
                logging.warning(f"Error: {e} for {symbol} ({interval})")
                break

            # Break if no new data is fetched
            if not ohlcv:
                break

            all_data.extend(ohlcv)

            since = ohlcv[-1][0] + self.exchange.parse_timeframe(interval) * 1000

        df = pd.DataFrame(
            all_data, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.set_index("datetime")

        logging.info(f"OHLCV Data for {symbol} ({interval}) fetched successfully")

        return df

    def _fetch_funding_rate(self, symbol):
        funding_rate = self.exchange.fetch_funding_rate(symbol)
        info = {key: funding_rate[key] for key in ["fundingRate", "fundingDatetime"]}
        info["nextFundingSeconds"] = (
            pd.Timestamp(info["fundingDatetime"]) - dt.datetime.now(pytz.utc)
        ).total_seconds()

        logging.info(f"Funding Rate info for {symbol} fetched successfully")
        return info

    def _fetch_funding_rate_hist(self, symbol=None, limit=30):
        symbol = symbol if symbol else self.future_symbol
        funding_rate_raw = self.exchange.fetch_funding_rate_history(symbol, limit=limit)
        funding_rate = pd.concat(
            [pd.json_normalize(x["info"]) for x in funding_rate_raw]
        )
        funding_rate[["fundingRate", "markPrice", "fundingTime"]] = funding_rate[
            ["fundingRate", "markPrice", "fundingTime"]
        ].astype(float)
        funding_rate["fundingTime"] = pd.to_datetime(
            funding_rate["fundingTime"], unit="ms"
        )
        funding_rate.columns = ["symbol", "funding_time", "funding_rate", "mark_price"]
        funding_rate = funding_rate.set_index("funding_time")

        logging.info(f"Funding Rate Historical data for {symbol} fetched successfully")
        return funding_rate

    def _calculate_mid_price(self, df):
        mid_price = (df["price_asks"].iloc[0] + df["price_bids"].iloc[0]) / 2
        return mid_price

    def _get_top_ob_prices(self, df):
        top_bid = df["price_bids"].iloc[0]
        top_ask = df["price_asks"].iloc[0]

        top_bid_ask = {"bid": top_bid, "ask": top_ask}

        return top_bid_ask
    

# ############################################################################
# #                                                                          #
# #                            Liquidity Analyzer                            #
# #                                                                          #
# ############################################################################
    
class LiquidityAnalyzer:
    def __init__(
        self,
        ob_data_spot: pd.DataFrame = pd.DataFrame(),
        ob_data_future=None,
        ohlcv_data=None,
        debug: bool = False,
    ):
        self._configure_logging(debug)
        self._init_data()

        self.ob_spot = ob_data_spot
        self.ob_future = ob_data_future
        # self.ohlcv_spot = ohlcv_data_spot
        # self.ohlcv_future = ohlcv_data_future
        self.ohlcv_data_market = ohlcv_data

        self.create_liquidity_data()

    def _configure_logging(self, debug):
        if debug:
            logging.basicConfig(
                format="%(levelname)s (%(asctime)s): %(message)s (Line:%(lineno)d in %(filename)s))",
                datefmt="%Y/%m/%d %I:%M:%S %p",
                level=logging.INFO,
            )

    def _init_data(self):
        # self.ohlcv_data_spot = {}
        # self.ohlcv_data_future = {}
        self.ohlcv_data = {"spot": {}, "future": {}}

        # self.average_volumes = {"spot": {}, "future": {}}
        # self.average_volumes_ccy = {"spot": {}, "future": {}}
        # self.last_volumes = {"spot": {}, "future": {}}
        # self.last_volumes_ccy = {"spot": {}, "future": {}}
        self.average_volumes = {"spot": {"base": {}, "quote": {}}, "future": {"base": {}, "quote": {}}}
        self.last_volumes = {"spot": {"base": {}, "quote": {}}, "future": {"base": {}, "quote": {}}}

        self.volume_profile = {"spot": {}, "future": {}}

    def _calculate_rolling_window(self, interval):

        # TO-DO: Put this as a constant in a separate file for more dynamic modification
        interval_normalization = {
            "1s": 60,
            "1m": 60,
            "5m": 72,
            "1h": 24,
            "1d": 30
        }

        return interval_normalization.get(interval, interval_normalization["1h"])

    def _translate_rolling_window(self, interval):
        if interval == "1s":
            return "minutely"
        elif interval == "1m":
            return "hourly"
        elif interval == "5m":
            return "quarter_daily"
        elif interval == "1h":
            return "daily"
        elif interval == "1d":
            return "monthly"

    def _calculate_vwap_and_cum_size(self, df):
        # Calculate VWAP
        df["VWAP_asks"] = (df["price_asks"] * df["size_asks"]).cumsum() / df[
            "size_asks"
        ].cumsum()
        df["VWAP_bids"] = (df["price_bids"] * df["size_bids"]).cumsum() / df[
            "size_bids"
        ].cumsum()

        # Calculate Cumulative Size
        df["cum_asks"] = df["size_asks"].cumsum()
        df["cum_bids"] = df["size_bids"].cumsum()

        # Calculate USD Notional per level
        df["notional_level_asks"] = df["price_asks"] * df["size_asks"]  # * -1
        df["notional_level_bids"] = df["price_bids"] * df["size_bids"]

        # Calculate USD Notional
        df["notional_asks"] = df["VWAP_asks"] * df["cum_asks"]  # * -1
        df["notional_bids"] = df["VWAP_bids"] * df["cum_bids"]

        # Calculate Mid Price
        mid_price = (df["price_asks"].iloc[0] + df["price_bids"].iloc[0]) / 2

        # Calculate Distance from Mid Price
        df["distance_from_mid_price_asks"] = round(
            (df["VWAP_asks"] - mid_price) / mid_price * 10000, 4
        )
        df["distance_from_mid_price_bids"] = round(
            abs((df["VWAP_bids"] - mid_price) / mid_price * 10000), 4
        )

        return df

    def create_liquidity_data(self):
        with ThreadPoolExecutor() as executor:
            (
                executor.submit(self.create_spot_order_book(self.ob_spot))
                if self.ob_spot is not None and not self.ob_spot.empty
                else None
            )
            (
                executor.submit(self.create_future_order_book(self.ob_future))
                if self.ob_future is not None and not self.ob_future.empty
                else None
            )
            (
                executor.submit(self.create_spot_ohlcv(self.ohlcv_data_market["spot"]))
                if self.ohlcv_data_market["spot"] is not None
                else None
            )
            (
                executor.submit(self.create_future_ohlcv(self.ohlcv_data_market["future"]))
                if self.ohlcv_data_market["future"] is not None
                else None
            )
            (
                executor.submit(
                    self.construct_aggregated_obs([self.ob_spot, self.ob_future])
                )
                if self.ob_future is not None and not self.ob_future.empty
                else None
            )

        self.process_volume_data(self.ohlcv_data["spot"], "spot")
        self.process_volume_data(self.ohlcv_data["future"], "future")

    def create_spot_order_book(self, order_book):
        df = order_book.copy()
        self.ob_data_spot = self._calculate_vwap_and_cum_size(df)

    def create_future_order_book(self, order_book):
        df = order_book.copy()
        self.ob_data_future = self._calculate_vwap_and_cum_size(df)

    def construct_aggregated_obs(self, dfs: list):
        ask_dfs = []
        bid_dfs = []
        for df in dfs:
            df = df.copy()
            ask_dfs.append(df[["price_asks", "size_asks", "symbol"]])
            bid_dfs.append(df[["symbol", "size_bids", "price_bids"]])

        total_ask_df = pd.concat(ask_dfs).rename(columns={"symbol": "symbol_asks"})
        total_ask_df = total_ask_df.sort_values(by="price_asks").reset_index(drop=True)

        total_bid_df = pd.concat(bid_dfs).rename(columns={"symbol": "symbol_bids"})
        total_bid_df = total_bid_df.sort_values(
            by="price_bids", ascending=False
        ).reset_index(drop=True)

        total_df = pd.concat([total_bid_df, total_ask_df], axis=1)
        self.ob_data_aggregated = self._calculate_vwap_and_cum_size(total_df)

    def create_spot_ohlcv(self, ohlcv_dict):
        self.ohlcv_data["spot"] = self._create_ohlcv(ohlcv_dict)

    def create_future_ohlcv(self, ohlcv_dict):
        self.ohlcv_data["future"] = self._create_ohlcv(ohlcv_dict)

    def _create_ohlcv(self, ohlcv_dict):
        """
        The rolling window size, which varies by the interval:
        - "1s": 60 (last 60 seconds)
        - "1m": 60 (last 60 minutes)
        - "5m": 72 (last 360 minutes or 6 hours)
        - "1h": 24 (last 24 hours)
        - "1d": 30 (last 30 days)
        
        Unless modified in the _calculate_rolling_window method
        """
        
        df_dict = {"data": {}, "analytics": {}}

        for key in ohlcv_dict:
            df = ohlcv_dict[key].copy()
            window = self._calculate_rolling_window(key)
            #rolling_string = str(int(key[0])*window)+key[1]
            df["notional"] = df["volume"] * df["close"]
            df["cum_volume"] = df["volume"].cumsum()
            df["cum_notional"] = df["notional"].cumsum()
            #df[f"rolling_{rolling_string}_volume"] = df["volume"].rolling(window).sum()
            df[f"rolling_volume"] = df["volume"].rolling(window).sum()
            df["rolling_volume_ma"] = df["volume"].rolling(window).mean()
            df["rolling_volume_ema"] = df["volume"].ewm(span=window, adjust=True).mean()
            #df[f"rolling_{rolling_string}_notional"] = df["notional"].rolling(window).sum()
            df[f"rolling_notional"] = df["notional"].rolling(window).sum()
            df["rolling_notional_ma"] = df["notional"].rolling(window).mean()
            df["rolling_notional_ema"] = (df["notional"].ewm(span=window, adjust=True).mean())
            df_analytics = df.describe()

            df_dict["data"][key] = df
            df_dict["analytics"][key] = df_analytics

        return df_dict

    def _create_ob_imbalance(self, df):
        df["ob_imbalance"] = (df["cum_asks"] - df["cum_bids"]) / (df["cum_asks"] + df["cum_bids"])
        return df
    
    def process_volume_data(self, ohlcv_dict, data_type):
        if ohlcv_dict is None:
            self.average_volatilities[data_type] = None
            self.last_volatilities[data_type] = None
        else:
            for interval in ohlcv_dict["data"].keys():
                if not ohlcv_dict["data"][interval].empty:
                    window = self._calculate_rolling_window(interval)
                    #rolling_string = str(int(interval[0])*window)+interval[1]
                    self.average_volumes[data_type]["base"][interval] = ohlcv_dict["data"][interval].iloc[-1]["rolling_volume_ema"]
                    self.average_volumes[data_type]["quote"][interval] = ohlcv_dict["data"][interval].iloc[-1]["rolling_notional_ema"]
                    self.last_volumes[data_type]["base"][interval] = ohlcv_dict["data"][interval].iloc[-1][f"rolling_volume"]
                    self.last_volumes[data_type]["quote"][interval] = ohlcv_dict["data"][interval].iloc[-1][f"rolling_notional"]
                    
                    # Old method
                    # self.average_volumes[data_type][interval] = ohlcv_dict["data"][interval].iloc[-1]["rolling_volume_ema"]
                    # self.average_volumes_ccy[data_type][interval] = ohlcv_dict["data"][interval].iloc[-1]["rolling_notional_ema"]
                    # self.last_volumes[data_type][interval] = ohlcv_dict["data"][interval].iloc[-1][f"rolling_volume"]
                    # self.last_volumes_ccy[data_type][interval] = ohlcv_dict["data"][interval].iloc[-1][f"rolling_notional"]

                    # Have to decide if this naming convention is the more appropriate...
                    # self.average_volumes[data_type][rolling_string] = ohlcv_dict["data"][interval].iloc[-1]["rolling_volume_ema"]
                    # self.average_volumes_ccy[data_type][rolling_string] = ohlcv_dict["data"][interval].iloc[-1]["rolling_notional_ema"]
                    # self.last_volumes[data_type][rolling_string] = ohlcv_dict["data"][interval].iloc[-1][f"rolling_{rolling_string}_volume"]
                    # self.last_volumes_ccy[data_type][rolling_string] = ohlcv_dict["data"][interval].iloc[-1][f"rolling_{rolling_string}_notional"]
                    
    def calculate_volume_profile(self, ohlcv_dict, data_type, interval="1h"):
        frequency = {
            "1s": "second",
            "1m": "minute",
            "5m": "minute",
            "1h": "hour",
            "1d": "day"
        }
        if ohlcv_dict is None:
            self.volume_profile[data_type] = None
        else:
            if not ohlcv_dict["data"][interval].empty:
                window = interval
                group = ohlcv_dict["data"][interval].groupby(
                    getattr(ohlcv_dict["data"][interval].index, frequency.get(interval, "hour"))
                    ).agg(volume_mean=('volume', 'mean'))
                self.volume_profile[data_type][window] = group

        return group

# ############################################################################
# #                                                                          #
# #                            Volatility Analyzer                           #
# #                                                                          #
# ############################################################################            

class VolatilityAnalyzer:
    def __init__(
        self, ohlcv_data: dict = None, window_type :str="medium", debug: bool = False
    ):
        self.ohlcv_market_data = ohlcv_data
        self.window_type = window_type

        self._configure_logging(debug)
        self._init_data()
        self.create_volatility_data()

    def _translate_rolling_window(self, interval):
        if interval == "1s":
            return "secondly"
        elif interval == "1m":
            return "minutely"
        elif interval == "5m":
            return "five_minutely"
        elif interval == "1h":
            return "hourly"
        elif interval == "1d":
            return "daily"
        
    def _configure_logging(self, debug):
        if debug:
            logging.basicConfig(
                format="%(levelname)s (%(asctime)s): %(message)s (Line:%(lineno)d in %(filename)s))",
                datefmt="%Y/%m/%d %I:%M:%S %p",
                level=logging.INFO,
            )

    def _calculate_rolling_window(self, interval, window_type):
        window_sizes = {
            "1s": {"short": 300, "medium": 600, "long": 1200},
            "1m": {"short": 60, "medium": 120, "long": 240},
            "5m": {"short": 36, "medium": 144, "long": 288},
            "1h": {"short": 12, "medium": 24, "long": 48},
            "1d": {"short": 15, "medium": 30, "long": 60},
        }

        interval_windows = window_sizes.get(interval, {})
        return interval_windows.get(window_type, 60)

    def _init_data(self):
        self.ohlcv_data_spot = {}
        self.ohlcv_data_future = {}
        self.ohlcv_data = {"spot": {}, "future": {}}

        self.average_volatilities = {"spot": {}, "future": {}}
        self.last_volatilities = {"spot": {}, "future": {}}

    def create_volatility_data(self, window_type=None):
        window_type = window_type or self.window_type
        self.ohlcv_data["spot"] = (
            self._calculate_volatility(self.ohlcv_market_data["spot"], window_type)
            if self.ohlcv_market_data["spot"] #is not None
            else None
        )
        self.ohlcv_data["future"] = (
            self._calculate_volatility(self.ohlcv_market_data["future"], window_type)
            if self.ohlcv_market_data["future"] #is not None
            else None
        )

        self.process_volatility(self.ohlcv_data["spot"], "spot")
        self.process_volatility(self.ohlcv_data["future"], "future")

    def _calculate_volatility(self, ohlcv_dict, window_type):
        df_dict = {"data": {}, "analytics": {}}

        for key in ohlcv_dict.keys():
            try:
                df = ohlcv_dict[key].copy()
                window = self._calculate_rolling_window(key, window_type)
                logging.info(f"Calculating volatility for {key} interval, window: {window}")
                df["returns"] = np.log(df["close"] / df["close"].shift(1))
                df["volatility"] = df["returns"].rolling(window).std()
                df["lower_band_67"] = df["close"] * (1 - df["volatility"])
                df["upper_band_67"] = df["close"] * (1 + df["volatility"])
                df["lower_band_95"] = df["close"] * (1 - 2 * df["volatility"])
                df["upper_band_95"] = df["close"] * (1 + 2 * df["volatility"])
                df["within_bands_67"] = np.where(
                    (df["close"].shift(-1) > df["lower_band_67"])
                    & (df["close"].shift(-1) < df["upper_band_67"]),
                    1,
                    0,
                )
                df["within_bands_95"] = np.where(
                    (df["close"].shift(-1) > df["lower_band_95"])
                    & (df["close"].shift(-1) < df["upper_band_95"]),
                    1,
                    0,
                )
                #df = df.dropna(subset=["volatility"])

                df_analytics = df.describe()

                df_dict["data"][key] = df
                df_dict["analytics"][key] = df_analytics
            except Exception as e:
                logging.warning(
                    f"Error calculating volatility for {key} interval. Error: {e}"
                )
                continue

        return df_dict

    def process_volatility(self, ohlcv_dict, data_type):
        if ohlcv_dict is None:
            self.average_volatilities[data_type] = None
            self.last_volatilities[data_type] = None
        else:
            for interval in ohlcv_dict["data"].keys():
                if not ohlcv_dict["data"][interval].empty:
                    #window = self._translate_rolling_window(interval)
                    self.average_volatilities[data_type][interval] = ohlcv_dict["analytics"][interval]["volatility"]["mean"]
                    self.last_volatilities[data_type][interval] = ohlcv_dict["data"][interval].iloc[-1]["volatility"]
                    



class ExecutionEstimator:
    def __init__(self, size, ohlcv_data_spot, ohlcv_data_future):
        self.size = size
        self.ohlcv_spot = ohlcv_data_spot
        self.ohlcv_future = ohlcv_data_future
    pass

# ############################################################################
# #                                                                          #
# #                                 Pricer                                   #
# #                                                                          #
# ############################################################################

class Pricer:
    def __init__(
        self,
        exchange,
        symbol,
        future_symbol,
        intervals: list = ["1s", "1m", "5m", "1h", "1d"],
        date_range: dict = {"1s": 1, "1m": 48, "5m": 168, "1h": 720, "1d": 8760},
        risk_aversion="neutral",
        volatility_window="medium",
        execution_costs=3,
        exec_enhancement=3,
        credentials: dict = None,
        debug=False,
    ):
        if debug:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

        self.exchange = exchange
        self.symbol = symbol
        self.future_symbol = future_symbol
        self.intervals = intervals
        self.execution_costs = execution_costs
        self.date_range = date_range
        self.credentials = credentials
        self.volatility_window = volatility_window

        self.fetch_and_calculate()

        self.spot_mid_price = self.market_data.spot_mid_price
        self.future_mid_price = self.market_data.future_mid_price

        self.base_factor = 1
        self.exec_enhancement = exec_enhancement

        if risk_aversion.lower()[0] == "h":
            self.base_factor = 1.1
        elif risk_aversion.lower()[0] == "l":
            self.base_factor = 0.9

        self._liquidity_factor = self.base_factor
        self._volatility_factor = self.base_factor

    def fetch_and_calculate(self):
        self.market_data = MarketData(
            self.exchange,
            self.symbol,
            self.future_symbol,
            intervals=self.intervals,
            date_range=self.date_range,
            credentials=self.credentials,
        )
        self.liquidity_analyzer = LiquidityAnalyzer(
            self.market_data.ob_data_spot,
            self.market_data.ob_data_future,
            self.market_data.ohlcv_data_spot,
            self.market_data.ohlcv_data_future,
        )
        self.volatility_analyzer = VolatilityAnalyzer(
            self.liquidity_analyzer.ohlcv_data_spot["data"],
            self.liquidity_analyzer.ohlcv_data_future["data"],
            window_type=self.volatility_window,
        )

    def reload_data(self):
        self.market_data.fetch_data()
        self.liquidity_analyzer = LiquidityAnalyzer(
            self.market_data.ob_data_spot,
            self.market_data.ob_data_future,
            self.market_data.ohlcv_data_spot,
            self.market_data.ohlcv_data_future,
        )
        self.volatility_analyzer = VolatilityAnalyzer(
            self.liquidity_analyzer.ohlcv_data_spot["data"],
            self.liquidity_analyzer.ohlcv_data_future["data"],
            window_type=self.volatility_window,

        )

        self._update_attributes()

    def _update_attributes(self):
        self.spot_mid_price = self.market_data.spot_mid_price
        self.future_mid_price = self.market_data.future_mid_price

    @property
    def volatility_factor(self):
        return self._volatility_factor

    @volatility_factor.setter
    def volatility_factor(self, value):
        self._volatility_factor = self.base_factor * value

    @property
    def liquidity_factor(self):
        return self._liquidity_factor

    @liquidity_factor.setter
    def liquidity_factor(self, value):
        self._liquidity_factor = self.base_factor * value

    def _vol_parameters(self, tgt_participation=None):
        self.volatility_parameters_dict = {}

        hourly_volatility = self.volatility_analyzer.last_hourly_future_volatility
        avg_hourly_volatility = (
            self.volatility_analyzer.hourly_avg_rolling_future_volatility
        )
        current_hourly_volatility_momentum = hourly_volatility / avg_hourly_volatility
        daily_volatility = self.volatility_analyzer.last_daily_future_volatility
        avg_daily_volatility = (
            self.volatility_analyzer.daily_avg_rolling_future_volatility
        )
        current_daily_volatility_momentum = daily_volatility / avg_daily_volatility

        if current_hourly_volatility_momentum > 1:
            hourly_volatility_multiplier = 1.01
        elif current_hourly_volatility_momentum < 1:
            hourly_volatility_multiplier = 0.99

        if current_daily_volatility_momentum > 1:
            daily_volatility_multiplier = 1.02
        elif current_daily_volatility_momentum < 1:
            daily_volatility_multiplier = 0.98

        self.volatility_parameters_dict["hourly_volatility_multiplier"] = (
            hourly_volatility_multiplier
        )
        self.volatility_parameters_dict["daily_volatility_multiplier"] = (
            daily_volatility_multiplier
        )
        self.volatility_parameters_dict["volatility_multipliers"] = (
            hourly_volatility_multiplier + daily_volatility_multiplier
        ) / 2
        self.volatility_parameters_dict["hourly_volatility"] = hourly_volatility

        return self.volatility_parameters_dict

    def __calculate_distance_from_mid(self, ob_data, amount, column_suffix, quote_type):
        if quote_type == "base":
            column_prefix = "cum"
        elif quote_type == "quote":
            column_prefix = "notional"

        closest_index = (
            ob_data[f"{column_prefix}_{column_suffix}"] - amount
        ).abs().idxmin() + 1
        if closest_index == len(ob_data):
            x = ob_data[[f"{column_prefix}_{column_suffix}"]]
            y = ob_data[f"distance_from_mid_price_{column_suffix}"]
            model = LinearRegression().fit(x, y)
            return model.predict(np.array([[amount]]))[0] / self.exec_enhancement
        else:
            return (
                ob_data.iloc[closest_index][f"distance_from_mid_price_{column_suffix}"]
                / self.exec_enhancement
            )

    def _liquidity_parameters(self, amount, side, quote_type):
        side = side.lower()[0]

        self.liquidity_parameters_dict = {}

        distance_from_mid_ask = 0
        distance_from_mid_bid = 0

        if quote_type == "base":
            order_size_daily_participation = (
                amount / self.liquidity_analyzer.last_24h_spot_volume
            )
            order_size_hourly_participation = (
                amount / self.liquidity_analyzer.last_1h_spot_volume
            )
            current_order_hourly_volume_momentum = (
                self.liquidity_analyzer.last_1h_spot_volume
                / self.liquidity_analyzer.avg_1h_spot_volume
            )
            current_order_daily_volume_momentum = (
                self.liquidity_analyzer.last_24h_spot_volume
                / self.liquidity_analyzer.avg_daily_spot_volume
            )

            if side == "b":
                top_ob_price = self.liquidity_analyzer.ob_data_future[
                    "price_asks"
                ].iloc[0]
                distance_from_mid_ask = self.__calculate_distance_from_mid(
                    self.liquidity_analyzer.ob_data_future, amount, "asks", quote_type
                )
                # closest_index = (self.liquidity_analyzer.ob_data_future["cum_asks"]-amount).abs().idxmin()+1
                # if closest_index == len(self.liquidity_analyzer.ob_data_future):
                #     x = self.liquidity_analyzer.ob_data_future[["cum_asks"]]
                #     y = self.liquidity_analyzer.ob_data_future["distance_from_mid_price_asks"]
                #     model = LinearRegression().fit(x, y)
                #     distance_from_mid_ask = model.predict(np.array([[amount]]))[0]
                # else:
                #     distance_from_mid_ask = self.liquidity_analyzer.ob_data_future.iloc[closest_index]["distance_from_mid_price_asks"]/self.exec_enhancement
            if side == "s":
                top_ob_price = self.liquidity_analyzer.ob_data_future[
                    "price_bids"
                ].iloc[0]
                distance_from_mid_bid = self.__calculate_distance_from_mid(
                    self.liquidity_analyzer.ob_data_future, amount, "bids", quote_type
                )
                # closest_index = (self.liquidity_analyzer.ob_data_future["cum_bids"]-amount).abs().idxmin()+1
                # if closest_index == len(self.liquidity_analyzer.ob_data_future):
                #     x = self.liquidity_analyzer.ob_data_future[["cum_bids"]]
                #     y = self.liquidity_analyzer.ob_data_future["distance_from_mid_price_bids"]
                #     model = LinearRegression().fit(x, y)
                #     distance_from_mid_bid = model.predict(np.array([[amount]]))[0]
                # else:
                #     distance_from_mid_bid = self.liquidity_analyzer.ob_data_future.iloc[closest_index]["distance_from_mid_price_bids"]/self.exec_enhancement

        elif quote_type == "quote":
            order_size_daily_participation = (
                amount / self.liquidity_analyzer.last_24h_spot_volume_ccy
            )
            order_size_hourly_participation = (
                amount / self.liquidity_analyzer.last_1h_spot_volume_ccy
            )
            current_order_hourly_volume_momentum = (
                self.liquidity_analyzer.last_1h_spot_volume
                / self.liquidity_analyzer.avg_1h_spot_volume_ccy
            )
            current_order_daily_volume_momentum = (
                self.liquidity_analyzer.last_24h_spot_volume
                / self.liquidity_analyzer.avg_daily_spot_volume_ccy
            )

            if side == "b":
                top_ob_price = self.liquidity_analyzer.ob_data_future[
                    "price_asks"
                ].iloc[0]
                distance_from_mid_ask = self.__calculate_distance_from_mid(
                    self.liquidity_analyzer.ob_data_future, amount, "asks", quote_type
                )
                # closest_index = (self.liquidity_analyzer.ob_data_future["notional_asks"]-amount).abs().idxmin()+1
                # logging.info(f"Closest Index: {closest_index}")
                # if closest_index == len(self.liquidity_analyzer.ob_data_future):
                #     x = self.liquidity_analyzer.ob_data_future[["notional_asks"]]
                #     y = self.liquidity_analyzer.ob_data_future["distance_from_mid_price_asks"]
                #     model = LinearRegression().fit(x, y)
                #     distance_from_mid_ask = model.predict(np.array([[amount]]))[0]/self.exec_enhancement
                # else:
                #     distance_from_mid_ask = self.liquidity_analyzer.ob_data_future.iloc[closest_index]["distance_from_mid_price_asks"]/self.exec_enhancement
            if side == "s":
                top_ob_price = self.liquidity_analyzer.ob_data_future[
                    "price_bids"
                ].iloc[0]
                distance_from_mid_bid = self.__calculate_distance_from_mid(
                    self.liquidity_analyzer.ob_data_future, amount, "bids", quote_type
                )
                # closest_index = (self.liquidity_analyzer.ob_data_future["notional_bids"]-amount).abs().idxmin()+1
                # if closest_index == len(self.liquidity_analyzer.ob_data_future):
                #     x = self.liquidity_analyzer.ob_data_future[["notional_bids"]]
                #     y = self.liquidity_analyzer.ob_data_future["distance_from_mid_price_bids"]
                #     model = LinearRegression().fit(x, y)
                #     distance_from_mid_bid = model.predict(np.array([[amount]]))[0]/self.exec_enhancement
                # else:
                #     distance_from_mid_bid = self.liquidity_analyzer.ob_data_future.iloc[closest_index]["distance_from_mid_price_bids"]/self.exec_enhancement

        if current_order_hourly_volume_momentum > 1:
            hourly_liquidity_multiplier = 0.98
        elif current_order_hourly_volume_momentum < 1:
            hourly_liquidity_multiplier = 1.02

        if current_order_daily_volume_momentum > 1:
            daily_liquidity_multiplier = 0.995
        elif current_order_daily_volume_momentum < 1:
            daily_liquidity_multiplier = 1.005

        self.liquidity_parameters_dict["amount"] = amount
        self.liquidity_parameters_dict["order_size_daily_participation_multiplier"] = (
            order_size_daily_participation
        )
        self.liquidity_parameters_dict["order_size_hourly_participation_multiplier"] = (
            order_size_hourly_participation
        )
        self.liquidity_parameters_dict["hourly_liquidity_multiplier"] = (
            hourly_liquidity_multiplier
        )
        self.liquidity_parameters_dict["daily_liquidity_multiplier"] = (
            daily_liquidity_multiplier
        )
        self.liquidity_parameters_dict["distance_from_mid_ask"] = distance_from_mid_ask
        self.liquidity_parameters_dict["distance_from_mid_bid"] = distance_from_mid_bid
        self.liquidity_parameters_dict["liquidity_multipliers"] = (
            hourly_liquidity_multiplier + daily_liquidity_multiplier
        ) / 2
        self.liquidity_parameters_dict["top_ob_price"] = top_ob_price

        return self.liquidity_parameters_dict

    def _calculate_market_depth_metric(
        self, amount, asset_type, weights={"1m": 0.3, "5m": 0.4, "1d": 0.3}
    ):
        """
        Calculate the market depth metric for a given amount and asset type (spot or future)
        """

        if asset_type == "spot":
            for key in self.liquidity_analyzer.last_volumes["spot"]:
                pass
        return

    def _ask_price_function(self, amount, side, quote_type):
        self._vol_factors = self._vol_parameters()
        self._liquidity_factors = self._liquidity_parameters(amount, side, quote_type)
        hourly_volatility = self.volatility_parameters_dict["hourly_volatility"]
        volatility_multipliers = self.volatility_parameters_dict[
            "volatility_multipliers"
        ]
        liquidity_multipliers = self.liquidity_parameters_dict["liquidity_multipliers"]
        distance_from_mid = self.liquidity_parameters_dict["distance_from_mid_ask"]
        order_size_hourly_participation_multiplier = self.liquidity_parameters_dict[
            "order_size_hourly_participation_multiplier"
        ]
        top_ob_price = self.liquidity_parameters_dict["top_ob_price"]

        self.ask_price = top_ob_price * (
            1
            + (
                (distance_from_mid / 10000)
                * liquidity_multipliers
                * volatility_multipliers
                * self._liquidity_factor
            )
            + self.execution_costs * 3 / 10000
            + hourly_volatility * order_size_hourly_participation_multiplier
        )

        return self.ask_price

    def _bid_price_function(self, amount, side, quote_type):
        self._vol_factors = self._vol_parameters()
        self._liquidity_factors = self._liquidity_parameters(amount, side, quote_type)
        hourly_volatility = self.volatility_parameters_dict["hourly_volatility"]
        volatility_multipliers = self.volatility_parameters_dict[
            "volatility_multipliers"
        ]
        liquidity_multipliers = self.liquidity_parameters_dict["liquidity_multipliers"]
        distance_from_mid = self.liquidity_parameters_dict["distance_from_mid_bid"]
        order_size_hourly_participation_multiplier = self.liquidity_parameters_dict[
            "order_size_hourly_participation_multiplier"
        ]
        top_ob_price = self.liquidity_parameters_dict["top_ob_price"]

        self.bid_price = top_ob_price * (
            1
            - (
                (distance_from_mid / 10000)
                * liquidity_multipliers
                * volatility_multipliers
                * self._liquidity_factor
            )
            + self.execution_costs * 3 / 10000
            + hourly_volatility * order_size_hourly_participation_multiplier
        )

        return self.bid_price

    def calculate_price(
        self,
        side: str = "two-way",
        base_amount: float = None,
        quote_amount: float = None,
    ):
        if base_amount is not None:
            amount = base_amount
            quote_type = "base"
        elif quote_amount is not None:
            amount = quote_amount
            quote_type = "quote"
        elif base_amount is None and quote_amount is None:
            raise ValueError("Either base_amount or quote_amount must be provided")
        elif base_amount is not None and quote_amount is not None:
            raise ValueError("Only base_amount OR quote_amount should provided")

        prices_dict = {"ask": {"amount": amount}, "bid": {"amount": amount}}

        if side.lower()[0] == "b":
            ask = self._ask_price_function(amount, side, quote_type)
            prices_dict["ask"]["price"] = ask
            prices_dict["ask"]["distance_from_mid"] = (
                ask / self.spot_mid_price - 1
            ) * 10000
        elif side.lower()[0] == "s":
            bid = self._bid_price_function(amount, side, quote_type)
            prices_dict["bid"]["price"] = bid
            prices_dict["bid"]["distance_from_mid"] = (
                bid / self.spot_mid_price - 1
            ) * 10000
        else: #two-way
            ask = self._ask_price_function(amount, "buy", quote_type)
            bid = self._bid_price_function(amount, "sell", quote_type)
            prices_dict["ask"]["price"] = ask
            prices_dict["bid"]["price"] = bid
            prices_dict["ask"]["distance_from_mid"] = (
                ask / self.spot_mid_price - 1
            ) * 10000
            prices_dict["bid"]["distance_from_mid"] = (
                bid / self.spot_mid_price - 1
            ) * 10000
            prices_dict["bid_ask_spread"] = (
                prices_dict["ask"]["distance_from_mid"]
                + prices_dict["bid"]["distance_from_mid"]
            )

        return prices_dict
