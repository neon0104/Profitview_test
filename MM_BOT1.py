from profitview import Link, http, logger

import json
import numpy as np
import pandas as pd
import scipy
import talib
import threading
import time

TIME_LOOKUP = {
    '1m': 60_000,
    '15m': 60_000 * 15,
    '1h': 60_000 * 60,
    '1d': 60_000 * 60 * 24,
}

# GARCH parameters
OMEGA = 1.605025e-08
ALPHA = 0.193613
BETA = 0.786155


def debounce(wait):
    """Postpone a function execution until after wait seconds
    have elapsed since the last time it was invoked.
    source: https://gist.github.com/walkermatt/2871026"""

    def decorator(func):
        def debounced(*args, **kwargs):
            def call_func():
                debounced.last_call = time.time()
                func(*args, **kwargs)

            if hasattr(debounced, 'timer'):
                debounced.timer.cancel()

            if time.time() - getattr(debounced, 'last_call', 0) > wait:
                call_func()
            else:
                debounced.timer = threading.Timer(wait, call_func)
                debounced.timer.start()

        return debounced

    return decorator


class Trading(Link):

    def __init__(self):
        super().__init__()
        # ALGO PARAMS
        self.src = 'bitmex'  # exchange name
        self.venue = 'BitMEX'  # Ensure this is the correct venue name
        self.sym = 'XBTUSDT'  # symbol we will trade
        self.level = '1m'  # OHLC candle granularity
        self.lookback = 150  # lookback period of close prices
        self.time_step = TIME_LOOKUP[self.level]  # time step in milliseconds

        # ALGO STRATEGY STATE
        self.closes = dict()  # time bin -> close price
        self.macd = dict(hist=np.nan, slope=np.nan)  # MACD histogram and slope value
        self.var_t1 = np.nan  # t-1 variance forecast
        self.tob = (np.nan, np.nan)  # top of book [bid, ask] price
        self.mid = np.nan  # current mid price
        self.risk = 0  # current position risk for sym
        self.orders = dict(bid={}, ask={})  # current open limit orders

        # ALGO PARAMS
        self.skew_damp = 2  # skew dampening
        self.max_risk = 1600000  # max position risk limit
        self.min_spread = 1e-4
        self.sharpe_target = 1.2  # target sharpe ratio
        self.fee_cost = -0.0002  # cost of entering and exiting position

        self.size_precision = 1000  # min step for size
        self.price_precision = 0.5  # min step for price
        self.price_decimals = 1  # decimals to round off for price

        # RUN ON STARTUP
        self.on_startup()

    @property
    def orders_intent(self):
        # 필요한 데이터가 초기화되었는지 확인
        if np.isnan(self.mid) or np.isnan(self.var_t1) or np.isnan(self.tob[0]) or np.isnan(self.tob[1]):
            logger.error(
                f"Essential market data not initialized properly: mid={self.mid}, var_t1={self.var_t1}, tob={self.tob}")
            return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

        tob_bid, tob_ask = self.tob
        half_spread = 0.5 * self.mid * self.spread
        skew = np.clip(self.macd['slope'] / self.skew_damp, -1, 1)

        bid = self.mid - half_spread * (1 - skew)
        ask = self.mid + half_spread * (1 + skew)

        if np.isnan(bid) or np.isnan(ask):
            logger.error("Calculated bid or ask is NaN, cannot proceed with orders.")
            return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

        bid = np.min([tob_bid, self.round_value(bid, self.price_precision)])
        ask = np.max([tob_ask, self.round_value(ask, self.price_precision)])

        if bid <= 0 or ask <= 0:
            logger.error("Calculated bid or ask is non-positive, cannot proceed with orders.")
            return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

        available_balance = self.get_available_balance(self.venue, currency='USDT')
        max_affordable_size = available_balance / bid if bid > 0 else 0

        if max_affordable_size <= 0:
            logger.error("Calculated max affordable size is non-positive, cannot proceed with orders.")
            return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

        bsize = np.clip(self.max_risk - self.risk, 0, max_affordable_size) if self.risk < self.max_risk else 0
        asize = np.clip(self.max_risk + self.risk, 0,
                        max_affordable_size) if self.risk < self.max_risk else max_affordable_size

        bsize = self.round_value(bsize, self.size_precision)
        asize = self.round_value(asize, self.size_precision)

        if np.isnan(bsize) or np.isnan(asize):
            logger.error("Invalid calculated order sizes, cannot proceed with orders.")
            return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

        logger.info('\n' + json.dumps({'bid': [bid, bsize], 'ask': [ask, asize]}))
        return {'bid': [bid, bsize], 'ask': [ask, asize]}

    @property
    def spread(self):
        norm_std = np.exp(np.sqrt(self.garch_var())) - 1
        return np.max([self.min_spread, self.sharpe_target * norm_std + self.fee_cost]) if not np.isnan(
            norm_std) else np.nan

    def minutely_update(self):
        # Ensure we have enough closes before attempting updates
        if len(self.closes) >= self.lookback:
            if self.time_bin_now not in self.closes:
                self.closes[self.time_bin_now] = self.last_closes[-1]
                self.update_signal()

            # update var_t1 using (t-2, t-1) closes
            self.var_t1 = self.garch_var(lookback=1)
        else:
            logger.info("Not enough data to perform updates.")

        threading.Timer(61 - self.second, self.minutely_update).start()

    @property
    def last_closes(self):
        start_time = self.time_bin_now - self.lookback * self.time_step
        times = [start_time + (i + 1) * self.time_step for i in range(self.lookback)]
        closes = [self.closes.get(x, np.nan) for x in times]

        # Ensure that there are no NaN values if possible, otherwise use a default value
        closes = pd.Series(closes).ffill().bfill().values
        return np.array(closes)

    def update_signal(self):
        closes = self.last_closes
        if not any(np.isnan(closes)):
            macd, signal, hist = talib.MACD(closes)
            N = 10  # less points required to interpolate accurately
            try:
                cubic = scipy.interpolate.CubicSpline(range(N), hist[-N:])
                self.macd['hist'] = hist[-1]
                self.macd['slope'] = float(cubic(N - 1, 1))

            except Exception as e:
                logger.error(f'unable to update signal - {e}', exc_info=True)

        else:
            logger.info(f'unable to compute signal as closes not initialized properly: {closes}')

    def update_close(self, data):
        time_bin = self.candle_bin(data['time'], self.level)
        self.closes[time_bin] = data['price']

    def round_value(self, x, tick):
        return tick * np.round(x / tick)

    def get_available_balance(self, venue, currency='USDT'):
        # Implement the method to fetch available balance
        # This is a placeholder implementation
        return 1000.0  # Example: returns a balance of 1000 USDT

    @property
    def orders_intent(self):

    # 필요한 데이터가 초기화되었는지 확인
    if np.isnan(self.mid) or np.isnan(self.var_t1) or np.isnan(self.tob[0]) or np.isnan(self.tob[1]):
        logger.error(
            f"Essential market data not initialized properly: mid={self.mid}, var_t1={self.var_t1}, tob={self.tob}")
        return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

    tob_bid, tob_ask = self.tob
    half_spread = 0.5 * self.mid * self.spread
    skew = np.clip(self.macd['slope'] / self.skew_damp, -1, 1)

    bid = self.mid - half_spread * (1 - skew)
    ask = self.mid + half_spread * (1 + skew)

    if np.isnan(bid) or np.isnan(ask):
        logger.error("Calculated bid or ask is NaN, cannot proceed with orders.")
        return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

    bid = np.min([tob_bid, self.round_value(bid, self.price_precision)])
    ask = np.max([tob_ask, self.round_value(ask, self.price_precision)])

    if bid <= 0 or ask <= 0:
        logger.error("Calculated bid or ask is non-positive, cannot proceed with orders.")
        return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

    available_balance = self.get_available_balance(self.venue, currency='USDT')
    max_affordable_size = available_balance / bid if bid > 0 else 0

    if max_affordable_size <= 0:
        logger.error("Calculated max affordable size is non-positive, cannot proceed with orders.")
        return {'bid': [np.nan, np.nan], 'ask': [np.nan, np.nan]}

    bsize = np.clip(self.max_risk - self.risk, 0, max_affordable_size) if self.risk < self.max_risk else 0
    asize = np.clip(self.max_risk + self.risk, 0,
                    max_affordable_size) if self.risk < self.max_risk else max_affordable_size

    bsize = self.round_value(bsize, self.size_precision)
    asize = self.round_value(asize, self.size_precision)

    if np.isnan(bsize) or np.isnan(asize) or b

    @debounce(1)
    def update_limit_orders(self):
        intent = self.orders_intent
        bid, bsize = intent['bid']
        ask, asize = intent['ask']

        # NaN 또는 비정상적인 값이 있는지 체크
        if np.isnan(bid) or np.isnan(ask) or bid <= 0 or ask <= 0 or bsize <= 0 or asize <= 0:
            logger.error(f"Invalid bid, ask, or size detected: bid={bid}, ask={ask}, bsize={bsize}, asize={asize}.")
            return

        available_balance = self.get_available_balance(self.venue, currency='USDT')
        if available_balance < bsize * bid:
            logger.error(f"Insufficient balance: {available_balance} USDT available, {bsize * bid} USDT required.")
            return

        log_msg = {
            'hist': np.round(self.macd['hist'], 4),
            'slope': np.round(self.macd['slope'], 2),
            'spread': np.round(1e4 * self.spread, 1),
            'bid': [bid, bsize],
            'ask': [ask, asize]
        }

        if not any(np.isnan((bid, ask, asize, bsize))):
            cancels, updates, inserts = [], [], []

            # call update or insert API for closest side first
            shift = int(np.log(self.mid / bid) > np.log(ask / self.mid))

            for key in np.roll(['bid', 'ask'], shift):
                sort_key = lambda x: (1 if key == 'ask' else -1) * x['order_price']
                orders = sorted(self.orders[key].values(), key=sort_key)

                price, size = intent[key]
                side = 'Sell' if key == 'ask' else 'Buy'

                if len(orders) > 0:
                    update = dict()
                    top, *remain = orders

                    if price != top['order_price']:
                        update['price'] = price
                    if size != top['remain_size']:
                        update['size'] = size

                    if len(update) > 0:
                        update['order_id'] = top['order_id']
                        cancels.append(update['order_id'])
                        inserts.append({'sym': self.sym, 'side': side, 'size': size, 'price': price})

                    cancels.extend([x['order_id'] for x in remain])

                elif size > 0:
                    inserts.append({'sym': self.sym, 'side': side, 'size': size, 'price': price})

            for order_id in cancels:
                try:
                    for x in self.cancel_order(self.venue, order_id=order_id)['data']:
                        key = 'bid' if x['side'] == 'Buy' else 'ask'
                        self.orders[key].pop(x['order_id'], None)
                except Exception as e:
                    logger.error(f"Failed to cancel order: {e}")

            for insert in inserts:
                try:
                    data = self.create_limit_order(self.venue, **insert)['data']
                    key = 'bid' if data['side'] == 'Buy' else 'ask'
                    self.orders[key][data['order_id']] = data
                except KeyError as e:
                    logger.error(f"Failed to create limit order: Missing key in response: {e}")
                except Exception as e:
                    logger.error(f"Failed to create limit order: {e}")

        logger.info('\n' + json.dumps(log_msg))

    # TRADING EVENTS
    def order_update(self, src, sym, data):
        if sym == self.sym:
            if data.get('remain_size') == 0:
                key = 'bid' if data['side'] == 'Buy' else 'ask'
                self.orders[key].pop(data['order_id'], None)

    def fill_update(self, src, sym, data):
        if sym == self.sym:
            sign = 1 if data['side'] == 'Buy' else -1
            self.risk = round(self.risk + sign * data['fill_size'])
            self.update_limit_orders()  # update limit orders on risk change

    def quote_update(self, src, sym, data):
        if sym == self.sym:
            self.tob = (data['bid'][0], data['ask'][0])
            if (mid := np.mean(self.tob)) != self.mid:
                self.mid = mid
                self.update_limit_orders()

    def trade_update(self, src, sym, data):
        if sym == self.sym:
            self.update_close(data)
            self.update_signal()

    # WEBHOOKS
    @http.route
    def get_state(self, data):
        return {k: getattr(self, k, None) for k in data.get('keys', [])}

    @http.route
    def post_state(self, data):
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self.get_state({'keys': list(data)})
