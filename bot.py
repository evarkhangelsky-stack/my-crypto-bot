import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import telebot

class TechnicalIndicators:
    """Собственные технические индикаторы"""

    @staticmethod
    def vwap(high, low, close, volume):
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(close, period):
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di


class BybitScalpingBot:
    def __init__(self):
        # API keys from environment
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')  # auth_token
        self.cryptopanic_api_plan = os.getenv('CRYPTOPANIC_API_PLAN', 'developer')  # default 'developer'

        required = [self.api_key, self.api_secret, self.telegram_token, self.telegram_chat_id]
        if not all(required):
            raise ValueError(
                "Missing required environment variables: "
                "BYBIT_API_KEY, BYBIT_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
            )

        # Initialize Bybit (linear futures)
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        # Настройка margin mode и leverage (добавлено)
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        for symbol in self.symbols:
            try:
                self.exchange.set_margin_mode('cross', symbol)  # 'cross' по умолчанию, или 'isolated'
                self.exchange.set_leverage(5, symbol)  # x5 плечо
                print(f"[{datetime.now()}] Leverage set to 5x and cross margin for {symbol}")
            except Exception as e:
                print(f"Error setting leverage/margin for {symbol}: {e}")

        # Initialize Telegram
        self.bot = telebot.TeleBot(self.telegram_token)

        # Trading parameters
        self.timeframe = '5m'
        self.positions = {symbol: None for symbol in self.symbols}

        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5
        self.taker_fee = 0.0006  # 0.06%

        print(f"[{datetime.now()}] Bot initialized for {self.symbols}")
        self.send_telegram(f"Bot started\nSymbols: {' '.join(self.symbols)}\nTimeframe: {self.timeframe}")

    def send_telegram(self, message):
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Telegram error: {e}")

    def fetch_ohlcv(self, symbol, limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"[{datetime.now()}] OHLCV fetched successfully for {symbol}, rows: {len(df)}")  # Отладка
            return df
        except Exception as e:
            print(f"[{datetime.now()}] Error fetching OHLCV for {symbol}: {e}")  # Отладка ошибки
            return None

    def fetch_orderbook_data(self, symbol):
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=50)
            total_bids = sum(bid[1] for bid in orderbook['bids'])
            total_asks = sum(ask[1] for ask in orderbook['asks'])
            total = total_bids + total_asks
            bid_ratio = (total_bids / total) * 100 if total > 0 else 50
            print(f"[{datetime.now()}] Orderbook fetched for {symbol}, bid_ratio: {bid_ratio:.2f}%")  # Отладка
            return {'bid_ratio': bid_ratio, 'total_volume': total}
        except Exception as e:
            print(f"[{datetime.now()}] Error fetching orderbook for {symbol}: {e}")
            return {'bid_ratio': 50, 'total_volume': 0}

    def fetch_coinglass_data(self, symbol_base):
        if not self.coinglass_api_key:
            return {}
        try:
            headers = {'cg-api-key': self.coinglass_api_key}
            url = f"https://open-api.coinglass.com/public/v2/long_short?symbol={symbol_base}&time_type=h1"
            res = requests.get(url, headers=headers, timeout=10).json()
            return res.get('data', [])[0] if res.get('success') else {}
        except Exception as e:
            print(f"Coinglass error: {e}")
            return {}

    def fetch_cryptopanic_news(self):
        if not self.cryptopanic_api_key:
            return []
        try:
            url = f"https://cryptopanic.com/api/{self.cryptopanic_api_plan}/v2/posts/?auth_token={self.cryptopanic_api_key}&kind=news"
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                print(f"[{datetime.now()}] CryptoPanic HTTP error: {res.status_code} - {res.text}")
                return []
            data = res.json()
            return data.get('results', [])[:5]
        except Exception as e:
            print(f"[{datetime.now()}] CryptoPanic error: {e}")
            return []

    def calculate_indicators(self, df):
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        adx, di_plus, di_minus = TechnicalIndicators.adx(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'], period=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=14)
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        print(f"[{datetime.now()}] Indicators calculated")  # Отладка
        return df

    def get_ai_filter(self, symbol, df, signal, orderbook, coinglass, news):
        if not self.deepseek_api_key:
            return True
        try:
            last = df.iloc[-1]
            news_text = "\n".join(n.get('title', '') for n in news)
            prompt = f"""Analyze trading signal for {symbol}:
Signal: {signal}
Price: {last['close']}
RSI: {last['rsi']:.2f}, ADX: {last['adx']:.2f}
Orderbook Bid Ratio: {orderbook['bid_ratio']:.2f}%
Coinglass L/S Ratio: {coinglass.get('longShortRatio', 'N/A')}
Recent News: {news_text}

Reply with ONLY "YES" or "NO" if this trade is high probability."""
            res = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1
                },
                timeout=15
            ).json()
            answer = res['choices'][0]['message']['content'].strip().upper()
            print(f"[{datetime.now()}] AI filter: {answer}")  # Отладка
            return "YES" in answer
        except Exception as e:
            print(f"[{datetime.now()}] AI error: {e}")
            return True

    def detect_signal(self, symbol, df):
        last = df.iloc[-1]
        price = last['close']
        rsi = last['rsi']
        adx = last['adx']
        vwap = last['vwap']
        atr = last['atr']
        bb_upper = last['bb_upper']
        bb_lower = last['bb_lower']
        ema_20 = last['ema_20']
        ema_50 = last['ema_50']

        print(f"[{datetime.now()}] {symbol} values: Price={price:.2f}, RSI={rsi:.2f}, ADX={adx:.2f}, VWAP={vwap:.2f}, EMA20={ema_20:.2f}, EMA50={ema_50:.2f}, ATR={atr:.2f}, BB Upper={bb_upper:.2f}, BB Lower={bb_lower:.2f}")  # Отладка значений

        if pd.isna([price, rsi, adx, vwap, atr]).any():
            print(f"[{datetime.now()}] NaN in indicators for {symbol} — no signal")  # Отладка
            return None, None, None

        ob = self.fetch_orderbook_data(symbol)
        bid_ratio = ob['bid_ratio']

        signal = None
        if adx < 25:  # Sideways
            if price <= bb_lower and rsi < 30 and bid_ratio > 60:
                signal = 'LONG'
            elif price >= bb_upper and rsi > 70 and bid_ratio < 40:
                signal = 'SHORT'
            print(f"[{datetime.now()}] Sideways check: Price vs BB Lower={price <= bb_lower}, RSI<30={rsi < 30}, Bid>60={bid_ratio > 60}; Price vs BB Upper={price >= bb_upper}, RSI>70={rsi > 70}, Bid<40={bid_ratio < 40}")  # Отладка условий
        else:  # Trending
            if price > vwap and ema_20 > ema_50 and rsi > 40 and bid_ratio > 60:
                signal = 'LONG'
            elif price < vwap and ema_20 < ema_50 and rsi < 60 and bid_ratio < 40:
                signal = 'SHORT'
            print(f"[{datetime.now()}] Trending check: Price>VWAP={price > vwap}, EMA20>EMA50={ema_20 > ema_50}, RSI>40={rsi > 40}, Bid>60={bid_ratio > 60}; Price<VWAP={price < vwap}, EMA20<EMA50={ema_20 < ema_50}, RSI<60={rsi < 60}, Bid<40={bid_ratio < 40}")  # Отладка условий

        if signal:
            base = symbol.split('/')[0]
            cg = self.fetch_coinglass_data(base)
            news = self.fetch_cryptopanic_news()

            if not self.get_ai_filter(symbol, df, signal, ob, cg, news):
                print(f"[{datetime.now()}] AI filter rejected signal for {symbol}")  # Отладка
                return None, None, None

            entry = price
            fee_adj = price * self.taker_fee
            if signal == 'LONG':
                sl = entry - (self.sl_atr_multiplier * atr) - fee_adj
                tp = entry + (self.tp_atr_multiplier * atr) + fee_adj
            else:
                sl = entry + (self.sl_atr_multiplier * atr) + fee_adj
                tp = entry - (self.tp_atr_multiplier * atr) - fee_adj

            print(f"[{datetime.now()}] Signal detected: {signal} for {symbol}")  # Отладка сигнала
            return signal, "Scalp", {'entry': entry, 'stop_loss': sl, 'take_profit': tp}

        print(f"[{datetime.now()}] No signal for {symbol}")  # Отладка отсутствия сигнала
        return None, None, None

    def place_order(self, symbol, signal, params):
        try:
            balance = self.get_balance()
            risk = balance * 0.01  # 1% risk per trade
            size = risk / abs(params['entry'] - params['stop_loss'])
            size = round(size, 3)

            msg = (
                f"📉 *Signal: {symbol}*\n"
                f"{signal} ({params['entry']:.2f})\n"
                f"SL: {params['stop_loss']:.2f}\n"
                f"TP: {params['take_profit']:.2f}"
            )
            self.send_telegram(msg)

            # Реальные ордера (market)
            if signal == 'LONG':
                order = self.exchange.create_market_buy_order(symbol, size)
            else:
                order = self.exchange.create_market_sell_order(symbol, size)

            # Если есть average price — обновляем entry
            actual_entry = order.get('average') or params['entry']
            params['entry'] = actual_entry

            self.positions[symbol] = {
                'side': signal,
                'entry': params['entry'],
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'size': size,
                'trailing_stop_activated': False
            }
            print(f"[{datetime.now()}] Order placed: {signal} {size} for {symbol}")  # Отладка ордера

        except Exception as e:
            print(f"[{datetime.now()}] Order error for {symbol}: {e}")

    def get_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            print(f"[{datetime.now()}] Полный ответ fetch_balance: {bal}")
            if 'USDT' in bal and 'free' in bal['USDT']:
                usdt_free = float(bal['USDT']['free'])
                print(f"[{datetime.now()}] USDT free balance: {usdt_free}")
                return usdt_free
            else:
                print(f"[{datetime.now()}] USDT не найден в ответе баланса")
                return 0.0  # Теперь fallback 0, чтобы не торговать
        except Exception as e:
            print(f"[{datetime.now()}] BALANCE FETCH FAILED: {str(e)}")
            return 0.0  # fallback 0

    def manage_position(self, symbol, df):
        pos = self.positions.get(symbol)
        if not pos:
            return

        curr = df.iloc[-1]['close']
        side = pos['side']
        entry = pos['entry']
        sl = pos['stop_loss']
        tp = pos['take_profit']

        if side == 'LONG':
            pnl_pct = ((curr - entry) / entry) * 100
        else:
            pnl_pct = ((entry - curr) / entry) * 100

        if (side == 'LONG' and curr <= sl) or (side == 'SHORT' and curr >= sl):
            self.close_position(symbol, curr, 'SL Hit')
        elif (side == 'LONG' and curr >= tp) or (side == 'SHORT' and curr <= tp):
            self.close_position(symbol, curr, 'TP Hit')
        elif pnl_pct > self.trailing_stop_percent and not pos['trailing_stop_activated']:
            pos['stop_loss'] = entry  # breakeven
            pos['trailing_stop_activated'] = True
            self.send_telegram(f'Trailing: {symbol} to Breakeven')

        print(f"[{datetime.now()}] Position checked for {symbol}, PNL %: {pnl_pct:.2f}")  # Отладка позиции

    def close_position(self, symbol, price, reason):
        pos = self.positions.get(symbol)
        if not pos:
            return

        if pos['side'] == 'LONG':
            pnl = (price - pos['entry']) * pos['size']
        else:
            pnl = (pos['entry'] - price) * pos['size']

        self.send_telegram(
            f"Closed {symbol}\n"
            f"Reason: {reason}\n"
            f"P&L: ${pnl:.2f}"
        )

        # Реальное закрытие позиции
        try:
            if pos['side'] == 'LONG':
                self.exchange.create_market_sell_order(symbol, pos['size'])
            else:
                self.exchange.create_market_buy_order(symbol, pos['size'])
            print(f"[{datetime.now()}] Position closed for {symbol}: {reason}")  # Отладка закрытия
        except Exception as e:
            print(f"[{datetime.now()}] Close order error for {symbol}: {e}")

        self.positions[symbol] = None

    def run(self):
        while True:
            print(f"[{datetime.now()}] Starting new cycle")  # Отладка начала цикла
            for symbol in self.symbols:
                try:
                    df = self.fetch_ohlcv(symbol)
                    if df is None:
                        print(f"[{datetime.now()}] Skipping {symbol} - no data")  # Отладка скипа
                        continue
                    df = self.calculate_indicators(df)

                    if self.positions.get(symbol):
                        self.manage_position(symbol, df)
                    else:
                        signal, s_type, params = self.detect_signal(symbol, df)
                        if signal:
                            self.place_order(symbol, signal, params)
                except Exception as e:
                    print(f"[{datetime.now()}] Error for {symbol}: {e}")
            print(f"[{datetime.now()}] Cycle finished, sleeping 30s")  # Отладка конца цикла
            time.sleep(30)  # защита от rate limit


if __name__ == "__main__":
    bot = BybitScalpingBot()
    bot.run()
