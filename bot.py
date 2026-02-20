import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import telebot
import csv

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

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=d_period).mean()
        return k_smooth, d

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


class BybitScalpingBot:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.cryptopanic_api_plan = os.getenv('CRYPTOPANIC_API_PLAN', 'developer')

        required = [self.api_key, self.api_secret, self.telegram_token, self.telegram_chat_id]
        if not all(required):
            raise ValueError("Missing required environment variables")

        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        for symbol in self.symbols:
            try:
                self.exchange.set_margin_mode('cross', symbol)
                self.exchange.set_leverage(5, symbol)
                print(f"[{datetime.now(timezone.utc)}] Leverage 5x and cross for {symbol}")
            except Exception as e:
                print(f"Error setting leverage/margin: {e}")

        self.bot = telebot.TeleBot(self.telegram_token)
        self.timeframe = '5m'
        self.positions = {s: None for s in self.symbols}

        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5
        self.taker_fee = 0.0006

        self.daily_loss_limit_pct = -4.2
        self.last_day = None
        self.day_start_equity = None
        self.trading_paused_until = None

        # CSV для лога сделок
        self.trade_log_file = "trade_log.csv"
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'entry', 'exit', 'size', 'pnl', 'pnl_pct',
                    'rsi', 'adx', 'vwap', 'ema_20', 'ema_50', 'atr', 'bb_upper', 'bb_lower',
                    'stoch_k', 'stoch_d', 'macd_hist', 'bid_ratio'
                ])

        print(f"[{datetime.now(timezone.utc)}] Bot initialized for {self.symbols}")
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
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_orderbook_data(self, symbol):
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=50)
            total_bids = sum(bid[1] for bid in orderbook['bids'])
            total_asks = sum(ask[1] for ask in orderbook['asks'])
            total = total_bids + total_asks
            bid_ratio = (total_bids / total) * 100 if total > 0 else 50
            return {'bid_ratio': bid_ratio, 'total_volume': total}
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {'bid_ratio': 50, 'total_volume': 0}

    def calculate_indicators(self, df):
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=14)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'], period=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        adx, di_plus, di_minus = TechnicalIndicators.adx(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        df['stoch_k'], df['stoch_d'] = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.macd(df['close'])
        return df

    def check_daily_loss_limit(self):
        now = datetime.now(timezone.utc)
        current_day = now.date()

        if self.last_day != current_day:
            try:
                bal = self.exchange.fetch_balance()
                equity = float(bal['info']['result']['list'][0]['totalEquity'])
                self.day_start_equity = equity
                self.last_day = current_day
                self.trading_paused_until = None
                print(f"[{now}] Новый день UTC. Депозит на начало: {equity:.2f} USDT")
                self.send_telegram(f"Новый день UTC. Баланс на старте: {equity:.2f} USDT")
            except Exception as e:
                print(f"Не удалось получить equity для лимита: {e}")
                return True

        if self.trading_paused_until and now < self.trading_paused_until:
            print(f"[{now}] Торговля остановлена до {self.trading_paused_until} из-за лимита убытков")
            return False

        if self.day_start_equity is None:
            return True

        try:
            bal = self.exchange.fetch_balance()
            current_equity = float(bal['info']['result']['list'][0]['totalEquity'])
            pnl_pct = (current_equity - self.day_start_equity) / self.day_start_equity * 100
            print(f"[{now}] Текущий PnL дня: {pnl_pct:.2f}% (начало: {self.day_start_equity:.2f}, сейчас: {current_equity:.2f})")

            if pnl_pct <= self.daily_loss_limit_pct:
                self.trading_paused_until = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                msg = f"🚨 Дневной лимит убытков -{self.daily_loss_limit_pct}% достигнут! Торговля остановлена до {self.trading_paused_until.strftime('%Y-%m-%d %H:%M UTC')}"
                print(msg)
                self.send_telegram(msg)
                return False
            return True
        except Exception as e:
            print(f"Ошибка проверки лимита: {e}")
            return True

    def sideways_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        rsi = last['rsi']
        stoch_k = last['stoch_k']
        macd_hist = last['macd_hist']
        bb_lower = last['bb_lower']
        bb_upper = last['bb_upper']
        bid_ratio = ob['bid_ratio']

        if price <= bb_lower and rsi < 35 and stoch_k < 20 and bid_ratio > 55:
            strength = 0.9 if rsi < 30 and stoch_k < 15 and bid_ratio > 65 else 0.6
            return 'LONG', strength
        if price >= bb_upper and rsi > 65 and macd_hist > 0 and bid_ratio < 45:
            strength = 0.9 if rsi > 70 and macd_hist > 0.5 and bid_ratio < 35 else 0.6
            return 'SHORT', strength
        return None, 0

    def trend_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        vwap = last['vwap']
        ema20 = last['ema_20']
        ema50 = last['ema_50']
        rsi = last['rsi']
        bid_ratio = ob['bid_ratio']

        if price > vwap and ema20 > ema50 and rsi > 35 and bid_ratio > 55:
            strength = 0.9 if rsi > 45 and bid_ratio > 65 else 0.6
            return 'LONG', strength
        if price < vwap and ema20 < ema50 and rsi < 65 and bid_ratio < 45:
            strength = 0.9 if rsi < 55 and bid_ratio < 35 else 0.6
            return 'SHORT', strength
        return None, 0

    def detect_signal(self, symbol, df):
        if not self.check_daily_loss_limit():
            return None, None, None

        last = df.iloc[-1]
        adx = last['adx']
        ob = self.fetch_orderbook_data(symbol)

        side_sig, side_strength = self.sideways_strategy(df, ob)
        trend_sig, trend_strength = self.trend_strategy(df, ob)

        final_signal = None
        final_strength = 0

        if adx < 25:
            if side_sig:
                final_signal = side_sig
                final_strength = side_strength
            elif trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength * 0.6
        elif adx > 30:
            if trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength
            elif side_sig:
                final_signal = side_sig
                final_strength = side_strength * 0.6
        else:
            if side_strength > trend_strength:
                final_signal = side_sig
                final_strength = side_strength
            else:
                final_signal = trend_sig
                final_strength = trend_strength

        if final_signal and final_strength >= 0.55:
            base = symbol.split('/')[0]
            cg = self.fetch_coinglass_data(base)
            news = self.fetch_cryptopanic_news()

            if not self.get_ai_filter(symbol, df, final_signal, ob, cg, news):
                print(f"[{datetime.now(timezone.utc)}] AI отклонил сигнал {final_signal} для {symbol}")
                return None, None, None

            entry = last['close']
            fee_adj = entry * self.taker_fee
            atr = last['atr']
            if final_signal == 'LONG':
                sl = entry - (self.sl_atr_multiplier * atr) - fee_adj
                tp = entry + (self.tp_atr_multiplier * atr) + fee_adj
            else:
                sl = entry + (self.sl_atr_multiplier * atr) + fee_adj
                tp = entry - (self.tp_atr_multiplier * atr) - fee_adj

            print(f"[{datetime.now(timezone.utc)}] Сигнал {final_signal} (сила {final_strength:.2f}) для {symbol}")
            return final_signal, "Scalp", {'entry': entry, 'stop_loss': sl, 'take_profit': tp}

        print(f"[{datetime.now(timezone.utc)}] Нет сильного сигнала (сила {final_strength:.2f}) для {symbol}")
        return None, None, None

    def log_trade(self, symbol, side, entry, exit_price, size, pnl, pnl_pct, df_last):
        timestamp = datetime.now(timezone.utc).isoformat()
        row = [
            timestamp, symbol, side, entry, exit_price, size, pnl, pnl_pct,
            df_last['rsi'], df_last['adx'], df_last['vwap'], df_last['ema_20'], df_last['ema_50'],
            df_last['atr'], df_last['bb_upper'], df_last['bb_lower'],
            df_last['stoch_k'], df_last['stoch_d'], df_last['macd_hist'], df_last.get('bid_ratio', 50)
        ]
        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[{timestamp}] Сделка записана в {self.trade_log_file}")

    def get_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                equity = float(bal['info']['result']['list'][0]['totalEquity'])
                print(f"[{datetime.now(timezone.utc)}] Баланс: totalEquity = {equity:.2f} USDT")
                return equity
            else:
                print(f"[{datetime.now(timezone.utc)}] USDT не найден в ответе баланса")
                return 0.0
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] BALANCE FETCH FAILED: {str(e)}")
            return 0.0

    # Остальные методы (fetch_coinglass_data, fetch_cryptopanic_news, get_ai_filter,
    # place_order, manage_position, close_position) остаются без изменений из твоего кода

    def run(self):
        while True:
            print(f"[{datetime.now(timezone.utc)}] Starting new cycle")
            self.check_daily_loss_limit()  # Проверка лимита перед циклом
            self.get_balance()  # Мониторинг баланса
            for symbol in self.symbols:
                try:
                    df = self.fetch_ohlcv(symbol)
                    if df is None:
                        print(f"[{datetime.now(timezone.utc)}] Skipping {symbol} - no data")
                        continue
                    df = self.calculate_indicators(df)

                    if self.positions.get(symbol):
                        self.manage_position(symbol, df)
                    else:
                        signal, s_type, params = self.detect_signal(symbol, df)
                        if signal:
                            self.place_order(symbol, signal, params)
                except Exception as e:
                    print(f"[{datetime.now(timezone.utc)}] Error for {symbol}: {e}")
            print(f"[{datetime.now(timezone.utc)}] Cycle finished, sleeping 30s")
            time.sleep(30)


if __name__ == "__main__":
    bot = BybitScalpingBot()
    bot.run()
