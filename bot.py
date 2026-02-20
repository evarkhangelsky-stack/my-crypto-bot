import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
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
        # API keys from environment
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

        # Initialize Bybit (linear futures)
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        # Настройка margin mode и leverage
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        for symbol in self.symbols:
            try:
                self.exchange.set_margin_mode('cross', symbol)
                self.exchange.set_leverage(5, symbol)
                print(f"[{datetime.now(timezone.utc)}] Leverage 5x and cross margin for {symbol}")
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
        self.taker_fee = 0.0006

        # Daily loss limit
        self.daily_loss_limit_pct = -4.2
        self.last_day = None
        self.day_start_equity = None
        self.trading_paused_until = None

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
            print(f"[{datetime.now(timezone.utc)}] OHLCV fetched successfully for {symbol}, rows: {len(df)}")
            return df
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_orderbook_data(self, symbol):
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=50)
            total_bids = sum(bid[1] for bid in orderbook['bids'])
            total_asks = sum(ask[1] for ask in orderbook['asks'])
            total = total_bids + total_asks
            bid_ratio = (total_bids / total) * 100 if total > 0 else 50
            print(f"[{datetime.now(timezone.utc)}] Orderbook fetched for {symbol}, bid_ratio: {bid_ratio:.2f}%")
            return {'bid_ratio': bid_ratio, 'total_volume': total}
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Error fetching orderbook for {symbol}: {e}")
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
                print(f"[{datetime.now(timezone.utc)}] CryptoPanic HTTP error: {res.status_code}")
                return []
            data = res.json()
            return data.get('results', [])[:5]
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] CryptoPanic error: {e}")
            return []

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
        print(f"[{datetime.now(timezone.utc)}] Indicators calculated")
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
            print(f"[{datetime.now(timezone.utc)}] AI filter: {answer}")
            return "YES" in answer
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] AI error: {e}")
            return True

    def check_daily_loss_limit(self):
        now = datetime.now(timezone.utc)
        current_day = now.date()

        if self.last_day != current_day:
            try:
                bal = self.exchange.fetch_balance()
                if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                    equity = float(bal['info']['result']['list'][0]['totalEquity'])
                else:
                    equity = float(bal['USDT']['total']) if 'USDT' in bal and 'total' in bal['USDT'] else 100.0
                
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
            if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                current_equity = float(bal['info']['result']['list'][0]['totalEquity'])
            else:
                current_equity = float(bal['USDT']['total']) if 'USDT' in bal and 'total' in bal['USDT'] else 100.0
                
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
        bb_lower = last['bb_lower']
        bb_upper = last['bb_upper']
        bid_ratio = ob['bid_ratio']

        if price <= bb_lower and rsi < 35 and bid_ratio > 55:
            strength = 0.9 if rsi < 30 and bid_ratio > 65 else 0.6
            return 'LONG', strength
        if price >= bb_upper and rsi > 65 and bid_ratio < 45:
            strength = 0.9 if rsi > 70 and bid_ratio < 35 else 0.6
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
        price = last['close']
        rsi = last['rsi']
        adx = last['adx']
        vwap = last['vwap']
        atr = last['atr']
        bb_upper = last['bb_upper']
        bb_lower = last['bb_lower']
        ema_20 = last['ema_20']
        ema_50 = last['ema_50']

        print(f"[{datetime.now(timezone.utc)}] {symbol} values: Price={price:.2f}, RSI={rsi:.2f}, ADX={adx:.2f}, VWAP={vwap:.2f}, EMA20={ema_20:.2f}, EMA50={ema_50:.2f}, ATR={atr:.2f}, BB Upper={bb_upper:.2f}, BB Lower={bb_lower:.2f}")

        if pd.isna([price, rsi, adx, vwap, atr]).any():
            print(f"[{datetime.now(timezone.utc)}] NaN in indicators for {symbol} — no signal")
            return None, None, None

        ob = self.fetch_orderbook_data(symbol)
        bid_ratio = ob['bid_ratio']

        # Используем новые стратегии с силой сигнала
        side_sig, side_strength = self.sideways_strategy(df, ob)
        trend_sig, trend_strength = self.trend_strategy(df, ob)

        final_signal = None
        final_strength = 0

        if adx < 25:
            # приоритет боковику
            if side_sig:
                final_signal = side_sig
                final_strength = side_strength
            elif trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength * 0.6
        elif adx > 30:
            # приоритет тренду
            if trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength
            elif side_sig:
                final_signal = side_sig
                final_strength = side_strength * 0.6
        else:
            # зона 25–30 — берём самый сильный
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

            entry = price
            fee_adj = price * self.taker_fee
            if final_signal == 'LONG':
                sl = entry - (self.sl_atr_multiplier * atr) - fee_adj
                tp = entry + (self.tp_atr_multiplier * atr) + fee_adj
            else:
                sl = entry + (self.sl_atr_multiplier * atr) + fee_adj
                tp = entry - (self.tp_atr_multiplier * atr) - fee_adj

            print(f"[{datetime.now(timezone.utc)}] СИГНАЛ! {final_signal} (сила {final_strength:.2f}) для {symbol}")
            return final_signal, "Scalp", {'entry': entry, 'stop_loss': sl, 'take_profit': tp}

        print(f"[{datetime.now(timezone.utc)}] Нет сильного сигнала (сила {final_strength:.2f}) для {symbol}")
        return None, None, None

    def get_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                equity = float(bal['info']['result']['list'][0]['totalEquity'])
                print(f"[{datetime.now(timezone.utc)}] Total equity: {equity:.2f} USDT")
                return equity
            elif 'USDT' in bal and 'free' in bal['USDT']:
                usdt_free = float(bal['USDT']['free'])
                print(f"[{datetime.now(timezone.utc)}] USDT free balance: {usdt_free}")
                return usdt_free
            else:
                print(f"[{datetime.now(timezone.utc)}] USDT не найден в ответе баланса")
                return 100.0
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] BALANCE FETCH FAILED: {str(e)}")
            return 100.0

    def place_order(self, symbol, signal, params):
        try:
            balance = self.get_balance()
            if balance <= 0:
                print(f"[{datetime.now(timezone.utc)}] Нулевой баланс, пропускаем ордер")
                return
                
            risk = balance * 0.01  # 1% risk per trade
            size = risk / abs(params['entry'] - params['stop_loss'])
            
            # Округляем до допустимого размера для Bybit
            if symbol.startswith('BTC'):
                size = round(size, 3)  # BTC допускает 0.001 точность
            else:
                size = round(size, 2)  # ETH допускает 0.01 точность

            if size <= 0:
                print(f"[{datetime.now(timezone.utc)}] Размер позиции слишком мал: {size}")
                return

            msg = (
                f"📉 *Сигнал: {symbol}*\n"
                f"{signal} ({params['entry']:.2f})\n"
                f"SL: {params['stop_loss']:.2f}\n"
                f"TP: {params['take_profit']:.2f}\n"
                f"Размер: {size}"
            )
            self.send_telegram(msg)

            # Реальные ордера
            if signal == 'LONG':
                order = self.exchange.create_market_buy_order(symbol, size)
            else:
                order = self.exchange.create_market_sell_order(symbol, size)

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
            print(f"[{datetime.now(timezone.utc)}] Order placed: {signal} {size} for {symbol}")
            self.send_telegram(f"✅ Ордер исполнен: {signal} {size} {symbol} по {actual_entry:.2f}")

        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Order error for {symbol}: {e}")
            self.send_telegram(f"❌ Ошибка ордера {symbol}: {str(e)[:100]}")

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
            pos['stop_loss'] = entry
            pos['trailing_stop_activated'] = True
            self.send_telegram(f'🔒 Trailing: {symbol} to Breakeven')

        print(f"[{datetime.now(timezone.utc)}] Position checked for {symbol}, PNL %: {pnl_pct:.2f}")

    def close_position(self, symbol, price, reason):
        pos = self.positions.get(symbol)
        if not pos:
            return

        if pos['side'] == 'LONG':
            pnl = (price - pos['entry']) * pos['size']
        else:
            pnl = (pos['entry'] - price) * pos['size']

        # Реальное закрытие позиции
        try:
            if pos['side'] == 'LONG':
                self.exchange.create_market_sell_order(symbol, pos['size'])
            else:
                self.exchange.create_market_buy_order(symbol, pos['size'])
            
            msg = (
                f"🔴 *Закрыта {symbol}*\n"
                f"Причина: {reason}\n"
                f"P&L: ${pnl:.2f}"
            )
            self.send_telegram(msg)
            print(f"[{datetime.now(timezone.utc)}] Position closed for {symbol}: {reason}, P&L: ${pnl:.2f}")
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Close order error for {symbol}: {e}")
            self.send_telegram(f"❌ Ошибка закрытия {symbol}: {str(e)[:100]}")

        self.positions[symbol] = None

    def run(self):
        while True:
            print(f"[{datetime.now(timezone.utc)}] Starting new cycle")
            self.get_balance()
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
