import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import telebot

# ==========================================
# НАСТРОЙКИ РИСК-МЕНЕДЖМЕНТА
# ==========================================
RISK_PER_TRADE = 0.01          # Риск 1% от баланса
DAILY_LOSS_LIMIT_PCT = 0.05    # Остановка если минус 5% за день
MAX_DAILY_LOSSES = 4           # Остановка если 4 стопа подряд
PARTIAL_TP_PCT = 0.25          # Закрыть 50% позиции на 1/4 пути к Тейку
ADX_MAX_FILTER = 45            # Не входить в контртренд если ADX > 45

# ==========================================
# УЧЕТ СТАТИСТИКИ (UTC)
# ==========================================
class TradingStats:
    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_losses_count = 0
        self.last_reset_day = datetime.now(timezone.utc).day
        self.trading_halted = False

    def check_reset(self):
        now_utc = datetime.now(timezone.utc)
        if now_utc.day != self.last_reset_day:
            print(f"🚀 {now_utc.strftime('%Y-%m-%d')} - Новый торговый день по UTC! Лимиты сброшены.")
            self.daily_pnl = 0.0
            self.daily_losses_count = 0
            self.last_reset_day = now_utc.day
            self.trading_halted = False

stats = TradingStats()

# ==========================================
# БЛОК ИНДИКАТОРОВ
# ==========================================
class TechnicalIndicators:
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
    def adx(high, low, close, period=14):
        plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
        minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), 0)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        middle = close.rolling(window=period).mean()
        upper = middle + (close.rolling(window=period).std() * std)
        lower = middle - (close.rolling(window=period).std() * std)
        return upper, middle, lower

# ==========================================
# ОСНОВНОЙ КЛАСС БОТА
# ==========================================
class BybitBot:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.active_positions = {}
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

    def send_telegram(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            requests.post(url, data={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'Markdown'})
        except Exception as e:
            print(f"TG Error: {e}")

    def fetch_ohlcv(self, symbol):
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            print(f"Fetch Error: {e}")
            return None

    def calculate_indicators(self, df):
        ti = TechnicalIndicators()
        df['rsi'] = ti.rsi(df['close'])
        df['ema_200'] = ti.ema(df['close'], 200)
        df['vwap'] = ti.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['adx'] = ti.adx(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = ti.bollinger_bands(df['close'])
        return df

    def calculate_qty(self, symbol, entry, sl):
        try:
            balance = float(self.exchange.fetch_balance()['total']['USDT'])
            risk_usd = balance * RISK_PER_TRADE
            stop_dist = abs(entry - sl)
            if stop_dist <= 0: return 0
            
            qty = risk_usd / stop_dist
            market = self.exchange.market(symbol)
            return float(self.exchange.amount_to_precision(symbol, qty))
        except Exception as e:
            print(f"Qty Error: {e}")
            return 0

    def detect_signal(self, df, symbol):
        last = df.iloc[-1]
        
        # Фильтр ADX - не лезем в сильный тренд
        if last['adx'] > ADX_MAX_FILTER:
            return None, None

        # Логика входа
        if last['rsi'] < 30 and last['close'] < last['bb_lower']:
            sl = last['close'] * 0.993
            tp = last['close'] * 1.015
            return 'LONG', {'entry': last['close'], 'sl': sl, 'tp': tp}

        if last['rsi'] > 70 and last['close'] > last['bb_upper']:
            sl = last['close'] * 1.007
            tp = last['close'] * 0.985
            return 'SHORT', {'entry': last['close'], 'sl': sl, 'tp': tp}

        return None, None

    def place_order(self, symbol, signal, params):
        if stats.trading_halted:
            return

        try:
            side = 'buy' if signal == 'LONG' else 'sell'
            qty = self.calculate_qty(symbol, params['entry'], params['sl'])
            
            if qty <= 0: return

            order = self.exchange.create_order(symbol, 'market', side, qty)
            
            self.active_positions[symbol] = {
                'side': side,
                'entry': params['entry'],
                'sl': params['sl'],
                'tp': params['tp'],
                'qty': qty,
                'half_closed': False
            }
            
            self.send_telegram(f"🚀 *{signal}* на {symbol}\nОбъем: {qty}\nРиск: {RISK_PER_TRADE*100}%")
        except Exception as e:
            print(f"Order Error: {e}")

    def manage_position(self, symbol, df):
        pos = self.active_positions[symbol]
        last_price = df.iloc[-1]['close']
        
        # Расчет профита в зависимости от стороны
        if pos['side'] == 'buy':
            current_profit_pct = (last_price - pos['entry']) / pos['entry']
            total_target_pct = (pos['tp'] - pos['entry']) / pos['entry']
            is_sl = last_price <= pos['sl']
            is_tp = last_price >= pos['tp']
        else:
            current_profit_pct = (pos['entry'] - last_price) / pos['entry']
            total_target_pct = (pos['entry'] - pos['tp']) / pos['entry']
            is_sl = last_price >= pos['sl']
            is_tp = last_price <= pos['tp']

        # 1. ЧАСТИЧНЫЙ ФИКС И БЕЗУБЫТОК
        progress = current_profit_pct / total_target_pct if total_target_pct > 0 else 0
        
        if progress >= PARTIAL_TP_PCT and not pos['half_closed']:
            try:
                side_close = 'sell' if pos['side'] == 'buy' else 'buy'
                half_qty = pos['qty'] / 2
                self.exchange.create_order(symbol, 'market', side_close, half_qty)
                
                pos['sl'] = pos['entry'] # Тянем стоп в БУ
                pos['half_closed'] = True
                self.send_telegram(f"✅ {symbol}: 50% закрыто, стоп в БЕЗУБЫТКЕ")
            except Exception as e:
                print(f"Half-close Error: {e}")

        # 2. ЗАКРЫТИЕ ПО ТЕЙКУ ИЛИ СТОПУ
        if is_tp or is_sl:
            res = "PROFIT" if is_tp else "LOSS"
            if res == "LOSS":
                stats.daily_losses_count += 1
                if stats.daily_losses_count >= MAX_DAILY_LOSSES:
                    stats.trading_halted = True
                    self.send_telegram("⚠️ Достигнут лимит стопов на сегодня. Торговля приостановлена.")
            
            del self.active_positions[symbol]
            self.send_telegram(f"🏁 Сделка закрыта: {res} на {symbol}")

    def run(self):
        print(f"🚀 Бот запущен. Риск на сделку: {RISK_PER_TRADE*100}%")
        while True:
            stats.check_reset()
            for symbol in self.symbols:
                try:
                    df = self.fetch_ohlcv(symbol)
                    if df is None: continue
                    df = self.calculate_indicators(df)
                    
                    if symbol in self.active_positions:
                        self.manage_position(symbol, df)
                    else:
                        signal, params = self.detect_signal(df, symbol)
                        if signal:
                            self.place_order(symbol, signal, params)
                    
                    last = df.iloc[-1]
                    print(f"[{symbol}] Price: {last['close']:.2f} | RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}")
                    
                except Exception as e:
                    print(f"Error in {symbol}: {e}")
                time.sleep(2)
            time.sleep(30) # Проверка каждые 30 секунд

if __name__ == "__main__":
    bot = BybitBot()
    bot.run()
