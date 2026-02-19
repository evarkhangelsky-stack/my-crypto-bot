import os, time, ccxt, requests, pandas as pd, numpy as np
from datetime import datetime, timezone

# ==========================================
# 1. КЛЮЧИ ДОЛЖНЫ БЫТЬ В RAILWAY VARIABLES
# ==========================================
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
DEEPSEEK_KEY = os.getenv('DEEPSEEK_API_KEY')
CP_KEY = os.getenv('CRYPTOPANIC_API_KEY')
CG_KEY = os.getenv('COINGLASS_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TG_CHAT = os.getenv('TELEGRAM_CHAT_ID')

# Параметры риск-менеджмента
RISK_PER_TRADE = 0.01          # 1% от баланса на сделку
DAILY_LOSS_LIMIT = 0.05        # Стоп на день если -5%
MAX_DAILY_LOSSES = 4           # Макс кол-во стопов в сутки
PARTIAL_FIX_PROGRESS = 0.25    # 25% пути до Тейка -> Закрыть 50% и БЕЗУБЫТОК
ADX_MAX_FILTER = 45            # Фильтр сильного тренда (против ножей)

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ (МАТЕМАТИКА)
# ==========================================
class TechnicalIndicators:
    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / loss)))

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
        minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), 0)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        return (100 * abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        mid = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        return mid + (std_dev * std), mid, mid - (std_dev * std)

# ==========================================
# 3. ОСНОВНОЙ БОТ (БИЗНЕС-ЛОГИКА)
# ==========================================
class BybitProfessionalBot:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.active_positions = {}
        self.daily_losses = 0
        self.last_reset_day = datetime.now(timezone.utc).day

    def send_tg(self, msg):
        try: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
                           data={'chat_id': TG_CHAT, 'text': msg, 'parse_mode': 'Markdown'})
        except: print("TG Error")

    def get_news(self):
        if not CP_KEY: return "No News Key"
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CP_KEY}&kind=news&filter=hot"
            res = requests.get(url).json()
            return " | ".join([p['title'] for p in res['results'][:3]])
        except: return "News unavailable"

    def get_ai_decision(self, symbol, signal, df, news):
        if not DEEPSEEK_KEY: return True
        try:
            last = df.iloc[-1]
            prompt = f"Trade: {symbol} {signal}. RSI: {last['rsi']:.1f}, ADX: {last['adx']:.1f}. News: {news}. Enter? YES/NO"
            headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
            res = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload).json()
            return "YES" in res['choices'][0]['message']['content'].upper()
        except: return True

    def calculate_qty(self, symbol, entry, sl):
        try:
            balance = float(self.exchange.fetch_balance()['total']['USDT'])
            risk_usd = balance * RISK_PER_TRADE
            qty = risk_usd / abs(entry - sl)
            return float(self.exchange.amount_to_precision(symbol, qty))
        except: return 0

    def manage_positions(self, symbol, df):
        pos = self.active_positions[symbol]
        price = df.iloc[-1]['close']
        
        # Логика БЕЗУБЫТКА
        target_dist = abs(pos['tp'] - pos['entry'])
        move = (price - pos['entry']) if pos['side'] == 'buy' else (pos['entry'] - price)
        progress = move / target_dist if target_dist > 0 else 0

        if progress >= PARTIAL_FIX_PROGRESS and not pos['half_closed']:
            try:
                side_close = 'sell' if pos['side'] == 'buy' else 'buy'
                self.exchange.create_order(symbol, 'market', side_close, pos['qty']/2)
                pos['sl'] = pos['entry'] # Перенос в БУ
                pos['half_closed'] = True
                self.send_tg(f"💰 {symbol}: 50% прибыли зафиксировано. Стоп в БЕЗУБЫТКЕ.")
            except Exception as e: print(f"Manage Error: {e}")

        # Проверка закрытия
        is_tp = (price >= pos['tp']) if pos['side'] == 'buy' else (price <= pos['tp'])
        is_sl = (price <= pos['sl']) if pos['side'] == 'buy' else (price >= pos['sl'])

        if is_tp or is_sl:
            res = "🍀 PROFIT" if is_tp else "🧨 LOSS"
            if not is_tp: self.daily_losses += 1
            del self.active_positions[symbol]
            self.send_tg(f"🏁 {symbol} закрыт: {res}")

    def run(self):
        self.send_tg("🤖 Бот запущен на Bybit. Режим реальной торговли.")
        ti = TechnicalIndicators()
        
        while True:
            # Сброс по UTC 00:00
            now = datetime.now(timezone.utc)
            if now.day != self.last_reset_day:
                self.daily_losses = 0
                self.last_reset_day = now.day

            if self.daily_losses >= MAX_DAILY_LOSSES:
                time.sleep(3600); continue

            for symbol in self.symbols:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'})
                    
                    df['rsi'] = ti.rsi(df['close'])
                    df['adx'] = ti.adx(df['high'], df['low'], df['close'])
                    df['bb_u'], df['bb_m'], df['bb_l'] = ti.bollinger_bands(df['close'])
                    
                    if symbol in self.active_positions:
                        self.manage_positions(symbol, df)
                    else:
                        last = df.iloc[-1]
                        signal = None
                        if last['rsi'] < 30 and last['close'] < last['bb_l'] and last['adx'] < ADX_MAX_FILTER:
                            signal, params = 'LONG', {'entry': last['close'], 'sl': last['close']*0.993, 'tp': last['close']*1.015}
                        elif last['rsi'] > 70 and last['close'] > last['bb_u'] and last['adx'] < ADX_MAX_FILTER:
                            signal, params = 'SHORT', {'entry': last['close'], 'sl': last['close']*1.007, 'tp': last['close']*0.985}
                        
                        if signal:
                            news = self.get_news()
                            if self.get_ai_decision(symbol, signal, df, news):
                                qty = self.calculate_qty(symbol, params['entry'], params['sl'])
                                if qty > 0:
                                    side = 'buy' if signal == 'LONG' else 'sell'
                                    self.exchange.create_order(symbol, 'market', side, qty)
                                    self.active_positions[symbol] = {
                                        'side': side, 'entry': params['entry'], 
                                        'sl': params['sl'], 'tp': params['tp'], 
                                        'qty': qty, 'half_closed': False
                                    }
                                    self.send_tg(f"🚀 Вход {symbol} {signal}. Риск 1%.")
                except Exception as e: print(f"Error {symbol}: {e}")
                time.sleep(5)
            time.sleep(20)

if __name__ == "__main__":
    BybitProfessionalBot().run()
