import os, time, ccxt, requests, pandas as pd, numpy as np
from datetime import datetime, timezone

# ==========================================
# 1. КОНФИГУРАЦИЯ (Railway Variables)
# ==========================================
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
DEEPSEEK_KEY = os.getenv('DEEPSEEK_API_KEY')
CG_KEY = os.getenv('COINGLASS_API_KEY')
CP_KEY = os.getenv('CRYPTOPANIC_API_KEY')
TG_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TG_CHAT = os.getenv('TELEGRAM_CHAT_ID')

# Параметры торговли
RISK_PER_TRADE = 0.01
DAILY_LOSS_LIMIT = 0.05
MAX_DAILY_LOSSES = 4
LEVERAGE = 5
FEE_RATE = 0.0006  # 0.06% Taker fee Bybit
PARTIAL_FIX_PROGRESS = 0.25 # 25% пути до ТП = фикс 50%

# ==========================================
# 2. ИНДИКАТОРЫ (ПОЛНОСТЬЮ ИЗ PDF/REPLIT)
# ==========================================
class TechnicalIndicators:
    @staticmethod
    def vwap(df):
        tp = (df['h'] + df['l'] + df['c']) / 3
        return (tp * df['v']).cumsum() / df['v'].cumsum()

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / loss)))

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        mid = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        return mid + (std_dev * std), mid, mid - (std_dev * std)

# ==========================================
# 3. СБОР ВНЕШНИХ ДАННЫХ (COINGLASS / NEWS)
# ==========================================
class Analytics:
    @staticmethod
    def get_coinglass_data(symbol):
        if not CG_KEY: return {"ratio": 50, "liqs": 0}
        try:
            coin = symbol.split('/')[0]
            headers = {"coinglassToken": CG_KEY}
            # Long/Short Ratio
            ls_res = requests.get(f"https://open-api-v4.coinglass.com/api/futures/long-short-ratio?symbol={coin}&time_type=h1", headers=headers, timeout=5).json()
            # Ликвидации за 1 час
            liq_res = requests.get(f"https://open-api-v4.coinglass.com/api/futures/liquidation/map?symbol={coin}&range=24h", headers=headers, timeout=5).json()
            
            ratio = float(ls_res['data'][0]['longRatio']) if ls_res.get('data') else 50
            liqs = len(liq_res.get('data', [])) # Считаем количество крупных кластеров ликвидаций
            return {"ratio": ratio, "liqs": liqs}
        except: return {"ratio": 50, "liqs": 0}

    @staticmethod
    def get_news():
        if not CP_KEY: return "No news"
        try:
            res = requests.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={CP_KEY}&kind=news").json()
            return " | ".join([p['title'] for p in res['results'][:3]])
        except: return "News Error"

# ==========================================
# 4. МОЗГ И ИСПОЛНЕНИЕ
# ==========================================
class ProBot:
    def __init__(self):
        self.exchange = ccxt.bybit({'apiKey': API_KEY, 'secret': API_SECRET, 'options': {'defaultType': 'future'}})
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.active_positions = {}
        self.daily_losses = 0
        self.last_reset = datetime.now(timezone.utc).day

    def send_tg(self, msg):
        try: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={'chat_id': TG_CHAT, 'text': msg, 'parse_mode': 'Markdown'})
        except: pass

    def get_ai_decision(self, symbol, signal, df, analytics, news):
        if not DEEPSEEK_KEY: return True
        try:
            last = df.iloc[-1]
            prompt = (f"Market Data for {symbol} ({signal}):\n"
                      f"- RSI: {last['rsi']:.1f}, ADX: {last['adx']:.1f}\n"
                      f"- L/S Ratio: {analytics['ratio']}%\n"
                      f"- Liquidation Clusters: {analytics['liqs']}\n"
                      f"- News: {news}\n"
                      "Trade? (YES/NO)")
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

    def manage_position(self, symbol, df):
        pos = self.active_positions[symbol]
        price = df.iloc[-1]['c']
        
        goal = abs(pos['tp'] - pos['entry'])
        move = (price - pos['entry']) if pos['side'] == 'buy' else (pos['entry'] - price)
        progress = move / goal if goal > 0 else 0

        # Фикс 50% и БУ
        if progress >= PARTIAL_FIX_PROGRESS and not pos['half_closed']:
            try:
                side_close = 'sell' if pos['side'] == 'buy' else 'buy'
                self.exchange.create_order(symbol, 'market', side_close, pos['qty']/2)
                # Ставим стоп в цену входа + комиссия
                offset = pos['entry'] * FEE_RATE * 2
                pos['sl'] = pos['entry'] + offset if pos['side'] == 'buy' else pos['entry'] - offset
                pos['half_closed'] = True
                self.send_tg(f"💰 {symbol}: 50% профита взято. Остальное в безубытке.")
            except Exception as e: print(f"Manage Error: {e}")

        # Закрытие по цели
        is_tp = (price >= pos['tp']) if pos['side'] == 'buy' else (price <= pos['tp'])
        is_sl = (price <= pos['sl']) if pos['side'] == 'buy' else (price >= pos['sl'])

        if is_tp or is_sl:
            res = "✅ PROFIT" if is_tp else "❌ LOSS"
            if not is_tp: self.daily_losses += 1
            del self.active_positions[symbol]
            self.send_tg(f"🏁 {symbol} закрыт: {res}. Цена: {price}")

    def run(self):
        self.send_tg("🚀 **Bybit Master Bot Active.**\nАналитика Coinglass + Ликвидации + AI.")
        while True:
            if datetime.now(timezone.utc).day != self.last_reset:
                self.daily_losses = 0
                self.last_reset = datetime.now(timezone.utc).day

            if self.daily_losses >= MAX_DAILY_LOSSES:
                time.sleep(3600); continue

            for symbol in self.symbols:
                try:
                    bars = self.exchange.fetch_ohlcv(symbol, '5m', limit=200) # Глубокая история
                    df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
                    df['rsi'] = TechnicalIndicators.rsi(df['c'])
                    df['adx'] = TechnicalIndicators.adx(df['h'], df['l'], df['c'])
                    df['bb_u'], _, df['bb_l'] = TechnicalIndicators.bollinger_bands(df['c'])
                    
                    if symbol in self.active_positions:
                        self.manage_position(symbol, df)
                    else:
                        last = df.iloc[-1]
                        signal = None
                        # Уровни с учетом комиссии
                        if last['rsi'] < 30 and last['c'] < last['bb_l'] and last['adx'] < 45:
                            signal, params = 'LONG', {'entry': last['c'], 'sl': last['c']*0.994, 'tp': last['c']*1.018}
                        elif last['rsi'] > 70 and last['c'] > last['bb_u'] and last['adx'] < 45:
                            signal, params = 'SHORT', {'entry': last['c'], 'sl': last['c']*1.006, 'tp': last['c']*0.982}

                        if signal:
                            analytics = Analytics.get_coinglass_data(symbol)
                            news = Analytics.get_news()
                            # Совокупный фильтр
                            ls_ok = (signal == 'LONG' and analytics['ratio'] < 60) or (signal == 'SHORT' and analytics['ratio'] > 40)
                            
                            if ls_ok and self.get_ai_decision(symbol, signal, df, analytics, news):
                                qty = self.calculate_qty(symbol, params['entry'], params['sl'])
                                if qty > 0:
                                    self.exchange.create_order(symbol, 'market', 'buy' if signal=='LONG' else 'sell', qty)
                                    self.active_positions[symbol] = {
                                        'side': 'buy' if signal=='LONG' else 'sell', 'entry': last['c'],
                                        'qty': qty, 'tp': params['tp'], 'sl': params['sl'], 'half_closed': False
                                    }
                                    self.send_tg(f"🔥 ВХОД {symbol} {signal}.\nL/S: {analytics['ratio']}%\nAI: OK")
                except Exception as e: print(f"Error {symbol}: {e}")
                time.sleep(2)
            time.sleep(20) # 20 секунд 20 секунд между проверками

if __name__ == "__main__":
    ProBot().run()
