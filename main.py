import os, requests, numpy as np, pandas as pd, pandas_ta as ta, telebot, time

# --- [CONFIG] ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")
CP_KEY = os.getenv("CRYPTOPANIC_API_KEY")
bot = telebot.TeleBot(TOKEN)

# --- [БЛОК 1: СБОР ДАННЫХ] ---
class DataCollector:
    def __init__(self, symbol="ETHUSDT"):
        self.symbol = symbol
        self.coin = symbol.replace("USDT", "")

    def get_bybit_market_data(self):
        try:
            url = "https://api.bybit.com/v5/market"
            k_res = requests.get(f"{url}/kline", params={"category": "linear", "symbol": self.symbol, "interval": "15", "limit": 200}, timeout=10).json()
            klines = k_res['result']['list'][::-1]
            t_res = requests.get(f"{url}/tickers", params={"category": "linear", "symbol": self.symbol}, timeout=10).json()
            ticker = t_res['result']['list'][0]
            o_res = requests.get(f"{url}/orderbook", params={"category": "linear", "symbol": self.symbol, "limit": 50}, timeout=10).json()
            return {"klines": klines, "ticker": ticker, "orderbook": o_res['result']}
        except Exception as e:
            print(f"Ошибка Bybit: {e}"); return None

    def get_coinglass_data(self):
        if not CG_KEY: return None
        try:
            headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
            res = requests.get(f"https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol={self.coin}", headers=headers, timeout=10).json()
            return res.get('data', [{}])[0]
        except Exception as e:
            print(f"Ошибка Coinglass: {e}"); return None

    def get_cryptopanic_news(self):
        if not CP_KEY: return []
        try:
            res = requests.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={CP_KEY}&currencies={self.coin}&kind=news&filter=hot", timeout=10).json()
            return res.get('results', [])[:5]
        except Exception as e:
            print(f"Ошибка News: {e}"); return []

    def collect_all(self):
        return {"market": self.get_bybit_market_data(), "blockchain": self.get_coinglass_data(), "news": self.get_cryptopanic_news()}

# --- [БЛОК 2-3: АНАЛИЗАТОР (ИНДИКАТОРЫ + МАТЕМАТИКА)] ---
class TechnicalAnalyzer:
    def __init__(self, raw_bundle):
        self.market = raw_bundle.get('market')
        
    def prepare_df(self):
        if not self.market: return None
        df = pd.DataFrame(self.market['klines'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 't'])
        for col in ['o', 'h', 'l', 'c', 'v']: 
            df[col] = pd.to_numeric(df[col])
        return df

    def calculate(self):
        df = self.prepare_df()
        if df is None or len(df) < 30: return None # Проверка на достаточность данных
        
        res = {'price': df['c'].iloc[-1]}
        
        # Трендовые
        res['ema20'] = ta.ema(df['c'], length=20).iloc[-1]
        res['ema50'] = ta.ema(df['c'], length=50).iloc[-1]
        res['ema200'] = ta.ema(df['c'], length=200).iloc[-1] if len(df) >= 200 else res['ema50']
        res['vwap'] = (df['v'] * (df['h'] + df['l'] + df['c']) / 3).sum() / df['v'].sum()
        
        # Осцилляторы
        res['rsi'] = ta.rsi(df['c'], length=14).iloc[-1]
        macd = ta.macd(df['c'])
        res['macd_h'] = macd.iloc[-1, 1] # Берем вторую колонку (гистограмму) напрямую
        
        # Волатильность (Исправленный блок)
        bb = ta.bbands(df['c'], length=20, std=2)
        # Вместо имен 'BBU_20_2.0' берем по индексу: 0 - нижняя, 1 - средняя, 2 - верхняя
        res['bb_up'] = bb.iloc[-1, 2]
        res['bb_low'] = bb.iloc[-1, 0]
        
        res['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14).iloc[-1]
        res['adx'] = ta.adx(df['h'], df['l'], df['c'], length=14).iloc[-1, 0]
        
        return res

    def analyze_orderbook(self):
        try:
            if not self.market or 'orderbook' not in self.market: return 0.5
            ob = self.market['orderbook']
            bids = sum([float(i[1]) for i in ob['b']])
            asks = sum([float(i[1]) for i in ob['a']])
            return bids / (bids + asks) if (bids + asks) > 0 else 0.5
        except: return 0.5

# --- [БЛОК 4-5: SMART ANALYST & AI] ---
class SmartAnalyst:
    def __init__(self, tech_data, raw_bundle):
        self.tech, self.blockchain, self.news = tech_data, raw_bundle.get('blockchain'), raw_bundle.get('news')

    def analyze_all(self):
        rep = {'ls_ratio': float(self.blockchain.get('v', 1.0)) if self.blockchain else 1.0}
        bull_w = ['buy', 'pump', 'growth', 'surge', 'bullish', 'support']
        score = 0
        titles = ""
        for n in self.news:
            titles += n['title'] + " | "
            if any(w in n['title'].lower() for w in bull_w): score += 1
        rep['sentiment'] = "Positive" if score > 0 else "Neutral/Negative"
        rep['news_summary'] = titles[:200]
        
        prompt = f"ETH:{self.tech['price']}. RSI:{round(self.tech['rsi'],1)}, Sent:{rep['sentiment']}. Pro assessment 15 words."
        try:
            res = requests.post("https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {DS_KEY}"},
                               json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, timeout=10).json()
            rep['ai_verdict'] = res['choices'][0]['message']['content']
        except: rep['ai_verdict'] = "AI Offline."
        return rep

# --- [БЛОК 6: ГРАФИКА (ГЛАЗА БОТА)] ---
class ChartGeometry:
    def __init__(self, raw_bundle):
        m = raw_bundle.get('market', {})
        self.klines = m.get('klines', [])
        if self.klines:
            self.c = np.array([float(x[4]) for x in self.klines])
            self.h = np.array([float(x[2]) for x in self.klines])
            self.l = np.array([float(x[3]) for x in self.klines])

    def detect_structure(self):
        if len(self.c) < 50: return "Unknown"
        h, l = max(self.h[-20:-1]), min(self.l[-20:-1])
        if self.c[-1] > h: return "BOS Bullish"
        if self.c[-1] < l: return "BOS Bearish"
        return "Range"

    def find_patterns(self):
        if len(self.c) < 60: return "Neutral"
        h1, h2 = max(self.h[-40:-20]), max(self.h[-20:])
        if abs(h1 - h2) / h1 < 0.002: return "Double Top"
        l1, l2 = min(self.l[-40:-20]), min(self.l[-20:])
        if abs(l1 - l2) / l1 < 0.002: return "Double Bottom"
        return "Neutral"

    def get_sr_levels(self):
        all_p = np.concatenate([self.h[-100:], self.l[-100:]])
        lvls = [round(p, 2) for p in all_p if np.sum(np.abs(all_p - p) / p < 0.001) > 3]
        return sorted(list(set(lvls)))[-3:]

# --- [БЛОК СТРАТЕГИИ] ---
class StrategyManager:
    def __init__(self, tech, struct, smart):
        self.t, self.s, self.a = tech, struct, smart
        
    def calculate_score(self):
        sc = 0
        if "Bullish" in self.s['structure']: sc += 3
        if "Bearish" in self.s['structure']: sc -= 3
        if self.t['price'] > self.t['ema200']: sc += 1
        if 30 < self.t['rsi'] < 55: sc += 1
        if self.a['ls_ratio'] < 0.95: sc += 2
        if self.a['sentiment'] == "Positive": sc += 1
        return sc

    def generate_setup(self):
        sc = self.calculate_score()
        side = "LONG" if sc >= 5 else "SHORT" if sc <= -5 else None
        if not side: return {"side": None}
        atr = self.t['atr']
        sl = round(self.t['price'] - (atr*2.5) if side=="LONG" else self.t['price'] + (atr*2.5), 2)
        tp = round(self.t['price'] + (atr*5) if side=="LONG" else self.t['price'] - (atr*5), 2)
        return {"side": side, "entry": self.t['price'], "sl": sl, "tp": tp, "score": sc}

# --- [ГЛАВНЫЙ ЦИКЛ] ---
def main_run():
    collector = DataCollector("ETHUSDT")
    bot.send_message(CHAT_ID, "🔮 **TITAN MONOLITH v2.0** АКТИВИРОВАН\nМониторинг ETH (15m/Bybit/Coinglass/News)")
    while True:
        try:
            raw = collector.collect_all()
            if not raw['market']: time.sleep(30); continue
            
            ana = TechnicalAnalyzer(raw)
            tech = ana.calculate()
            imb = ana.analyze_orderbook()
            
            geo = ChartGeometry(raw)
            struct = {'structure': geo.detect_structure(), 'patterns': geo.find_patterns()}
            lvls = geo.get_sr_levels()
            
            smart = SmartAnalyst(tech, raw)
            smart.tech['imb'] = imb
            ai_data = smart.analyze_all()
            
            setup = StrategyManager(tech, struct, ai_data).generate_setup()
            
            if setup['side']:
                msg = (f"🚨 **{setup['side']} SIGNAL**\n━━━━━━━━━━━━\n🎯 Entry: `{setup['entry']}`\n🛡 SL: `{setup['sl']}` | 💰 TP: `{setup['tp']}`\n"
                       f"📊 Score: `{setup['score']}/10` | RSI: `{round(tech['rsi'],1)}` | ADX: `{round(tech['adx'],1)}` \n"
                       f"🧠 **AI:** _{ai_data['ai_verdict']}_")
                bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
                time.sleep(3600)
            
if __name__ == "__main__":
    # Список монет для мониторинга
    SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

    def main_run():
        # Создаем сборщики данных для каждой монеты
        collectors = {sym: DataCollector(sym) for sym in SYMBOLS}
        
        print("--- СИСТЕМА ЗАПУЩЕНА ---")
        bot.send_message(CHAT_ID, f"🚀 **TITAN MULTI-BOT v2.1**\nМониторинг: {', '.join(SYMBOLS)}")
        
        while True:
            for sym in SYMBOLS:
                try:
                    print(f"[{time.strftime('%H:%M:%S')}] Анализ {sym}...")
                    raw = collectors[sym].collect_all()
                    
                    if not raw or not raw.get('market'):
                        continue
                    
                    # Полный цикл анализа
                    ana = TechnicalAnalyzer(raw)
                    tech = ana.calculate()
                    if not tech: continue
                    
                    smart = SmartAnalyst(tech, raw)
                    smart.tech['imb'] = ana.analyze_orderbook()
                    ai_data = smart.analyze_all()
                    
                    geo = ChartGeometry(raw)
                    struct = {'structure': geo.detect_structure(), 'patterns': geo.find_patterns()}
                    
                    setup = StrategyManager(tech, struct, ai_data).generate_setup()
                    
                    if setup.get('side'):
                        msg = (f"🚨 **{setup['side']} SIGNAL: {sym}**\n"
                               f"━━━━━━━━━━━━\n"
                               f"🎯 Entry: `{setup['entry']}`\n"
                               f"🛡 SL: `{setup['sl']}` | 💰 TP: `{setup['tp']}`\n"
                               f"📊 Score: `{setup['score']}/10`\n"
                               f"🧠 **AI:** _{ai_data.get('ai_verdict', 'N/A')}_")
                        bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
                        time.sleep(5)
                
                except Exception as e:
                    print(f"Ошибка в цикле {sym}: {e}")
                    time.sleep(10)
            
            print(f"[{time.strftime('%H:%M:%S')}] Круг завершен. Пауза 5 мин.")
            time.sleep(300)

    # Непосредственный вызов функции запуска
    main_run()
