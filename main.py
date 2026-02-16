import os, telebot, requests, time, numpy as np

# --- CONFIG ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")
CP_KEY = os.getenv("CRYPTOPANIC_API_KEY")
bot = telebot.TeleBot(TOKEN)

def get_comprehensive_data():
    try:
        # 1. СБОР ДАННЫХ
        b_url = "https://api.bybit.com/v5/market"
        c = requests.get(f"{b_url}/kline", params={"category":"linear","symbol":"ETHUSDT","interval":"5","limit":"200"}).json()['result']['list'][::-1]
        t = requests.get(f"{b_url}/tickers", params={"category":"linear","symbol":"ETHUSDT"}).json()['result']['list'][0]
        o = requests.get(f"{b_url}/orderbook", params={"category":"linear","symbol":"ETHUSDT","limit":"50"}).json()['result']
        
        cg_h = {"accept": "application/json", "CG-API-KEY": CG_KEY}
        ls_res = requests.get("https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol=ETH", headers=cg_h, timeout=10).json()
        
        cp_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CP_KEY}&currencies=ETH&kind=news&filter=hot"
        news_res = requests.get(cp_url, timeout=10).json()
        news_feed = " | ".join([f"{p['title']} (👍{p['votes']['positive']})" for p in news_res['results'][:3]])

        # 2. ТЯЖЕЛАЯ МАТЕМАТИКА
        closes = np.array([float(x[4]) for x in c])
        highs, lows, vols = np.array([float(x[2]) for x in c]), np.array([float(x[3]) for x in c]), np.array([float(x[5]) for x in c])

        def ema(d, n):
            a = 2/(n+1)
            e = [d[0]]
            for x in d[1:]: e.append(e[-1] + a*(x-e[-1]))
            return np.array(e)

        # Технические индикаторы
        ema20, ema50, ema200 = ema(closes, 20), ema(closes, 50), ema(closes, 200)
        vwap = np.sum(closes * vols) / np.sum(vols)
        std = np.std(closes[-20:])
        upper_bb, lower_bb = np.mean(closes[-20:]) + (std*2), np.mean(closes[-20:]) - (std*2)
        
        # MACD & RSI
        m_l = ema(closes, 12) - ema(closes, 26)
        s_l = ema(m_l, 9)
        macd_h = m_l[-1] - s_l[-1]
        
        diff = np.diff(closes)
        rsi = 100 - (100 / (1 + (np.mean(diff[diff>0]) / (abs(np.mean(diff[diff<0])) + 1e-5))))
        
        # Дивергенции (на отрезке 15 свечей)
        bull_div = (closes[-1] < closes[-15]) and (rsi > 100 - (100 / (1 + (np.mean(np.diff(closes[-15:])[np.diff(closes[-15:])>0]) / (abs(np.mean(np.diff(closes[-15:])[np.diff(closes[-15:])<0])) + 1e-5)))))
        
        # Стакан, OI и Уровни
        imb = sum([float(b[1]) for b in o['b']]) / (sum([float(b[1]) for b in o['b']]) + sum([float(a[1]) for a in o['a']]))
        oi_m = float(t['openInterestValue']) / 1e6
        sup, res = np.min(lows[-60:]), np.max(highs[-60:])
        atr = np.mean(highs[-14:] - lows[-14:])

        return {
            "p": closes[-1], "vwap": vwap, "macd": macd_h, "rsi": rsi, "ema200": ema200[-1],
            "bb": (upper_bb, lower_bb), "imb": imb, "oi": oi_m, "ls": ls_res.get('data', [{}])[0].get('v', 1.0),
            "sup": sup, "res": res, "news": news_feed, "atr": atr, "bull_div": bull_div, "fund": float(t['fundingRate'])
        }
    except Exception as e:
        print(f"Engine Error: {e}"); return None

if __name__ == "__main__":
    bot.send_message(CHAT_ID, "🚀 **GOD MODE ACTIVATED**\nВсе стратегии и API подключены. Охота началась.")
    
    while True:
        m = get_comprehensive_data()
        if m:
            strategy = None
            # 1. Trend: Цена выше всех EMA и VWAP
            if m['p'] > m['vwap'] and m['p'] > m['ema200'] and m['macd'] > 0: strategy = "STRAT: Trend Following"
            # 2. Reversal: Цена у нижней полосы BB + Перепроданность RSI
            elif m['p'] <= m['bb'][1] and m['rsi'] < 30: strategy = "STRAT: Mean Reversion (Long)"
            # 3. Divergence: Бычья дивергенция
            elif m['bull_div'] and m['imb'] > 0.55: strategy = "STRAT: Bullish Divergence"
            # 4. Support: Отскок от годового минимума
            elif (m['p'] - m['sup'])/m['sup'] < 0.001 and m['imb'] > 0.6: strategy = "STRAT: Support Bounce"
            # 5. Liquidity: Низкий L/S + Рост OI (Крупный игрок заходит)
            elif m['ls'] < 0.9 and m['imb'] > 0.6: strategy = "STRAT: Institutional Buy"

            if strategy:
                sl, tp = round(m['p'] - m['atr']*2.5, 2), round(m['p'] + m['atr']*6, 2)
                prompt = (f"Signal: {strategy}. ETH: {m['p']}, OI: ${m['oi']}M, News: {m['news']}. "
                          f"Verificators: RSI {m['rsi']}, MACD {m['macd']}, Imbalance {m['imb']}. Professional audit in 30 words.")
                
                try:
                    ai_advice = requests.post("https://api.deepseek.com/chat/completions", 
                        headers={"Authorization": f"Bearer {DS_KEY}"},
                        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()['choices'][0]['message']['content']
                except: ai_advice = "Multi-strategy confirmation achieved."

                msg = (f"🔥 **GOD MODE SIGNAL: {strategy}**\n\n"
                       f"📥 Entry: `{m['p']}`\n"
                       f"🛡 SL: `{sl}` | 🎯 TP: `{tp}`\n\n"
                       f"📊 **Full Market Data:**\n"
                       f"- MACD: `{round(m['macd'],3)}` | RSI: `{round(m['rsi'],1)}`\n"
                       f"- VWAP: `{round(m['vwap'],2)}` | EMA200: `{round(m['ema200'],2)}`\n"
                       f"- Open Interest: `${m['oi']}M` | L/S: `{m['ls']}`\n"
                       f"- Imbalance: `{m['imb']}` (Orderbook)\n"
                       f"- S/R: `{m['sup']}` / `{m['res']}`\n\n"
                       f"📰 **Live Feed:** {m['news']}\n\n"
                       f"🧠 **AI Audit:** {ai_advice}")
                
                bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
                time.sleep(3600)
        time.sleep(180)
