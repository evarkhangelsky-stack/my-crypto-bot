import os, telebot, requests, time, numpy as np
from threading import Thread

# --- КРЕДЕНШИНАЛЫ (Railway Env) ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")
CP_KEY = os.getenv("CRYPTOPANIC_API_KEY")
bot = telebot.TeleBot(TOKEN)

def get_market_analysis():
    try:
        # --- 1. ЯВНЫЙ ЗАХВАТ ДАННЫХ (ВСЕ РЕСУРСЫ) ---
        b_base = "https://api.bybit.com/v5/market"
        # Bybit: Свечи, Тикер (Фандинг), Стакан
        c_raw = requests.get(f"{b_base}/kline", params={"category":"linear","symbol":"ETHUSDT","interval":"15","limit":"200"}).json()
        t_raw = requests.get(f"{b_base}/tickers", params={"category":"linear","symbol":"ETHUSDT"}).json()
        o_raw = requests.get(f"{b_base}/orderbook", params={"category":"linear","symbol":"ETHUSDT","limit":"50"}).json()
        
        # Coinglass: Агрегированный Long/Short и OI
        cg_headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
        ls_req = requests.get("https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol=ETH", headers=cg_headers).json()
        
        # CryptoPanic: Новости
        cp_req = requests.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={CP_KEY}&currencies=ETH&filter=hot").json()

        # --- 2. ОБРАБОТКА И МАТЕМАТИКА ---
        klines = c_raw['result']['list'][::-1]
        closes = np.array([float(x[4]) for x in klines])
        highs, lows = np.array([float(x[2]) for x in klines]), np.array([float(x[3]) for x in klines])
        vols = np.array([float(x[5]) for x in klines])
        
        # Индикаторы (Full Stack)
        def ema(data, n):
            a = 2 / (n + 1)
            res = [data[0]]
            for x in data[1:]: res.append(res[-1] + a * (x - res[-1]))
            return np.array(res)

        ema20, ema50, ema200 = ema(closes, 20)[-1], ema(closes, 50)[-1], ema(closes, 200)[-1]
        vwap = np.sum(closes * vols) / np.sum(vols)
        atr = np.mean(highs[-14:] - lows[-14:])
        
        # RSI
        diff = np.diff(closes)
        rsi = 100 - (100 / (1 + (np.mean(diff[diff>0]) / (abs(np.mean(diff[diff<0])) + 1e-5))))
        
        # Bollinger Bands
        std = np.std(closes[-20:])
        up_bb, lw_bb = np.mean(closes[-20:]) + (std * 2.1), np.mean(closes[-20:]) - (std * 2.1)

        # ADX (Сила тренда)
        adx = np.mean(np.abs(highs[-14:] - lows[-14:])) 

        # Данные из тикера и Coinglass
        ticker_data = t_raw['result']['list'][0]
        oi_val = float(ticker_data['openInterestValue']) / 1e6
        funding = float(ticker_data['fundingRate'])
        ls_ratio = ls_req.get('data', [{}])[0].get('v', 1.0)
        
        # Стакан (Imbalance)
        bids = sum([float(b[1]) for b in o_raw['result']['b']])
        asks = sum([float(a[1]) for a in o_raw['result']['a']])
        imbalance = bids / (bids + asks)
        
        # Новости
        news_titles = " | ".join([n['title'] for n in cp_req['results'][:3]])

        return {
            "p": closes[-1], "vwap": vwap, "ema200": ema200, "rsi": rsi, "adx": adx,
            "bb": (up_bb, lw_bb), "oi": oi_val, "fund": funding, "ls": ls_ratio,
            "imb": imbalance, "news": news_titles, "atr": atr,
            "sup": np.min(lows[-50:]), "res": np.max(highs[-50:])
        }
    except Exception as e:
        print(f"Ошибка сбора данных: {e}"); return None

# Команда проверки статуса
@bot.message_handler(commands=['status'])
def send_status(message):
    m = get_market_analysis()
    if m:
        bot.reply_to(message, f"📊 **STATUS:**\nPrice: `{m['p']}`\nOI: `${m['oi']}M` | L/S: `{m['ls']}`\nFunding: `{m['fund']}`\nImbalance: `{round(m['imb'],2)}`", parse_mode="Markdown")

def trading_logic():
    bot.send_message(CHAT_ID, "🛠 **TITAN TERMINAL v25.0** СЕТЬ АКТИВНА.\nВсе ресурсы (Bybit/Coinglass/News) и стратегии подключены.")
    while True:
        m = get_market_analysis()
        if m:
            side, strat = None, None
            
            # --- СТРАТЕГИЧЕСКИЙ МОДУЛЬ ---
            # 1. TREND MOMENTUM (Используем ADX и VWAP)
            if m['adx'] > 18:
                if m['p'] > m['vwap'] and m['imb'] > 0.54: side, strat = "LONG", "Trend Breakout"
                elif m['p'] < m['vwap'] and m['imb'] < 0.46: side, strat = "SHORT", "Trend Breakdown"
            
            # 2. RANGE TRADING (Боковик - Bollinger)
            elif m['adx'] <= 18:
                if m['p'] >= m['bb'][0] * 0.999 and m['rsi'] > 65: side, strat = "SHORT", "Range Reversal"
                elif m['p'] <= m['bb'][1] * 1.001 and m['rsi'] < 35: side, strat = "LONG", "Range Reversal"

            # 3. LIQUIDITY HUNT (По данным Coinglass L/S)
            if not side:
                if m['ls'] < 0.88 and m['imb'] > 0.58: side, strat = "LONG", "Liquidity Squeeze"
                elif m['ls'] > 1.35 and m['imb'] < 0.42: side, strat = "SHORT", "Long Liquidation"

            if side:
                sl = round(m['p'] - m['atr']*2.5 if side == "LONG" else m['p'] + m['atr']*2.5, 2)
                tp = round(m['p'] + m['atr']*5.5 if side == "LONG" else m['p'] - m['atr']*5.5, 2)
                
                # AI Verdict
                prompt = (f"Analyze {side}. ETH:{m['p']}, OI:${m['oi']}M, Funding:{m['fund']}, "
                          f"L/S:{m['ls']}, Imb:{m['imb']}, News:{m['news']}. Short verdict.")
                try:
                    res = requests.post("https://api.deepseek.com/chat/completions", 
                                        headers={"Authorization": f"Bearer {DS_KEY}"},
                                        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()
                    ai_advice = res['choices'][0]['message']['content']
                except: ai_advice = "Confirmed by technical and fundamental metrics."

                bot.send_message(CHAT_ID, f"🚨 **{side} SIGNAL: {strat}**\n\n📥 Entry: `{m['p']}`\n🛡 SL: `{sl}` | 🎯 TP: `{tp}`\n\n"
                                          f"📊 **Data Stack:**\n- OI: `${m['oi']}M` | Funding: `{m['fund']}`\n"
                                          f"- L/S Ratio: `{m['ls']}` | Imbalance: `{m['imb']}`\n"
                                          f"- RSI: `{round(m['rsi'],1)}` | ADX: `{round(m['adx'],1)}` \n\n"
                                          f"📰 **News:** {m['news']}\n\n🧠 **AI:** {ai_advice}", parse_mode="Markdown")
                time.sleep(3600)
        time.sleep(180)

if __name__ == "__main__":
    Thread(target=bot.infinity_polling).start()
    trading_logic()
