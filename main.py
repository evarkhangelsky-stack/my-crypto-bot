import os, telebot, requests, time

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_full_analysis():
    try:
        base_url = "https://api.bybit.com/v5/market"
        symbol = "ETHUSDT"
        
        # 1. Получаем свечи (5м) для индикаторов
        k_res = requests.get(f"{base_url}/kline", params={"category":"linear","symbol":symbol,"interval":"5","limit":"100"}).json()
        closes = [float(c[4]) for c in k_res['result']['list'][::-1]]
        highs = [float(c[2]) for c in k_res['result']['list'][::-1]]
        lows = [float(c[3]) for c in k_res['result']['list'][::-1]]

        # --- БЛОК МАТЕМАТИКИ (ИНДИКАТОРЫ) ---
        # RSI
        diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        avg_gain = sum([d for d in diffs[-14:] if d > 0]) / 14
        avg_loss = sum([-d for d in diffs[-14:] if d < 0]) / 14
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss or 0.001))))

        # STOCHASTIC (%K)
        low_14 = min(lows[-14:])
        high_14 = max(highs[-14:])
        stoch_k = ((closes[-1] - low_14) / ((high_14 - low_14) or 0.001)) * 100

        # MACD (Упрощенный: EMA12 - EMA26)
        ema12 = sum(closes[-12:]) / 12
        ema26 = sum(closes[-26:]) / 26
        macd = ema12 - ema26

        # 2. Стакан (Depth)
        ob = requests.get(f"{base_url}/orderbook", params={"category":"linear","symbol":symbol,"limit":"50"}).json()
        bids = sum([float(b[1]) for b in ob['result']['b']])
        asks = sum([float(a[1]) for a in ob['result']['a']])
        imbalance = (bids / (bids + asks)) * 100

        # 3. Рыночный контекст (Tickers)
        t_res = requests.get(f"{base_url}/tickers", params={"category":"linear","symbol":symbol}).json()
        t = t_res['result']['list'][0]

        return {
            "price": t['lastPrice'],
            "rsi": round(rsi, 2),
            "stoch": round(stoch_k, 2),
            "macd": round(macd, 4),
            "imbalance": round(imbalance, 2),
            "oi": t['openInterest'],
            "funding": t['fundingRate'],
            "vol24h": t['volume24h'],
            "high24h": t['highPrice24h'],
            "low24h": t['lowPrice24h']
        }
    except Exception as e:
        print(f"Ошибка сбора: {e}")
        return None

if __name__ == "__main__":
    print(">>> ПОЛНЫЙ МОНИТОРИНГ ЗАПУЩЕН")
    while True:
        data = get_full_analysis()
        if data:
            # Математический фильтр (пропускаем только если есть движуха)
            # Мы можем задать условия здесь, но пока шлем отчет, чтобы видеть данные
            
            prompt = (f"Анализ ETH: Цена {data['price']}, RSI {data['rsi']}, Stoch {data['stoch']}, MACD {data['macd']}, "
                      f"Стакан {data['imbalance']}%, OI {data['oi']}. Дай краткий вердикт трейдеру.")
            
            try:
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()
                advice = ai_res['choices'][0]['message']['content']
            except: advice = "AI думает..."

            msg = (f"📊 **FULL ETH DATA**\n\n"
                   f"💰 Price: `${data['price']}`\n"
                   f"📈 RSI: `{data['rsi']}` | Stoch: `{data['stoch']}` | MACD: `{data['macd']}`\n"
                   f"⚖️ Book Imbalance: `{data['imbalance']}%`\n"
                   f"🎯 OI: `{data['oi']}` | Funding: `{data['funding']}`\n"
                   f"🌊 Vol 24h: `{data['vol24h']}`\n"
                   f"🏔 High/Low: `{data['high24h']}` / `{data['low24h']}`\n\n"
                   f"🧠 **AI VERDICT:**\n{advice}")
            
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
        
        time.sleep(300) # Проверка раз в 5 минут, чтобы не спамить
