import os, telebot, requests, time, numpy as np

# Настройки
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_pro_analysis():
    try:
        # 1. Получаем расширенные данные свечей (150 шт для EMA 100)
        res = requests.get("https://api.bybit.com/v5/market/kline", 
                           params={"category":"linear","symbol":"ETHUSDT","interval":"5","limit":"150"}).json()
        candles = res['result']['list'][::-1]
        closes = np.array([float(c[4]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])

        # --- ТЕХНИЧЕСКИЙ АРСЕНАЛ ---
        # 1. EMA 50/100 (Тренд)
        ema50 = sum(closes[-50:]) / 50
        trend = "BULL" if closes[-1] > ema50 else "BEAR"

        # 2. ATR (Волатильность для Стоп-Лосса)
        tr = np.maximum(highs[-14:] - lows[-14:], np.abs(highs[-14:] - closes[-15:-1]))
        atr = np.mean(tr)

        # 3. MACD
        ema12 = np.mean(closes[-12:])
        ema26 = np.mean(closes[-26:])
        macd = ema12 - ema26

        # 4. Объемный анализ (VSA)
        avg_vol = np.mean(volumes[-20:])
        high_vol = volumes[-1] > avg_vol * 1.5 # Всплеск объема

        # 5. Ликвидации и OI (CoinGlass)
        headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
        cg_data = requests.get("https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol=ETH", headers=headers).json()
        ls_ratio = cg_data['data'][0]['v'] if cg_data.get('data') else 1.0

        return {
            "price": closes[-1], "trend": trend, "macd": macd,
            "atr": round(atr, 2), "high_vol": high_vol,
            "ls_ratio": ls_ratio, "rsi": round(calculate_rsi(closes), 2)
        }
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = np.diff(series)
    gain = (delta[delta > 0].sum()) / period
    loss = (-delta[delta < 0].sum()) / period
    return 100 - (100 / (1 + (gain / (loss or 0.001))))

if __name__ == "__main__":
    bot.send_message(CHAT_ID, "🎖 Система 'Gemini Core v7.0' активна. Профессиональный мониторинг запущен.")
    
    while True:
        m = get_pro_analysis()
        if m:
            # СЛОЖНАЯ ЛОГИКА ВХОДА
            signal = None
            # Покупаем если: Тренд Бычий + RSI вышел из перепроданности + Объемы растут
            if m['trend'] == "BULL" and m['rsi'] < 45 and m['macd'] > 0 and m['high_vol']:
                signal = "LONG (По тренду на откате)"
            # Продаем если: Тренд Медвежий + RSI перекуплен + Всплеск объема
            elif m['trend'] == "BEAR" and m['rsi'] > 55 and m['macd'] < 0:
                signal = "SHORT (По тренду)"

            if signal:
                # Стоп ставим по ATR (умный стоп)
                sl_dist = m['atr'] * 2
                sl = m['price'] - sl_dist if "LONG" in signal else m['price'] + sl_dist
                tp = m['price'] + (sl_dist * 2.5) if "LONG" in signal else m['price'] - (sl_dist * 2.5)

                prompt = (f"Анализ {signal} для ETH. Тренд {m['trend']}, RSI {m['rsi']}, MACD {m['macd']}, "
                          f"ATR {m['atr']}, High Volume: {m['high_vol']}, LS Ratio {m['ls_ratio']}. "
                          f"Оцени вероятность успеха и дай четкий план.")
                
                try:
                    ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                        headers={"Authorization": f"Bearer {DS_KEY}"},
                        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()
                    advice = ai_res['choices'][0]['message']['content']
                except: advice = "Вход по стратегии Trend-Following."

                bot.send_message(CHAT_ID, f"🚨 **SMART SIGNAL: {signal}**\n\n"
                                          f"📥 Вход: `{m['price']}`\n"
                                          f"🛡 Stop (ATR): `{round(sl, 2)}` | 🎯 TP: `{round(tp, 2)}`\n\n"
                                          f"📊 **Data Stack:**\n"
                                          f"- Trend: `{m['trend']}` | RSI: `{m['rsi']}`\n"
                                          f"- Vol Burst: `{'YES' if m['high_vol'] else 'NO'}`\n"
                                          f"- ATR: `{m['atr']}`\n\n"
                                          f"🧠 **AI:** {advice}", parse_mode="Markdown")
                time.sleep(3600) # После профи-сигнала ждем 1 час

        time.sleep(180)
