import os, telebot, requests, time, numpy as np

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")

bot = telebot.TeleBot(TOKEN)

def calculate_rsi(series, period=14):
    delta = np.diff(series)
    gain = (delta[delta > 0].sum()) / period
    loss = (-delta[delta < 0].sum()) / period
    return 100 - (100 / (1 + (gain / (loss or 0.001))))

def get_pro_analysis():
    try:
        res = requests.get("https://api.bybit.com/v5/market/kline", 
                           params={"category":"linear","symbol":"ETHUSDT","interval":"5","limit":"150"}, timeout=10).json()
        candles = res['result']['list'][::-1]
        closes = np.array([float(c[4]) for c in candles])
        highs = np.array([float(c[2]) for c in candles])
        lows = np.array([float(c[3]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])

        ema50 = np.mean(closes[-50:])
        trend = "BULL" if closes[-1] > ema50 else "BEAR"
        
        tr = np.maximum(highs[-14:] - lows[-14:], np.abs(highs[-14:] - closes[-15:-1]))
        atr = np.mean(tr)
        
        avg_vol = np.mean(volumes[-20:])
        high_vol = volumes[-1] > avg_vol * 1.5

        return {
            "price": closes[-1], "trend": trend, "atr": round(atr, 2), 
            "high_vol": high_vol, "rsi": round(calculate_rsi(closes), 2)
        }
    except Exception as e:
        print(f"Ошибка сбора данных: {e}")
        return None

if __name__ == "__main__":
    print(">>> ЗАПУСК СИСТЕМЫ БЕЗ ГАЛЛЮЦИНАЦИЙ")
    bot.send_message(CHAT_ID, "🚀 Система исправлена. Теперь только актуальные данные!")
    
    while True:
        m = get_pro_analysis() # Вот здесь создается 'm'
        
        if m:
            signal = None
            if m['trend'] == "BULL" and m['rsi'] < 45 and m['high_vol']:
                signal = "LONG (Откат по тренду)"
            elif m['trend'] == "BEAR" and m['rsi'] > 55 and m['high_vol']:
                signal = "SHORT (Импульс вниз)"

            if signal:
                # Математика Python (Всегда точная)
                sl_dist = m['atr'] * 2
                sl = round(m['price'] - sl_dist if "LONG" in signal else m['price'] + sl_dist, 2)
                tp = round(m['price'] + (sl_dist * 3) if "LONG" in signal else m['price'] - (sl_dist * 3), 2)

                # Промпт для ИИ (Только логика, без цифр)
                prompt = (f"Анализ {signal} для ETH. Цена: {m['price']}. Тренд {m['trend']}, RSI {m['rsi']}, Vol Burst: {m['high_vol']}. "
                          "Напиши ПОЧЕМУ мы входим. НЕ ПРИДУМЫВАЙ СВОИ ЦЕНЫ. Пиши только логику за 20 слов.")
                
                try:
                    ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                        headers={"Authorization": f"Bearer {DS_KEY}"},
                        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, timeout=15).json()
                    advice = ai_res['choices'][0]['message']['content']
                except:
                    advice = "Вход подтвержден всплеском объема и направлением тренда."

                # Финальное сообщение: Python вставляет цифры, ИИ дает текст
                text = (f"🚨 **SMART SIGNAL: {signal}**\n\n"
                        f"📥 Вход: `{m['price']}`\n"
                        f"🛡 Stop: `{sl}` | 🎯 TP: `{tp}`\n\n"
                        f"📊 **Metrics:**\n"
                        f"- Trend: `{m['trend']}` | RSI: `{m['rsi']}`\n"
                        f"- ATR: `{m['atr']}` | Vol: `High`\n\n"
                        f"🧠 **AI Анализ:** {advice}")
                
                bot.send_message(CHAT_ID, text, parse_mode="Markdown")
                time.sleep(1800) # Пауза 30 минут

        time.sleep(180)
