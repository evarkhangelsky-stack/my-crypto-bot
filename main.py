import os, telebot, requests, time
import pandas as pd
import pandas_ta as ta

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

# Список монет для мониторинга
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def get_market_data(symbol):
    try:
        base_url = "https://api.bybit.com/v5/market"
        # 1. Свечи 1м
        k_res = requests.get(f"{base_url}/kline", params={"category":"linear","symbol":symbol,"interval":"1","limit":"50"}, timeout=10).json()
        df = pd.DataFrame(k_res['result']['list'], columns=['ts','o','h','l','c','v','tv'])
        df['c'] = df['c'].astype(float)
        df = df.iloc[::-1]
        rsi = ta.rsi(df['c'], length=14).iloc[-1]

        # 2. Стакан
        ob = requests.get(f"{base_url}/orderbook", params={"category":"linear","symbol":symbol,"limit":"25"}, timeout=10).json()
        bids = sum([float(i[1]) for i in ob['result']['b']])
        asks = sum([float(i[1]) for i in ob['result']['a']])
        imbalance = (bids / (bids + asks)) * 100

        # 3. Общие данные
        t_res = requests.get(f"{base_url}/tickers", params={"category":"linear","symbol":symbol}, timeout=10).json()
        ticker = t_res['result']['list'][0]

        return {
            "symbol": symbol,
            "price": df['c'].iloc[-1],
            "rsi": round(rsi, 2),
            "imbalance": round(imbalance, 2),
            "oi": ticker['openInterest']
        }
    except Exception as e:
        print(f"Ошибка сбора {symbol}: {e}")
        return None

if __name__ == "__main__":
    print(f">>> МОНИТОРИНГ {SYMBOLS} ЗАПУЩЕН")
    last_signal_times = {s: 0 for s in SYMBOLS}
    
    while True:
        for symbol in SYMBOLS:
            data = get_market_data(symbol)
            
            if data:
                # Математика сигнала
                is_long = data['rsi'] < 30 and data['imbalance'] > 65
                is_short = data['rsi'] > 70 and data['imbalance'] < 35
                
                current_time = time.time()
                # Сигнал или отчет раз в 30 минут
                if is_long or is_short or (current_time - last_signal_times[symbol] > 1800):
                    
                    status = "🟢 LONG" if is_long else "🔴 SHORT" if is_short else "⚪️ WAIT"
                    
                    prompt = f"Монета: {symbol}. Сигнал: {status}. RSI: {data['rsi']}, Imbalance: {data['imbalance']}%. Дай совет скальперу за 10 слов."
                    
                    try:
                        ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                            headers={"Authorization": f"Bearer {DS_KEY}"},
                            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()
                        advice = ai_res['choices'][0]['message']['content']
                    except:
                        advice = "AI временно недоступен."

                    msg = (f"🚀 **{symbol} {status}**\n\n"
                           f"💰 Цена: `${data['price']}`\n"
                           f"📊 RSI: `{data['rsi']}` | Стакан: `{data['imbalance']}%` 📈\n"
                           f"🎯 OI: `{data['oi']}`\n\n"
                           f"🧠 **AI:** {advice}")
                    
                    bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
                    last_signal_times[symbol] = current_time
            
            time.sleep(5) # Короткая пауза между монетами, чтобы не спамить API

        time.sleep(60) # Проверка всего списка раз в минуту
