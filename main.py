import os, telebot, requests, time
import pandas as pd
import pandas_ta as ta
import numpy as np

# Берем переменные из Railway
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_data():
    base = "https://fapi.binance.com"
    klines = requests.get(f"{base}/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=50").json()
    df = pd.DataFrame(klines, columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
    df[['c', 'v', 'qav', 'tq']] = df[['c', 'v', 'qav', 'tq']].astype(float)
    
    oi_data = requests.get(f"{base}/fapi/v1/openInterest?symbol=ETHUSDT").json()
    oi = float(oi_data['openInterest'])
    
    # Расчет CVD
    df['delta'] = df['tq'] - (df['qav'] - df['tq'])
    df['rsi'] = ta.rsi(df['c'], length=14)
    return df, oi

def ask_deepseek(context):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Ты крипто-аналитик. Дай краткий совет по ETH на основе данных."},
            {"role": "user", "content": context}
        ]
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка ИИ: {str(e)}"

# Главный цикл
if __name__ == "__main__":
    print("Бот запущен...")
    last_oi = 0
    while True:
        try:
            df, current_oi = get_data()
            price = df['c'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Для теста: срабатывает всегда. Потом заменим на фильтр.
            if True: 
                context = f"ETH: ${price}, RSI: {rsi:.2f}, OI: {current_oi}"
                ai_msg = ask_deepseek(context)
                bot.send_message(CHAT_ID, f"📊 **Отчет ETH**\nЦена: ${price}\nRSI: {rsi:.2f}\n\n🧠 **DeepSeek:**\n{ai_msg}")
            
            time.sleep(300) # Раз в 5 минут
        except Exception as e:
            print(f"Ошибка цикла: {e}")
            time.sleep(60)
