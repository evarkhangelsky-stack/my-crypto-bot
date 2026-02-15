import os, telebot, requests, time
import pandas as pd
import pandas_ta as ta
import numpy as np

# Инициализация
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

def get_data():
    symbol = "ETHUSDT"
    base = "https://fapi.binance.com"
    
    # 1. Тянем свечи (для RSI и EMA)
    klines = requests.get(f"{base}/fapi/v1/klines?symbol={symbol}&interval=5m&limit=50").json()
    df = pd.DataFrame(klines, columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
    df[['c', 'v', 'qav']] = df[['c', 'v', 'qav']].astype(float)
    
    # 2. Тянем Open Interest
    oi_data = requests.get(f"{base}/fapi/v1/openInterest?symbol={symbol}").json()
    oi = float(oi_data['openInterest'])
    
    # 3. Расчет CVD (упрощенно через Taker Buy Volume)
    # CVD = Сумма (Taker Buy Quote Volume - (Total Quote Volume - Taker Buy Quote Volume))
    buy_vol = df['tq'].astype(float)
    sell_vol = df['qav'] - buy_vol
    df['delta'] = buy_vol - sell_vol
    df['cvd'] = df['delta'].cumsum()
    
    # 4. Технические индикаторы
    df['rsi'] = ta.rsi(df['c'], length=14)
    
    return df, oi

def ask_deepseek(context):
    url = "https://api.deepseek.com/v1/chat/completions" # Проверь актуальный эндпоинт в ЛК
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Ты проф трейдер. Анализируй данные ETH и дай краткий прогноз: Long, Short или Wait."},
            {"role": "user", "content": context}
        ]
    }
    try:
        r = requests.post(url, json=payload, headers=headers)
        return r.json()['choices'][0]['message']['content']
    except:
        return "Ошибка связи с DeepSeek"

# Основной цикл
last_oi = 0
while True:
    try:
        df, current_oi = get_data()
        current_price = df['c'].iloc[-1]
        cvd_change = df['delta'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        if last_oi != 0:
            oi_diff = ((current_oi - last_oi) / last_oi) * 100
            
            # ТРИГГЕР: Аномалия (OI скакнул > 1% или RSI в зонах 30/70)
            if abs(oi_diff) > 1.0 or rsi > 70 or rsi < 30:
                context = (f"ETH Price: {current_price}, OI Change: {oi_diff:.2f}%, "
                           f"Last Delta: {cvd_change:.2f}, RSI: {rsi:.2f}")
                
                ai_opinion = ask_deepseek(context)
                
                msg = (f"🔍 **АНОМАЛИЯ ETH**\n"
                       f"Цена: ${current_price}\n"
                       f"OI: {oi_diff:+.2f}%\n"
                       f"CVD Delta: {cvd_change:.2f}\n"
                       f"RSI: {rsi:.2f}\n\n"
                       f"🧠 **DeepSeek:**\n{ai_opinion}")
                
                bot.send_message(CHAT_ID, msg)
        
        last_oi = current_oi
        time.sleep(300) # Проверка каждые 5 минут
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
