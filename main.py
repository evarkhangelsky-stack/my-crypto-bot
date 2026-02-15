import os, telebot, requests, time
import pandas as pd

# 1. Забираем ключи из Railway
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_data():
    try:
        # Используем публичное евро-зеркало (не требует подписи и не виснет)
        url = "https://api-eu.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": "ETHUSDT", "interval": "15", "limit": "50"}
        res = requests.get(url, params=params, timeout=10).json()
        
        # Берем последнюю цену и считаем RSI
        candles = res['result']['list']
        df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df['c'] = df['c'].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Стандартный расчет RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 0.00001))))
        
        return df['c'].iloc[-1], round(rsi.iloc[-1], 2)
    except Exception as e:
        print(f"Ошибка получения данных: {e}")
        return None, None

def ask_ai(price, rsi):
    try:
        prompt = f"Цена ETH: ${price}, RSI: {rsi}. Дай совет трейдеру одним коротким предложением."
        res = requests.post("https://api.deepseek.com/chat/completions", 
            headers={"Authorization": f"Bearer {DS_KEY}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, timeout=15).json()
        return res['choices'][0]['message']['content']
    except:
        return "DeepSeek пока занят анализом..."

if __name__ == "__main__":
    print(">>> БОТ ЗАПУЩЕН И ЖДЕТ 1-й ОТЧЕТ")
    while True:
        price, rsi = get_data()
        if price:
            advice = ask_ai(price, rsi)
            text = f"📊 **ОТЧЕТ BYBIT**\n\n💰 ETH: `${price}`\n📈 RSI: `{rsi}`\n\n🧠 **AI:** {advice}"
            bot.send_message(CHAT_ID, text, parse_mode="Markdown")
            print(f">>> Отправлено в TG: {price}")
        
        time.sleep(600) # Проверка раз в 10 минут
