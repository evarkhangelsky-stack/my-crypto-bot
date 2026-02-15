import os, telebot, requests, time
import pandas as pd

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_data():
    try:
        # Используем ГЛАВНЫЙ адрес Bybit (без -eu)
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": "ETHUSDT", "interval": "15", "limit": "50"}
        res = requests.get(url, params=params, timeout=15).json()
        
        if 'result' not in res:
            print(f"(!) Биржа ответила странно: {res}")
            return None, None

        candles = res['result']['list']
        df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df['c'] = df['c'].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 0.00001))))
        
        return df['c'].iloc[-1], round(rsi.iloc[-1], 2)
    except Exception as e:
        print(f"(!) Ошибка сети: {e}")
        return None, None

if __name__ == "__main__":
    print(">>> ПРОВЕРКА СВЯЗИ ЗАПУЩЕНА")
    # Сразу при запуске пытаемся отправить сообщение
    price, rsi = get_data()
    if price:
        text = f"✅ Связь с Bybit есть!\n💰 ETH: `${price}`\n📈 RSI: `{rsi}`"
        bot.send_message(CHAT_ID, text, parse_mode="Markdown")
        print(">>> ПЕРВОЕ СООБЩЕНИЕ ОТПРАВЛЕНО")
    else:
        print("(!) Не удалось получить данные при старте")

    while True:
        time.sleep(600) # Ждем 10 минут перед следующим разом
        price, rsi = get_data()
        if price:
            bot.send_message(CHAT_ID, f"📊 Обновление: ETH ${price}, RSI {rsi}")
