import os, telebot, requests, time
import pandas as pd

# Ключи из переменных Railway
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_market_data():
    try:
        base = "https://fapi.binance.com"
        # Просим 100 свечей
        res = requests.get(f"{base}/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=100", timeout=10).json()
        df = pd.DataFrame(res, columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
        df['c'] = df['c'].astype(float)
        
        # Проверка: достаточно ли данных для RSI (нужно минимум 15)
        if len(df) < 20:
            print("(!) Мало данных от Binance, ждем...")
            return None, None, None

        # Считаем RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Защита от деления на ноль
        rs = gain / loss.replace(0, 0.00001)
        rsi_series = 100 - (100 / (1 + rs))
        
        # Берем последнее значение, если оно не пустое
        current_price = df['c'].iloc[-1]
        current_rsi = rsi_series.iloc[-1]
        
        # Если RSI еще не посчитался (NaN), возвращаем None
        if pd.isna(current_rsi):
            return None, None, None

        # Open Interest
        oi_data = requests.get(f"{base}/fapi/v1/openInterest?symbol=ETHUSDT", timeout=10).json()
        return current_price, current_rsi, float(oi_data['openInterest'])
        
    except Exception as e:
        print(f"(!) Ошибка API: {e}")
        return None, None, None

def ask_deepseek(report):
    try:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Ты крипто-эксперт. Дай очень краткий совет."},
                {"role": "user", "content": report}
            ]
        }
        r = requests.post(url, json=data, headers=headers, timeout=15)
        return r.json()['choices'][0]['message']['content']
    except:
        return "Нейросеть думает..."

if __name__ == "__main__":
    print(">>> МОНИТОРИНГ ЗАПУЩЕН")
    while True:
        try:
            price, rsi, oi = get_market_data()
            
            if price:
                msg = f"ETH: ${price} | RSI: {rsi:.2f} | OI: {oi}"
                print(f"--- Данные в норме: {price}")
                
                ai_advice = ask_deepseek(msg)
                bot.send_message(CHAT_ID, f"📊 **ОТЧЕТ**\n{msg}\n\n🧠 **AI:** {ai_advice}")
                print(">>> Сообщение отправлено в Telegram")
            else:
                print("--- Данные пока не готовы, повтор через 30 сек...")
                time.sleep(30)
                continue

            time.sleep(300) # 5 минут пауза
            
        except Exception as e:
            print(f"(!) Ошибка цикла: {e}")
            time.sleep(60)
