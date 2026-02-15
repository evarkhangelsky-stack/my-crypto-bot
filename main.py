import os, telebot, requests, time
import pandas as pd

# Ключи из переменных Railway
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
BY_KEY = os.getenv("BYBIT_API_KEY")
BY_SECRET = os.getenv("BYBIT_API_SECRET")

bot = telebot.TeleBot(TOKEN)

def get_bybit_data():
    try:
        # Получаем свечи ETHUSDT (Bybit v5 API)
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": "ETHUSDT", "interval": "5", "limit": "100"}
        res = requests.get(url, params=params, timeout=10).json()
        
        data = res['result']['list']
        df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df['c'] = df['c'].astype(float)
        # Переворачиваем, так как Bybit отдает от новых к старым
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Расчет RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 0.00001))))
        
        # Получаем Open Interest
        oi_url = "https://api.bybit.com/v5/market/open-interest"
        oi_res = requests.get(oi_url, params={"category": "linear", "symbol": "ETHUSDT", "intervalTime": "5min"}, timeout=10).json()
        oi = float(oi_res['result']['list'][0]['openInterest'])
        
        return df['c'].iloc[-1], rsi.iloc[-1], oi
    except Exception as e:
        print(f"(!) Ошибка Bybit: {e}")
        return None, None, None

def ask_deepseek(report):
    try:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Ты эксперт по крипте. Дай очень короткий совет (1-2 предложения)."},
                {"role": "user", "content": report}
            ]
        }
        r = requests.post(url, json=data, headers=headers, timeout=20)
        return r.json()['choices'][0]['message']['content']
    except:
        return "Нейросеть занята анализом..."

if __name__ == "__main__":
    print(">>> МОНИТОРИНГ BYBIT ЗАПУЩЕН")
    while True:
        try:
            price, rsi, oi = get_bybit_data()
            
            if price:
                msg = f"ETH: ${price} | RSI: {rsi:.2f} | OI: {oi}"
                print(f"--- Данные получены: {price}")
                
                ai_advice = ask_deepseek(msg)
                bot.send_message(CHAT_ID, f"📊 **ОТЧЕТ (BYBIT)**\n{msg}\n\n🧠 **AI:** {ai_advice}")
                print(">>> Сообщение отправлено в Telegram")
            else:
                print("--- Ждем данные от биржи...")

            time.sleep(300) # Проверка каждые 5 минут
            
        except Exception as e:
            print(f"(!) Ошибка в цикле: {e}")
            time.sleep(60)
