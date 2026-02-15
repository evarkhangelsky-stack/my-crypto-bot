import os, telebot, requests, time
import pandas as pd

# Загрузка ключей
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_market_data():
    base = "https://fapi.binance.com"
    # Берем 100 свечей для стабильного расчета RSI
    url = f"{base}/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=100"
    r = requests.get(url).json()
    
    df = pd.DataFrame(r, columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
    df['c'] = df['c'].astype(float)
    
    # Защита: если данных вдруг меньше нужного для RSI
    if len(df) < 30:
        return None, None, None

    # Считаем RSI вручную (без лишних библиотек)
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Получаем Open Interest
    oi_r = requests.get(f"{base}/fapi/v1/openInterest?symbol=ETHUSDT").json()
    oi = float(oi_r['openInterest'])
    
    return df['c'].iloc[-1], rsi.iloc[-1], oi

def ask_deepseek(text):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Ты эксперт-трейдер. Проанализируй данные ETH и дай очень краткий совет."},
            {"role": "user", "content": text}
        ]
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Ошибка ИИ: {str(e)}"

if __name__ == "__main__":
    print("Бот запущен и мониторит ETH...")
    while True:
        try:
            price, rsi_val, oi_val = get_market_data()
            
            if price is not None:
                # Сейчас шлем всегда, чтобы убедиться в работе. 
                # Потом поставим фильтр на аномалии.
                report = f"ETH: ${price}, RSI: {rsi_val:.2f}, OI: {oi_val}"
                ai_advice = ask_deepseek(report)
                
                bot.send_message(CHAT_ID, f"📊 **ОТЧЕТ ETH**\n{report}\n\n🧠 **DeepSeek:**\n{ai_advice}")
                print("Сигнал отправлен!")
            
            time.sleep(300) # Проверка раз в 5 минут
        except Exception as e:
            print(f"Ошибка в цикле: {e}")
            time.sleep(60)
