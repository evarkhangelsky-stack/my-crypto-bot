import os, telebot, requests, time
import pandas as pd

# 1. Загрузка настроек
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_market_data():
    """Получает цену, RSI и OI с биржи"""
    try:
        base = "https://fapi.binance.com"
        # Берем 100 свечей для расчета RSI
        r = requests.get(f"{base}/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=100", timeout=10).json()
        df = pd.DataFrame(r, columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
        df['c'] = df['c'].astype(float)
        
        # Расчет RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        
        # Open Interest
        oi_r = requests.get(f"{base}/fapi/v1/openInterest?symbol=ETHUSDT", timeout=10).json()
        
        return df['c'].iloc[-1], rsi.iloc[-1], float(oi_r['openInterest'])
    except Exception as e:
        print(f"Ошибка при сборе данных: {e}")
        return None, None, None

def ask_deepseek(report_text):
    """Запрос к нейросети"""
    try:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Ты крипто-аналитик. Дай краткий вывод по данным ETH."},
                {"role": "user", "content": report_text}
            ]
        }
        res = requests.post(url, json=payload, headers=headers, timeout=20)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"ИИ недоступен (ошибка: {e})"

if __name__ == "__main__":
    print(">>> МОНИТОРИНГ ЗАПУЩЕН")
    while True:
        try:
            price, rsi, oi = get_market_data()
            
            if price is not None:
                status_msg = f"ETH: ${price} | RSI: {rsi:.2f} | OI: {oi}"
                print(f"Данные получены: {status_msg}")
                
                # Запрос к ИИ
                ai_comment = ask_deepseek(status_msg)
                
                # Отправка в телеграм
                final_text = f"📊 **ОТЧЕТ ETH**\n{status_msg}\n\n🧠 **DeepSeek:**\n{ai_comment}"
                bot.send_message(CHAT_ID, final_text)
                print(">>> Отчет успешно отправлен в Telegram")
            
            # Ждем 5 минут до следующей проверки
            time.sleep(300)
            
        except Exception as e:
            print(f"Критическая ошибка цикла: {e}")
            time.sleep(60) # При ошибке ждем минуту и пробуем снова
