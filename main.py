import os, telebot, requests, time
import pandas as pd

bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

def get_market_data():
    base = "https://fapi.binance.com"
    # Запрашиваем 100 свечей, чтобы RSI точно рассчитался
    r = requests.get(f"{base}/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=100").json()
    df = pd.DataFrame(r, columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
    df['c'] = df['c'].astype(float)
    
    if len(df) < 20:
        return None, None, None

    # RSI вручную
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # OI
    oi_r = requests.get(f"{base}/fapi/v1/openInterest?symbol=ETHUSDT").json()
    
    return df['c'].iloc[-1], rsi.iloc[-1], float(oi_r['openInterest'])

def ask_deepseek(txt):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": f"Кратко проанализируй ETH: {txt}"}]
    }
    try:
        res = requests.post(url, json=data, headers=headers, timeout=15)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"DeepSeek недоступен: {str(e)}"

if __name__ == "__main__":
    print("Мониторинг запущен...")
    while True:
        try:
            price, rsi, oi = get_market_data()
            
            if price is not None:
                report = f"Цена: {price}, RSI: {rsi:.2f}, OI: {oi}"
                ai_verdict = ask_deepseek(report)
                bot.send_message(CHAT_ID, f"📊 **ETH REPORT**\n{report}\n\n🧠 **AI:** {ai_verdict}")
                print("Отчет отправлен в Телеграм")
            
            time.sleep(300) # Проверка каждые 5 минут
        except Exception as e:
            print(f"Ошибка цикла: {e}")
            time.sleep(60)
            
