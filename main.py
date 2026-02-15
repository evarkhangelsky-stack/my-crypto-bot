import os, telebot, requests, time
import pandas as pd
import pandas_ta as ta

# Переменные
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

def get_market_data():
    base = "https://fapi.binance.com"
    # Свечи
    r = requests.get(f"{base}/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=50")
    df = pd.DataFrame(r.json(), columns=['ts','o','h','l','c','v','cts','qav','nt','tb','tq','i'])
    df[['c', 'v', 'qav', 'tq']] = df[['c', 'v', 'qav', 'tq']].astype(float)
    # OI
    oi_r = requests.get(f"{base}/fapi/v1/openInterest?symbol=ETHUSDT")
    oi = float(oi_r.json()['openInterest'])
    # RSI и Простая Дельта
    df['rsi'] = ta.rsi(df['c'], length=14)
    delta = df['tq'].iloc[-1] - (df['qav'].iloc[-1] - df['tq'].iloc[-1])
    return df['c'].iloc[-1], df['rsi'].iloc[-1], oi, delta

def ask_deepseek(txt):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": f"Кратко проанализируй ETH: {txt}"}]
    }
    try:
        res = requests.post(url, json=data, headers=headers, timeout=10)
        return res.json()['choices'][0]['message']['content']
    except:
        return "DeepSeek временно недоступен"

if __name__ == "__main__":
    print("Бот стартовал")
    while True:
        try:
            p, r, o, d = get_market_data()
            # Условие True, чтобы точно пришло сообщение для теста
            if True:
                report = f"Цена: {p}, RSI: {r:.2f}, OI: {o}, Delta: {d:.2f}"
                ai_verdict = ask_deepseek(report)
                bot.send_message(CHAT_ID, f"📊 **ETH LIVE**\n{report}\n\n🧠 **DeepSeek:**\n{ai_verdict}")
            time.sleep(300)
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(60)
