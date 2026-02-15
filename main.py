import os, telebot, requests, time
import pandas as pd

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
BY_KEY = os.getenv("BYBIT_API_KEY")
BY_SECRET = os.getenv("BYBIT_API_SECRET")

bot = telebot.TeleBot(TOKEN)

def get_bybit_data():
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": "ETHUSDT", "interval": "5", "limit": "50"}
        # Публичный запрос без подписи для получения графиков (так стабильнее)
        res = requests.get(url, params=params, timeout=10)
        
        if res.status_code != 200:
            print(f"(!) Ошибка сервера Bybit: {res.status_code}")
            return None, None, None

        data = res.json()
        if 'result' not in data or not data['result']['list']:
            print("(!) Bybit прислал пустой список свечей")
            return None, None, None
            
        candles = data['result']['list']
        df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df['c'] = df['c'].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 0.00001))))
        
        # OI
        oi_url = "https://api.bybit.com/v5/market/open-interest"
        oi_res = requests.get(oi_url, params={"category": "linear", "symbol": "ETHUSDT", "intervalTime": "5min"}, timeout=10).json()
        oi = float(oi_res['result']['list'][0]['openInterest'])
        
        return df['c'].iloc[-1], rsi.iloc[-1], oi
    except Exception as e:
        print(f"(!) Критическая ошибка: {e}")
        return None, None, None

def ask_ai(txt):
    try:
        res = requests.post("https://api.deepseek.com/chat/completions", 
            headers={"Authorization": f"Bearer {DS_KEY}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": txt}]}, timeout=20)
        return res.json()['choices'][0]['message']['content']
    except:
        return "AI думает..."

if __name__ == "__main__":
    print(">>> МОНИТОРИНГ BYBIT ЗАПУЩЕН")
    while True:
        price, rsi, oi = get_bybit_data()
        if price:
            report = f"BYBIT ETH: ${price} | RSI: {rsi:.2f} | OI: {oi}"
            advice = ask_ai(f"Дай совет по трейду: {report}")
            bot.send_message(CHAT_ID, f"📊 **ОТЧЕТ (BYBIT)**\n{report}\n\n🧠 **AI:** {advice}")
            print(f">>> Отправлено в TG: {price}")
            time.sleep(300)
        else:
            print("--- Ожидание данных...")
            time.sleep(30)
