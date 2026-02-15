import os, telebot, requests, time, hmac, hashlib
import pandas as pd

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
BY_KEY = os.getenv("BYBIT_API_KEY")
BY_SECRET = os.getenv("BYBIT_API_SECRET")

bot = telebot.TeleBot(TOKEN)

def get_signature(params):
    """Генерация подписи для приватных запросов Bybit (баланс)"""
    timestamp = str(int(time.time() * 1000))
    param_str = timestamp + BY_KEY + "5000" + params
    hash = hmac.new(bytes(BY_SECRET, "utf-8"), param_str.encode("utf-8"), hashlib.sha256)
    return hash.hexdigest(), timestamp

def get_bybit_market(symbol="ETHUSDT", interval="5"):
    try:
        url = "https://api-eu.bybit.com/v5/market/kline"
        res = requests.get(url, params={"category": "linear", "symbol": symbol, "interval": interval, "limit": "50"}).json()
        df = pd.DataFrame(res['result']['list'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df['c'] = df['c'].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Считаем RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 0.00001))))
        return df['c'].iloc[-1], rsi.iloc[-1]
    except:
        return None, None

def get_balance():
    try:
        timestamp = str(int(time.time() * 1000))
        params = "accountType=UNIFIED&coin=USDT"
        recv_window = "5000"
        raw_sig = timestamp + BY_KEY + recv_window + params
        signature = hmac.new(bytes(BY_SECRET, "utf-8"), raw_sig.encode("utf-8"), hashlib.sha256).hexdigest()
        
        headers = {
            "X-BAPI-API-KEY": BY_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window
        }
        res = requests.get("https://api-eu.bybit.com/v5/account/wallet-balance", headers=headers, params={"accountType":"UNIFIED", "coin":"USDT"}).json()
        return res['result']['list'][0]['coin'][0]['walletBalance']
    except:
        return "0.0"

def get_extra_data():
    # Данные за 24 часа и Open Interest
    url = "https://api-eu.bybit.com/v5/market/tickers"
    res = requests.get(url, params={"category": "linear", "symbol": "ETHUSDT"}).json()
    ticker = res['result']['list'][0]
    return ticker['price24hPcnt'], ticker['openInterest']

if __name__ == "__main__":
    print(">>> ЗАПУЩЕН ПОЛНЫЙ МОНИТОРИНГ BYBIT")
    while True:
        p5, r5 = get_bybit_market(interval="5")   # 5 минут
        p60, r60 = get_bybit_market(interval="60") # 1 час
        change24, oi = get_extra_data()
        balance = get_balance()

        if p5:
            msg = (f"💰 Баланс: {balance} USDT\n"
                   f"📊 ETH: ${p5} ({float(change24)*100:.2f}% за 24ч)\n"
                   f" indicador RSI 5m: {r5:.2f}\n"
                   f" indicador RSI 1h: {r60:.2f}\n"
                   f"🔥 Open Interest: {oi}")
            
            # Отправляем в AI для глубокого анализа
            ai_advice = requests.post("https://api.deepseek.com/chat/completions", 
                headers={"Authorization": f"Bearer {DS_KEY}"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": f"Сделай краткий тех. анализ: {msg}"}]}).json()['choices'][0]['message']['content']

            bot.send_message(CHAT_ID, f"🚀 **BYBIT FULL REPORT**\n\n{msg}\n\n🧠 **AI Анализ:**\n{ai_advice}")
            print(">>> Полный отчет отправлен")
        
        time.sleep(900) # Раз в 15 минут, чтобы не спамить
