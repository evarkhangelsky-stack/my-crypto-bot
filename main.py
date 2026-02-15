import os, telebot, requests, time, hmac, hashlib
import pandas as pd

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
BY_KEY = os.getenv("BYBIT_API_KEY")
BY_SECRET = os.getenv("BYBIT_API_SECRET")

bot = telebot.TeleBot(TOKEN)

def get_balance():
    """Безопасное получение баланса"""
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        # Строгий порядок для Bybit v5
        param_str = timestamp + BY_KEY + recv_window + "accountType=UNIFIED&coin=USDT"
        signature = hmac.new(bytes(BY_SECRET, "utf-8"), param_str.encode("utf-8"), hashlib.sha256).hexdigest()
        
        headers = {
            "X-BAPI-API-KEY": BY_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window
        }
        res = requests.get("https://api-eu.bybit.com/v5/account/wallet-balance", 
                           headers=headers, params={"accountType":"UNIFIED", "coin":"USDT"}, timeout=10).json()
        return res['result']['list'][0]['coin'][0]['walletBalance']
    except:
        return "Не удалось получить (проверь ключи)"

def get_market_full(interval="5"):
    """Получение графиков и RSI"""
    try:
        url = "https://api-eu.bybit.com/v5/market/kline"
        res = requests.get(url, params={"category": "linear", "symbol": "ETHUSDT", "interval": interval, "limit": "50"}, timeout=10).json()
        df = pd.DataFrame(res['result']['list'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df['c'] = df['c'].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss.replace(0, 0.00001))))
        return df['c'].iloc[-1], rsi.iloc[-1]
    except:
        return None, None

def get_ticker():
    """24h изменение и Open Interest"""
    try:
        res = requests.get("https://api-eu.bybit.com/v5/market/tickers", params={"category": "linear", "symbol": "ETHUSDT"}, timeout=10).json()
        t = res['result']['list'][0]
        return t['price24hPcnt'], t['openInterest']
    except:
        return "0", "0"

if __name__ == "__main__":
    print(">>> МОНИТОРИНГ ЗАПУЩЕН")
    while True:
        try:
            p5, r5 = get_market_full("5")
            p60, r60 = get_market_full("60")
            change, oi = get_ticker()
            balance = get_balance()

            if p5:
                # Формируем отчет
                msg = (f"💰 Баланс: {balance} USDT\n"
                       f"📊 ETH: ${p5} ({float(change)*100:.2f}% за 24ч)\n"
                       f"📉 RSI 5m: {r5:.2f} | 1h: {r60:.2f}\n"
                       f"🔥 OI: {oi}")
                
                # Запрос к DeepSeek
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": f"Анализируй эти данные Bybit и дай совет по ETH на 1 предложение: {msg}"}]}, timeout=20).json()
                advice = ai_res['choices'][0]['message']['content']

                bot.send_message(CHAT_ID, f"🚀 **BYBIT FULL REPORT**\n\n{msg}\n\n🧠 **AI:** {advice}")
                print(f">>> Отправлено: {p5}")
            
            time.sleep(600) # Проверка каждые 10 минут
        except Exception as e:
            print(f"(!) Сбой: {e}")
            time.sleep(60)
