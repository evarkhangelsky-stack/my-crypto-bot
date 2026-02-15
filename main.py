import os, telebot, requests, time
import pandas as pd
import pandas_ta as ta

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_kline_data(symbol="ETHUSDT", interval="1"):
    """Получение свечей и расчет базовых индикаторов"""
    try:
        url = "https://api.bybit.com/v5/market/kline"
        res = requests.get(url, params={"category": "linear", "symbol": symbol, "interval": interval, "limit": "50"}, timeout=10).json()
        df = pd.DataFrame(res['result']['list'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 'tv'])
        df[['o', 'h', 'l', 'c', 'v']] = df[['o', 'h', 'l', 'c', 'v']].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Индикаторы для скальпа
        df['rsi'] = ta.rsi(df['c'], length=14)
        return {"price": df['c'].iloc[-1], "rsi": round(df['rsi'].iloc[-1], 2)}
    except: return None

def get_orderbook_data(symbol="ETHUSDT"):
    """Анализ стакана: дисбаланс сил"""
    try:
        url = "https://api.bybit.com/v5/market/orderbook"
        res = requests.get(url, params={"category": "linear", "symbol": symbol, "limit": "50"}, timeout=10).json()
        bids = sum([float(i[1]) for i in res['result']['b']]) # Объем на покупку
        asks = sum([float(i[1]) for i in res['result']['a']]) # Объем на продажу
        imbalance = (bids / (bids + asks)) * 100
        spread = float(res['result']['a'][0][0]) - float(res['result']['b'][0][0])
        return {"imbalance": round(imbalance, 2), "spread": round(spread, 3)}
    except: return None

def get_ticker_data(symbol="ETHUSDT"):
    """Funding и Open Interest"""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        res = requests.get(url, params={"category": "linear", "symbol": symbol}, timeout=10).json()
        t = res['result']['list'][0]
        return {"oi": t['openInterest'], "funding": t['fundingRate'], "change": t['price24hPcnt']}
    except: return None

if __name__ == "__main__":
    print(">>> СКАНЕР ЗАПУЩЕН")
    while True:
        try:
            # Сбор данных со всех уровней
            m1 = get_kline_data(interval="1")
            m5 = get_kline_data(interval="5")
            m15 = get_kline_data(interval="15")
            book = get_orderbook_data()
            market = get_ticker_data()

            if m1 and book and market:
                # Математический фильтр для скальпа
                signal_type = "NEUTRAL"
                if m1['rsi'] < 30 and book['imbalance'] > 60: signal_type = "SCALP LONG"
                if m1['rsi'] > 70 and book['imbalance'] < 40: signal_type = "SCALP SHORT"

                report = (
                    f"⚡️ **SCALP SCANNER (ETH)**\n"
                    f"💵 Цена: `${m1['price']}` | Спред: `{book['spread']}`\n"
                    f"📊 RSI: 1м:`{m1['rsi']}` | 5м:`{m5['rsi']}` | 15м:`{m15['rsi']}`\n"
                    f"⚖️ Стакан: `Bids {book['imbalance']}% / Asks {100-book['imbalance']}%`\n"
                    f"🎯 OI: `{market['oi']}` | Funding: `{market['funding']}`\n"
                    f"🚨 Тех. сигнал: **{signal_type}**"
                )

                # Отправляем AI для подтверждения
                prompt = f"Ты скальпер. Есть сигнал {signal_type}. Данные: {report}. Подтверждаешь вход? Ответ в 2 предложениях."
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, timeout=20).json()
                
                bot.send_message(CHAT_ID, report + "\n\n🧠 **AI Анализ:**\n" + ai_res['choices'][0]['message']['content'], parse_mode="Markdown")
            
            time.sleep(300) # Проверка каждые 5 минут для интрадея
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(60)
