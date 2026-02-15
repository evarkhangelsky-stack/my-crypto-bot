import os, telebot, requests, time

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_eth_data():
    try:
        base_url = "https://api.bybit.com/v5/market"
        # 1. Свечи для RSI (считаем сами без лишних библиотек)
        k_res = requests.get(f"{base_url}/kline", params={"category":"linear","symbol":"ETHUSDT","interval":"5","limit":"50"}).json()
        closes = [float(c[4]) for c in k_res['result']['list'][::-1]]
        
        # Простой расчет RSI
        diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in diffs]
        losses = [-d if d < 0 else 0 for d in diffs]
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        rs = avg_gain / (avg_loss if avg_loss != 0 else 0.001)
        rsi = 100 - (100 / (1 + rs))

        # 2. Стакан (Силы покупателей и продавцов)
        ob = requests.get(f"{base_url}/orderbook", params={"category":"linear","symbol":"ETHUSDT","limit":"25"}).json()
        bids_vol = sum([float(b[1]) for b in ob['result']['b']])
        asks_vol = sum([float(a[1]) for a in ob['result']['a']])
        imbalance = (bids_vol / (bids_vol + asks_vol)) * 100

        # 3. OI и текущая цена
        t_res = requests.get(f"{base_url}/tickers", params={"category":"linear","symbol":"ETHUSDT"}).json()
        t = t_res['result']['list'][0]

        return {
            "price": t['lastPrice'],
            "rsi": round(rsi, 2),
            "imbalance": round(imbalance, 2),
            "oi": t['openInterest'],
            "funding": t['fundingRate']
        }
    except Exception as e:
        print(f"Ошибка сбора данных: {e}")
        return None

if __name__ == "__main__":
    print(">>> МОНИТОРИНГ ETH ЗАПУЩЕН")
    while True:
        data = get_eth_data()
        if data:
            # Условие для сигнала (чтобы не спамить просто так)
            is_urgent = data['rsi'] < 35 or data['rsi'] > 65 or data['imbalance'] > 65 or data['imbalance'] < 35
            
            prompt = f"Данные ETH: цена {data['price']}, RSI {data['rsi']}, доминирование в стакане {data['imbalance']}%. Дай совет скальперу в 1 предложение."
            
            try:
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()
                advice = ai_res['choices'][0]['message']['content']
            except:
                advice = "AI анализирует график..."

            msg = (f"💎 **ETH MONITOR**\n\n"
                   f"💵 Цена: `${data['price']}`\n"
                   f"📊 RSI (5m): `{data['rsi']}`\n"
                   f"⚖️ Стакан: `{data['imbalance']}%` в покупках\n"
                   f"🎯 OI: `{data['oi']}` | Funding: `{data['funding']}`\n\n"
                   f"🧠 **AI:** {advice}")
            
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            print(f">>> Отчет отправлен: {data['price']}")

        time.sleep(120) # Проверка каждые 2 минуты
