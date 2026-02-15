import os, telebot, requests, time

# Загрузка ключей
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_bybit_data():
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        # Получаем данные тикера для цены и OI
        res = requests.get(url, params={"category":"linear","symbol":"ETHUSDT"}, timeout=10).json()
        t = res['result']['list'][0]
        
        # Получаем свечи для RSI
        k_url = "https://api.bybit.com/v5/market/kline"
        k_res = requests.get(k_url, params={"category":"linear","symbol":"ETHUSDT","interval":"5","limit":"20"}, timeout=10).json()
        closes = [float(c[4]) for c in k_res['result']['list'][::-1]]
        
        # Упрощенный RSI
        diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        avg_gain = sum([d for d in diffs[-14:] if d > 0]) / 14
        avg_loss = sum([-d for d in diffs[-14:] if d < 0]) / 14
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss or 0.001))))

        return {"price": float(t['lastPrice']), "rsi": round(rsi, 2), "oi": t['openInterest']}
    except Exception as e:
        print(f"Bybit Error: {e}")
        return None

def get_coinglass_simple():
    """Более надежный запрос к CoinGlass для Free API"""
    try:
        headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
        # Используем эндпоинт, который чаще всего доступен бесплатно
        url = "https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol=ETH"
        res = requests.get(url, headers=headers, timeout=10).json()
        # Если данные есть, берем первый элемент
        if res.get('data') and len(res['data']) > 0:
            return {"ls_ratio": res['data'][0]['v']}
        return {"ls_ratio": "N/A"}
    except:
        return {"ls_ratio": "N/A"}

if __name__ == "__main__":
    print(">>> БОТ ЗАПУЩЕН И ГОТОВ К РАБОТЕ")
    # Отправим тестовое сообщение сразу при запуске
    bot.send_message(CHAT_ID, "🚀 Бот успешно запущен и начинает мониторинг Bybit + CoinGlass!")
    
    while True:
        bb = get_bybit_data()
        cg = get_coinglass_simple()
        
        if bb:
            # Логика определения сетапа
            signal = "LONG" if bb['rsi'] < 30 else "SHORT" if bb['rsi'] > 70 else "NEUTRAL"
            
            prompt = (f"ETH {signal} по {bb['price']}. RSI: {bb['rsi']}, Long/Short: {cg['ls_ratio']}. "
                      f"Дай прогноз за 10 слов.")
            
            try:
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, 
                    timeout=15).json()
                advice = ai_res['choices'][0]['message']['content']
            except:
                advice = "AI анализирует график..."

            msg = (f"💎 **ETH MONITOR**\n\n"
                   f"💵 Price: `${bb['price']}`\n"
                   f"📊 RSI (5m): `{bb['rsi']}`\n"
                   f"⚖️ L/S Ratio: `{cg['ls_ratio']}`\n"
                   f"🎯 OI: `{bb['oi']}`\n\n"
                   f"🧠 **AI:** {advice}")
            
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            print(f">>> Сообщение отправлено: {bb['price']}")
        
        time.sleep(120) # Проверка каждые 2 минуты
