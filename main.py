import os, telebot, requests, time

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_coinglass_data():
    """Получение ликвидаций и соотношения Long/Short"""
    try:
        headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
        # Ликвидации за 1 час (агрегированные)
        url_liq = "https://open-api.coinglass.com/public/v2/liquidation_info?symbol=ETH"
        res_liq = requests.get(url_liq, headers=headers, timeout=10).json()
        
        # Соотношение Long/Short
        url_ls = "https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol=ETH"
        res_ls = requests.get(url_ls, headers=headers, timeout=10).json()
        
        return {
            "liq_buy": res_liq['data'][0]['buyVol'] if res_liq.get('data') else 0,
            "liq_sell": res_liq['data'][0]['sellVol'] if res_liq.get('data') else 0,
            "ls_ratio": res_ls['data'][0]['v'] if res_ls.get('data') else 1.0
        }
    except: return None

def get_binance_price():
    """Сверка цены с лидером рынка"""
    try:
        res = requests.get("https://api.binance.com/api/3/ticker/price?symbol=ETHUSDT").json()
        return float(res['price'])
    except: return None

# ... тут остаются твои функции get_data() и get_market_context() для Bybit ...

if __name__ == "__main__":
    print(">>> ЗАПУСК ВСЕВИДЯЩЕГО ОКА (BYBIT + BINANCE + COINGLASS)")
    while True:
        # Собираем данные со всех фронтов
        bybit = get_data(interval="5") # Твоя функция из прошлого кода
        ctx = get_market_context()     # Твоя функция из прошлого кода
        cg = get_coinglass_data()
        binance_p = get_binance_price()

        if bybit and cg and binance_p:
            # Расчет арбитражной разницы
            diff = round(bybit['price'] - binance_p, 2)
            
            # Математика сигнала с учетом ликвидаций
            is_urgent = (bybit['rsi'] < 30 and cg['liq_sell'] > 100000) # Сигнал, если RSI низкий И много ликвидаций
            
            # Формируем запрос для DeepSeek с ПОЛНЫМИ данными
            prompt = (f"Данные ETH: Bybit ${bybit['price']}, Binance ${binance_p}. "
                      f"Ликвидации шортов: ${cg['liq_buy']}, лонгов: ${cg['liq_sell']}. "
                      f"Long/Short Ratio: {cg['ls_ratio']}. RSI: {bybit['rsi']}. "
                      f"Дай прогноз скальперу.")

            try:
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}).json()
                advice = ai_res['choices'][0]['message']['content']
            except: advice = "AI анализирует поток данных..."

            msg = (f"🌍 **GLOBAL ETH RADAR**\n\n"
                   f"📊 **Prices:** Bybit `${bybit['price']}` | Bin `${binance_p}` (Diff: `{diff}`)\n"
                   f"🔥 **Liquids (1h):** 🟢 `${cg['liq_buy']}` | 🔴 `${cg['liq_sell']}`\n"
                   f"⚖️ **L/S Ratio:** `{cg['ls_ratio']}`\n"
                   f"📉 **Technical:** RSI `{bybit['rsi']}` | Imbalance `{round(ctx['imbalance'], 2)}%`\n\n"
                   f"🧠 **AI:** {advice}")
            
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            print(f">>> Полный отчет отправлен")

        time.sleep(300) # Проверка раз в 5 минут
