import os, telebot, requests, time

# Конфигурация
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")

bot = telebot.TeleBot(TOKEN)

def get_bybit_data():
    """Сбор данных с Bybit: Цена, RSI, Стакан, OI"""
    try:
        base_url = "https://api.bybit.com/v5/market"
        symbol = "ETHUSDT"
        # Свечи для RSI
        k_res = requests.get(f"{base_url}/kline", params={"category":"linear","symbol":symbol,"interval":"5","limit":"50"}, timeout=10).json()
        closes = [float(c[4]) for c in k_res['result']['list'][::-1]]
        
        # Чистый расчет RSI
        diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        avg_gain = sum([d for d in diffs[-14:] if d > 0]) / 14
        avg_loss = sum([-d for d in diffs[-14:] if d < 0]) / 14
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss or 0.001))))
        
        # Стакан
        ob = requests.get(f"{base_url}/orderbook", params={"category":"linear","symbol":symbol,"limit":"50"}, timeout=10).json()
        bids = sum([float(b[1]) for b in ob['result']['b']])
        asks = sum([float(a[1]) for a in ob['result']['a']])
        
        # Тикер (Цена, OI, Funding)
        t_res = requests.get(f"{base_url}/tickers", params={"category":"linear","symbol":symbol}, timeout=10).json()
        t = t_res['result']['list'][0]
        
        return {
            "price": float(t['lastPrice']),
            "rsi": round(rsi, 2),
            "imbalance": round((bids / (bids + asks)) * 100, 2),
            "oi": t['openInterest'],
            "funding": t['fundingRate']
        }
    except Exception as e:
        print(f"Bybit error: {e}")
        return None

def get_coinglass_data():
    """Данные CoinGlass: Ликвидации и Long/Short Ratio"""
    try:
        headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
        # Ликвидации ETH за час
        res_liq = requests.get("https://open-api.coinglass.com/public/v2/liquidation_info?symbol=ETH", headers=headers, timeout=10).json()
        # Long/Short Ratio (агрегированный)
        res_ls = requests.get("https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol=ETH", headers=headers, timeout=10).json()
        
        return {
            "liq_buy": res_liq['data'][0]['buyVol'] if res_liq.get('data') else 0,
            "liq_sell": res_liq['data'][0]['sellVol'] if res_liq.get('data') else 0,
            "ls_ratio": res_ls['data'][0]['v'] if res_ls.get('data') else 1.0
        }
    except:
        return {"liq_buy": 0, "liq_sell": 0, "ls_ratio": 1.0}

def get_binance_price():
    """Цена с Binance (бесплатно, без ключа)"""
    try:
        res = requests.get("https://api.binance.com/api/3/ticker/price?symbol=ETHUSDT", timeout=10).json()
        return float(res['price'])
    except:
        return None

if __name__ == "__main__":
    print(">>> ЗАПУСК ВСЕВИДЯЩЕГО ОКА (BYBIT + BINANCE + COINGLASS)")
    while True:
        bb = get_bybit_data()
        cg = get_coinglass_data()
        bin_p = get_binance_price()

        if bb and bin_p:
            diff = round(bb['price'] - bin_p, 2)
            
            # Промпт для AI теперь включает данные с трех площадок
            prompt = (f"ETH Анализ: Bybit ${bb['price']}, Binance ${bin_p}. RSI: {bb['rsi']}, "
                      f"Ликвидации лонгов: ${cg['liq_sell']}, Ликвидации шортов: ${cg['liq_buy']}. "
                      f"Long/Short Ratio: {cg['ls_ratio']}. Дай краткий совет трейдеру.")

            try:
                ai_res = requests.post("https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {DS_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, 
                    timeout=15).json()
                advice = ai_res['choices'][0]['message']['content']
            except:
                advice = "DeepSeek анализирует рыночную ситуацию..."

            msg = (f"🛸 **ETH GLOBAL DATA**\n\n"
                   f"💵 Bybit: `${bb['price']}` (Binance Diff: `{diff}`)\n"
                   f"🔥 Liquidation (1h): 🔴 `${cg['liq_sell']}` | 🟢 `${cg['liq_buy']}`\n"
                   f"⚖️ L/S Ratio: `{cg['ls_ratio']}`\n"
                   f"📊 RSI: `{bb['rsi']}` | Стакан: `{bb['imbalance']}%` 📈\n"
                   f"🎯 OI: `{bb['oi']}` | Funding: `{bb['funding']}`\n\n"
                   f"🧠 **AI:** {advice}")
            
            bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
            print(f">>> Отчет отправлен: {bb['price']}")
        
        time.sleep(300) # Проверка раз в 5 минут
