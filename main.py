# ... (внутри блока if signal:)

# Уточняем промпт, чтобы AI не галлюцинировал ценами
prompt = (f"ТЫ ПРО-ТРЕЙДЕР. ТЕКУЩАЯ ЦЕНА ETH: {m['price']}. ЭТО ЕДИНСТВЕННАЯ ВЕРНАЯ ЦЕНА. "
          f"Данные: Тренд {m['trend']}, RSI {m['rsi']}, ATR {m['atr']}, Vol Burst: {m['high_vol']}. "
          f"Напиши ПОЧЕМУ мы входим. НЕ ПИШИ ЦЕНЫ, я их подставлю сам. Пиши только логику за 20 слов.")

try:
    ai_res = requests.post("https://api.deepseek.com/chat/completions", 
        headers={"Authorization": f"Bearer {DS_KEY}"},
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, 
        timeout=15).json()
    advice = ai_res['choices'][0]['message']['content']
except:
    advice = "Логика: Вход по тренду с подтверждением объема и ATR."

# Формируем сообщение, где ЦЕНЫ подставляет Python (он не ошибается), а AI пишет только ТЕКСТ
msg = (f"🚨 **SMART SIGNAL: {signal}**\n\n"
       f"📥 Вход: `{m['price']}` (ТЕКУЩАЯ РЫНОЧНАЯ)\n"
       f"🛡 Stop (ATR): `{round(sl, 2)}` | 🎯 TP: `{round(tp, 2)}`\n\n"
       f"📊 **Data Stack:**\n"
       f"- Trend: `{m['trend']}` | RSI: `{m['rsi']}`\n"
       f"- ATR: `{m['atr']}` | Vol Burst: `{'YES' if m['high_vol'] else 'NO'}`\n\n"
       f"🧠 **AI АНАЛИЗ:**\n{advice}")

bot.send_message(CHAT_ID, msg, parse_mode="Markdown")
