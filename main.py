import os, telebot, requests
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))

@bot.message_handler(commands=["check"])
def check(m):
    try:
        data = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=ETHUSDT").json()
        price = data["lastPrice"]
        bot.reply_to(m, f"ETH Price: {price}\nСтатус: Работаю через GitHub!")
    except:
        bot.reply_to(m, "Ошибка биржи")

bot.infinity_polling()
