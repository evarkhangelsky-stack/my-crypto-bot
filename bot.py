import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import telebot

class TechnicalIndicators:
    """Технические индикаторы из оригинального PDF"""
    @staticmethod
    def vwap(high, low, close, volume):
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(close, period):
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        return middle + (std_dev * std), middle, middle - (std_dev * std)

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean(), plus_di, minus_di

class BybitScalpingBot:
    def __init__(self):
        # Инициализация API ключей
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_key = os.getenv('COINGLASS_API_KEY')
        self.panic_key = os.getenv('CRYPTOPANIC_API_KEY')

        # Подключение к Bybit
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        self.bot = telebot.TeleBot(self.telegram_token)
        
        # Настройки стратегии
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.timeframe = '5m'
        self.position = None # Хранит данные об открытой сделке
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5

    def send_telegram(self, message):
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Telegram error: {e}")

    def fetch_coinglass(self, symbol):
        """Данные Coinglass: Соотношение лонг/шорт"""
        if not self.coinglass_key: return "50/50"
        try:
            coin = symbol.split('/')[0]
            url = f"https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol={coin}"
            headers = {"coinglassApi": self.coinglass_key}
            res = requests.get(url, headers=headers, timeout=5).json()
            return res['data'][0]['longRate'] if 'data' in res else "50"
        except: return "50"

    def fetch_news(self, symbol):
        """Данные CryptoPanic: Сантимент"""
        if not self.panic_key: return "Neutral"
        try:
            coin = symbol.split('/')[0]
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.panic_key}&currencies={coin}"
            res = requests.get(url, timeout=5).json()
            return "Positive" if len(res.get('results', [])) > 1 else "Neutral"
        except: return "Neutral"

    def calculate_indicators(self, df):
        """Индикаторы из твоего оригинального кода"""
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        df['adx'], _, _ = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        df['bb_up'], df['bb_mid'], df['bb_low'] = TechnicalIndicators.bollinger_bands(df['close'])
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        return df

    def get_ai_filter(self, df, signal, symbol, news, cg):
        """Финальный фильтр DeepSeek"""
        if not self.deepseek_api_key: return True
        try:
            last = df.iloc[-1]
            prompt = f"""Trading Signal Analysis for {symbol}:
            Signal: {signal} | Price: {last['close']} | RSI: {last['rsi']:.1f}
            News Sentiment: {news} | Coinglass Long Rate: {cg}%
            Reply ONLY with "Approve" or "Reject"."""
            
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.deepseek_api_key}', 'Content-Type': 'application/json'},
                json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.3},
                timeout=10
            )
            return 'approve' in response.json()['choices'][0]['message']['content'].lower()
        except: return True

    def check_market(self, symbol):
        """Анализ конкретной монеты"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = self.calculate_indicators(df)
        
        last = df.iloc[-1]
        news = self.fetch_news(symbol)
        cg = self.fetch_coinglass(symbol)
        
        signal = None
        # Логика из PDF: Боковик или Тренд
        if last['adx'] < 25:
            if last['close'] <= last['bb_low'] and last['rsi'] < 30: signal = 'LONG'
            elif last['close'] >= last['bb_up'] and last['rsi'] > 70: signal = 'SHORT'
        else:
            if last['close'] > last['vwap'] and last['ema_20'] > last['ema_50']: signal = 'LONG'
            elif last['close'] < last['vwap'] and last['ema_20'] < last['ema_50']: signal = 'SHORT'

        if signal and self.get_ai_filter(df, signal, symbol, news, cg):
            self.send_telegram(f"🎯 *Signal {signal}* for {symbol}\nPrice: {last['close']}\nNews: {news}\nL/S Ratio: {cg}%")
            # Здесь можно добавить self.place_order(symbol, signal, last)

    def run(self):
        self.send_telegram("🚀 Бот запущен: BTC & ETH + News + Coinglass")
        while True:
            for symbol in self.symbols:
                try:
                    print(f"Checking {symbol}...")
                    self.check_market(symbol)
                except Exception as e:
                    print(f"Error {symbol}: {e}")
                time.sleep(2)
            time.sleep(300) # Проверка каждые 5 минут

if __name__ == "__main__":
    BybitScalpingBot().run()
