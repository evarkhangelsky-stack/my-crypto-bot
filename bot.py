import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import telebot

class TechnicalIndicators:
    """Собственные технические индикаторы из оригинального кода"""
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
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

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
        # Инициализация ключей и настроек
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_key = os.getenv('COINGLASS_API_KEY')
        self.panic_key = os.getenv('CRYPTOPANIC_API_KEY')

        # Настройка биржи
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        self.bot = telebot.TeleBot(self.telegram_token)
        
        # Параметры торговли
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.timeframe = '5m'
        self.position = None
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0

    def send_telegram(self, message):
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Telegram error: {e}")

    def fetch_coinglass(self, symbol):
        """Интеграция Coinglass для фильтрации по ликвидациям"""
        if not self.coinglass_key: return "N/A"
        try:
            coin = symbol.split('/')[0]
            url = f"https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol={coin}"
            headers = {"coinglassApi": self.coinglass_key}
            res = requests.get(url, headers=headers, timeout=5).json()
            return res.get('data', [])[0].get('longRate', 'N/A')
        except: return "N/A"

    def fetch_news(self, symbol):
        """Интеграция CryptoPanic для анализа сантимента"""
        if not self.panic_key: return "Neutral"
        try:
            coin = symbol.split('/')[0]
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.panic_key}&currencies={coin}"
            res = requests.get(url, timeout=5).json()
            posts = res.get('results', [])
            return "Positive" if len(posts) > 2 else "Neutral"
        except: return "Neutral"

    def fetch_ohlcv(self, symbol):
        """Загрузка свечей"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def calculate_indicators(self, df):
        """Расчет тех. анализа"""
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        df['adx'], df['di_plus'], df['di_minus'] = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(df['close'])
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        return df

    def get_ai_filter(self, df, signal_type, symbol, news, cg_data):
        """Улучшенный ИИ фильтр с учетом внешних данных"""
        if not self.deepseek_api_key: return True
        try:
            last = df.iloc[-1]
            prompt = f"""Analyze trading signal:
            Symbol: {symbol} | Signal: {signal_type} | Price: {last['close']}
            RSI: {last['rsi']:.1f} | AD
