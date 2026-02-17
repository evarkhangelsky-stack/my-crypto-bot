import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import telebot

class TechnicalIndicators:
    """Собственные технические индикаторы [cite: 735, 736]"""
    @staticmethod
    def vwap(high, low, close, volume):
        typical_price = (high + low + close) / 3 [cite: 740]
        return (typical_price * volume).cumsum() / volume.cumsum() [cite: 741]

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff() [cite: 745]
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean() [cite: 746]
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() [cite: 746]
        rs = gain / loss [cite: 746]
        return 100 - (100 / (1 + rs)) [cite: 747]

    @staticmethod
    def ema(close, period):
        return close.ewm(span=period, adjust=False).mean() [cite: 751]

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low [cite: 755]
        tr2 = abs(high - close.shift()) [cite: 756]
        tr3 = abs(low - close.shift()) [cite: 757]
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) [cite: 758]
        return tr.rolling(window=period).mean() [cite: 759]

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        middle = close.rolling(window=period).mean() [cite: 762]
        std_dev = close.rolling(window=period).std() [cite: 762]
        upper = middle + (std_dev * std) [cite: 762]
        lower = middle - (std_dev * std) [cite: 763]
        return upper, middle, lower [cite: 764]

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff() [cite: 768]
        minus_dm = -low.diff() [cite: 769]
        plus_dm[plus_dm < 0] = 0 [cite: 770]
        minus_dm[minus_dm < 0] = 0 [cite: 770]
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1) [cite: 774]
        atr = tr.rolling(window=period).mean() [cite: 775]
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr) [cite: 776]
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr) [cite: 776]
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) [cite: 777]
        return dx.rolling(window=period).mean(), plus_di, minus_di [cite: 778]

class BybitScalpingBot:
    def __init__(self):
        # API keys [cite: 781-784]
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')

        # Initialize Bybit [cite: 790-795]
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        self.bot = telebot.TeleBot(self.telegram_token) [cite: 797]
        
        # Настройки для нескольких монет
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT'] [cite: 799]
        self.timeframe = '5m' [cite: 800]
        self.position = None [cite: 801]
        self.sl_atr_multiplier = 1.2 [cite: 802]
        self.tp_atr_multiplier = 2.0 [cite: 803]
        self.trailing_stop_percent = 0.5 [cite: 804]

    def send_telegram(self, message):
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown') [cite: 812]
        except Exception as e:
            print(f"Telegram error: {e}") [cite: 815]

    def fetch_coinglass_data(self, symbol):
        """Интеграция Coinglass: Ликвидации и Long/Short Ratio"""
        if not self.coinglass_api_key: return "N/A"
        try:
            coin = symbol.split('/')[0]
            url = f"https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol={coin}"
            headers = {"coinglassApi": self.coinglass_api_key}
            response = requests.get(url, headers=headers, timeout=5).json()
            return response['data'][0]['longRate'] if 'data' in response else "N/A"
        except: return "N/A"

    def fetch_news_sentiment(self, symbol):
        """Интеграция CryptoPanic: Новостной фон"""
        if not self.cryptopanic_api_key: return "Neutral"
        try:
            coin = symbol.split('/')[0]
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_api_key}&currencies={coin}"
            response = requests.get(url, timeout=5).json()
            results = response.get('results', [])
            if not results: return "Neutral"
            # Простой скоринг позитивности
            pos = sum([1 for r in results[:5] if r.get('votes', {}).get('positive', 0) > r.get('votes', {}).get('negative', 0)])
            return "Positive" if pos >= 3 else "Negative" if pos <= 1 else "Neutral"
        except: return "Neutral"

    def fetch_ohlcv(self, symbol, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit) [cite: 819]
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) [cite: 821]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') [cite: 823]
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}") [cite: 825]
            return None

    def calculate_indicators(self, df):
        """Расчет твоих индикаторов из Replit [cite: 838]"""
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume']) [cite: 841]
        df['rsi'] = TechnicalIndicators.rsi(df['close']) [cite: 843]
        adx, di_p, di_m = TechnicalIndicators.adx(df['high'], df['low'], df['close']) [cite: 845]
        df['adx'], df['di_plus'], df['di_minus'] = adx, di_p, di_m [cite: 847-849]
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(df['close']) [cite: 853-855]
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close']) [cite: 857]
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20) [cite: 859]
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50) [cite: 859]
        return df

    def get_ai_filter(self, df, signal_type, symbol, news, cg_ratio):
        """Улучшенный ИИ фильтр с учетом новостей и Coinglass [cite: 861]"""
        if not self.deepseek_api_key: return True [cite: 863]
        try:
            last = df.iloc[-1]
            prompt = f"""Analyze trading signal for {symbol}:
            Signal: {signal_type}, Price: {last['close']}, RSI: {last['rsi']:.2f}, ADX: {last['adx']:.2f}
            News Sentiment: {news}, Coinglass L/S Ratio: {cg_ratio}
            EMA Trend: {'Bullish' if last['ema_20'] > last['ema_50'] else 'Bearish'}
            Reply with ONLY: "Approve" or "Reject" """ [cite: 866-874]
            
            response = requests.post('https://api.deepseek.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.deepseek_api_key}', 'Content-Type': 'application/json'},
                json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.3},
                timeout=10) [cite: 875-887]
            
            if response.status_code == 200:
                res = response.json()['choices'][0]['message']['content'].strip().lower() [cite: 889]
                return 'approve' in res [cite: 891]
            return True
        except: return True

    def detect_signal(self, df, symbol):
        """Твоя логика поиска сигналов [cite: 901]"""
        last = df.iloc[-1]
        news = self.fetch_news_sentiment(symbol)
        cg_ratio = self.fetch_coinglass_data(symbol)
        
        signal = None
        # Фаза боковика [cite: 918]
        if last['adx'] < 25:
            if last['close'] <= last['bb_lower'] and last['rsi'] < 30: signal = 'LONG' [cite: 920]
            elif last['close'] >= last['bb_upper'] and last['rsi'] > 70: signal = 'SHORT' [cite: 924]
        # Фаза тренда [cite: 928]
        else:
            if last['close'] > last['vwap'] and last['ema_20'] > last['ema_50'] and 40 < last['rsi'] < 70: signal = 'LONG' [cite: 929]
            elif last['close'] < last['vwap'] and last['ema_20'] < last['ema_50'] and 30 < last['rsi'] < 60: signal = 'SHORT' [cite: 934]

        if signal and self.get_ai_filter(df, signal, symbol, news, cg_ratio):
            entry = last['close'] [cite: 945]
            sl = entry - (self.sl_atr_multiplier * last['atr']) if signal == 'LONG' else entry + (self.sl_atr_multiplier * last['atr']) [cite: 946, 949]
            tp = entry + (self.tp_atr_multiplier * last['atr']) if signal == 'LONG' else entry - (self.tp_atr_multiplier * last['atr']) [cite: 947, 950]
            return signal, {'entry': entry, 'sl': sl, 'tp': tp, 'news': news, 'cg': cg_ratio}
        return None, None

    def run(self):
        self.send_telegram("🚀 Multi-Bot Started (BTC & ETH) + News + Coinglass") [cite: 807]
        while True:
            for symbol in self.symbols:
                try:
                    df = self.fetch_ohlcv(symbol)
                    if df is not None:
                        df = self.calculate_indicators(df)
                        # Управление позицией (если есть) [cite: 1011]
                        # Поиск сигнала
                        signal, params = self.detect_signal(df, symbol)
                        if signal:
                            msg = f"🎯 *Signal {signal}* for {symbol}\nPrice: {params['entry']}\nNews: {params['news']}\nL/S Ratio: {params['cg']}"
                            self.send_telegram(msg)
                except Exception as e:
                    print(f"Error in loop for {symbol}: {e}")
                time.sleep(5)
            time.sleep(300) # Пауза 5 минут между циклами [cite: 1083]

if __name__ == "__main__":
    BybitScalpingBot().run() [cite: 1089]
