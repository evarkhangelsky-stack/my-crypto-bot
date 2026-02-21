import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import telebot
import csv
import json

class TechnicalIndicators:
    """Собственные технические индикаторы"""

    @staticmethod
    def vwap(high, low, close, volume):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def rsi(close, period=14):
        """Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(close, period):
        """Exponential Moving Average"""
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        """Bollinger Bands"""
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=d_period).mean()
        return k_smooth, d

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


class FREDAnalyzer:
    """Анализ макроэкономических данных из FRED (Federal Reserve)"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred"
        self.cache = {}
        
    def get_inflation_data(self):
        """Получает последние данные по инфляции (CPI)"""
        cache_key = 'inflation'
        if cache_key in self.cache:
            data, time = self.cache[cache_key]
            if datetime.now() - time < timedelta(days=1):  # Кэш на день
                return data
        
        if not self.api_key:
            return None
            
        try:
            # Серия CPIAUCSL - Consumer Price Index for All Urban Consumers
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': 'CPIAUCSL',
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 2  # Последние 2 значения для расчета изменения
            }
            
            response = requests.get(url, params=params, timeout=10).json()
            observations = response['observations']
            
            if len(observations) >= 2:
                current = float(observations[0]['value'])
                previous = float(observations[1]['value'])
                change_pct = ((current - previous) / previous) * 100
                
                result = {
                    'current_cpi': current,
                    'previous_cpi': previous,
                    'monthly_change_pct': change_pct,
                    'date': observations[0]['date'],
                    'trend': 'INCREASING' if change_pct > 0.2 else 'DECREASING' if change_pct < -0.2 else 'STABLE',
                    'signal': self._get_inflation_signal(change_pct)
                }
                
                self.cache[cache_key] = (result, datetime.now())
                return result
                
        except Exception as e:
            print(f"FRED error: {e}")
            return None
    
    def _get_inflation_signal(self, change_pct):
        """Интерпретирует данные по инфляции"""
        if change_pct > 0.5:
            return 'BEARISH'  # Высокая инфляция - риск повышения ставок (плохо для рискованных активов)
        elif change_pct > 0.2:
            return 'CAUTION'
        elif change_pct < -0.2:
            return 'BULLISH'   # Дефляция - риск стимулирования (хорошо для рискованных активов)
        else:
            return 'NEUTRAL'
    
    def get_interest_rate(self):
        """Получает текущую ставку ФРС"""
        cache_key = 'interest_rate'
        if cache_key in self.cache:
            data, time = self.cache[cache_key]
            if datetime.now() - time < timedelta(days=1):
                return data
                
        if not self.api_key:
            return None
            
        try:
            # Серия FEDFUNDS - Effective Federal Funds Rate
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10).json()
            rate = float(response['observations'][0]['value'])
            result = {
                'rate': rate,
                'date': response['observations'][0]['date'],
                'environment': 'HIGH_RATE' if rate > 5 else 'MEDIUM_RATE' if rate > 2 else 'LOW_RATE'
            }
            self.cache[cache_key] = (result, datetime.now())
            return result
        except Exception as e:
            print(f"FRED rate error: {e}")
            return None


class MultiTimeframeAnalyzer:
    """Анализирует все таймфреймы от 15m до 1d"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.timeframes = {
            '15m': {'weight': 0.15, 'name': '15-минутный', 'cache_ttl': 5},    # Кэш 5 мин
            '30m': {'weight': 0.20, 'name': '30-минутный', 'cache_ttl': 10},   # Кэш 10 мин
            '1h': {'weight': 0.25, 'name': 'Часовой', 'cache_ttl': 15},         # Кэш 15 мин
            '4h': {'weight': 0.25, 'name': '4-часовой', 'cache_ttl': 60},       # Кэш 1 час
            '1d': {'weight': 0.15, 'name': 'Дневной', 'cache_ttl': 240},        # Кэш 4 часа
        }
        self.cache = {}
        
    def get_trend_context(self, symbol):
        """
        Возвращает контекст тренда со всех ТФ
        """
        context = {
            'trend': 'NEUTRAL',
            'strength': 0,
            'description': '↔️ Смешанный тренд',
            'details': {},
            'alignment': 'NEUTRAL',
            'score': 0
        }
        
        total_score = 0
        total_weight = 0
        directions = []
        
        for tf, config in self.timeframes.items():
            df = self._get_cached_data(symbol, tf, config['cache_ttl'])
            if df is None or len(df) < 30:
                continue
                
            tf_trend, tf_score, tf_desc = self._analyze_timeframe(df)
            
            context['details'][tf] = {
                'trend': tf_trend,
                'score': tf_score,
                'description': tf_desc
            }
            
            directions.append(tf_trend)
            total_score += tf_score * config['weight']
            total_weight += config['weight']
        
        if total_weight > 0:
            avg_score = total_score / total_weight
            context['score'] = avg_score
            
            # Определяем общий тренд
            if avg_score > 0.2:
                context['trend'] = 'BULL'
                context['strength'] = avg_score
                context['description'] = f"⬆️ Бычий тренд (сила {avg_score:.2f})"
            elif avg_score < -0.2:
                context['trend'] = 'BEAR'
                context['strength'] = abs(avg_score)
                context['description'] = f"⬇️ Медвежий тренд (сила {abs(avg_score):.2f})"
            
            # Проверяем согласованность ТФ
            if all(d == 'BULL' for d in directions if d != 'NEUTRAL'):
                context['alignment'] = 'STRONG_BULL'
            elif all(d == 'BEAR' for d in directions if d != 'NEUTRAL'):
                context['alignment'] = 'STRONG_BEAR'
            elif len(set(directions)) == 1:
                context['alignment'] = 'CONSISTENT'
            else:
                context['alignment'] = 'MIXED'
        
        return context
    
    def _get_cached_data(self, symbol, timeframe, cache_ttl_minutes):
        """Получает данные с кэшированием"""
        now = datetime.now(timezone.utc)
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if now - timestamp < timedelta(minutes=cache_ttl_minutes):
                return data
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
            df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
            df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
            
            self.cache[cache_key] = (df, now)
            print(f"[{now}] MTF: Загружен {timeframe} для {symbol}")
            return df
            
        except Exception as e:
            print(f"MTF error loading {timeframe}: {e}")
            return None
    
    def _analyze_timeframe(self, df):
        """
        Анализирует один таймфрейм и возвращает:
        - направление тренда (BULL/BEAR/NEUTRAL)
        - силу тренда (-1 до 1)
        - описание
        """
        last = df.iloc[-1]
        prev = df.iloc[-5]  # 5 свечей назад
        
        score = 0
        reasons = []
        
        # 1. EMA alignment
        if last['ema_20'] > last['ema_50']:
            score += 0.4
            reasons.append("EMA20 > EMA50")
        else:
            score -= 0.4
            reasons.append("EMA20 < EMA50")
        
        # 2. Цена относительно EMA20
        if last['close'] > last['ema_20']:
            score += 0.3
            reasons.append("Цена выше EMA20")
        else:
            score -= 0.3
            reasons.append("Цена ниже EMA20")
        
        # 3. RSI направление
        if last['rsi'] > 50:
            score += 0.2
            reasons.append(f"RSI {last['rsi']:.1f} > 50")
        else:
            score -= 0.2
            reasons.append(f"RSI {last['rsi']:.1f} < 50")
        
        # 4. Моментум (сравнение с 5 свечей назад)
        if last['close'] > prev['close']:
            score += 0.1
            reasons.append("Цена растет")
        else:
            score -= 0.1
            reasons.append("Цена падает")
        
        # Определяем направление
        if score > 0.3:
            trend = 'BULL'
            desc = f"⬆️ Бычий ({', '.join(reasons[:2])})"
        elif score < -0.3:
            trend = 'BEAR'
            desc = f"⬇️ Медвежий ({', '.join(reasons[:2])})"
        else:
            trend = 'NEUTRAL'
            desc = f"↔️ Нейтральный"
        
        return trend, score, desc


class BybitScalpingBot:
    def __init__(self):
        # API keys from environment
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.cryptopanic_api_plan = os.getenv('CRYPTOPANIC_API_PLAN', 'developer')
        self.fred_api_key = os.getenv('FRED_API_KEY')

        # Кэш для CryptoPanic
        self.cryptopanic_cache = []
        self.cryptopanic_cache_time = None
        self.cryptopanic_cache_duration = timedelta(hours=1)

        required = [self.api_key, self.api_secret, self.telegram_token, self.telegram_chat_id]
        if not all(required):
            raise ValueError("Missing required environment variables")

        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        for symbol in self.symbols:
            try:
                self.exchange.set_margin_mode('cross', symbol)
                self.exchange.set_leverage(5, symbol)
                print(f"[{datetime.now(timezone.utc)}] Leverage 5x and cross for {symbol}")
            except Exception as e:
                print(f"Error setting leverage/margin: {e}")

        self.bot = telebot.TeleBot(self.telegram_token)
        self.timeframe = '5m'
        self.positions = {s: None for s in self.symbols}

        # Торговые параметры
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5
        self.taker_fee = 0.0006

        # Новые параметры управления позициями
        self.max_hold_time = timedelta(hours=2)      # Максимальное время удержания
        self.min_profit_for_breakeven = 0.3          # % при котором передвигаем SL в безубыток
        self.trailing_activation = 0.5                # % для активации трейлинга
        self.trailing_distance = 0.3                   # % отступа трейлинга
        self.min_balance_for_trading = 50              # Минимальный баланс для торговли

        # Daily loss limit
        self.daily_loss_limit_pct = -4.2
        self.last_day = None
        self.day_start_equity = None
        self.trading_paused_until = None

        # CSV для лога сделок
        self.trade_log_file = "trade_log.csv"
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'entry', 'exit', 'size', 'pnl', 'pnl_pct',
                    'rsi', 'adx', 'vwap', 'ema_20', 'ema_50', 'atr', 'bb_upper', 'bb_lower',
                    'stoch_k', 'stoch_d', 'macd_hist', 'bid_ratio', 'hold_time_minutes'
                ])

        # Мультитаймфреймовый анализатор
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.exchange)
        self.mtf_context = {}
        self.mtf_last_update = {}

        # FRED анализатор
        self.fred = FREDAnalyzer(self.fred_api_key)
        self.macro_context = {}
        self.macro_last_update = None

        print(f"[{datetime.now(timezone.utc)}] Bot initialized for {self.symbols}")
        self.send_telegram(f"Bot started\nSymbols: {' '.join(self.symbols)}\nTimeframe: {self.timeframe}")

    def send_telegram(self, message):
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Telegram error: {e}")

    def fetch_ohlcv(self, symbol, limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_orderbook_data(self, symbol):
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=50)
            total_bids = sum(bid[1] for bid in orderbook['bids'])
            total_asks = sum(ask[1] for ask in orderbook['asks'])
            total = total_bids + total_asks
            bid_ratio = (total_bids / total) * 100 if total > 0 else 50
            return {'bid_ratio': bid_ratio, 'total_volume': total}
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {'bid_ratio': 50, 'total_volume': 0}

    def fetch_coinglass_data(self, symbol_base):
        if not self.coinglass_api_key:
            return {}
        try:
            headers = {'cg-api-key': self.coinglass_api_key}
            url = f"https://open-api.coinglass.com/public/v2/long_short?symbol={symbol_base}&time_type=h1"
            res = requests.get(url, headers=headers, timeout=10).json()
            return res.get('data', [])[0] if res.get('success') else {}
        except Exception as e:
            print(f"Coinglass error: {e}")
            return {}

    def fetch_cryptopanic_news(self):
        """Запрашивает новости с кэшированием на 1 час"""
        if not self.cryptopanic_api_key:
            return []

        now = datetime.now(timezone.utc)
        
        if self.cryptopanic_cache and self.cryptopanic_cache_time:
            if now - self.cryptopanic_cache_time < self.cryptopanic_cache_duration:
                print(f"[{now}] CryptoPanic: используем кэшированные новости")
                return self.cryptopanic_cache

        try:
            url = f"https://cryptopanic.com/api/{self.cryptopanic_api_plan}/v2/posts/?auth_token={self.cryptopanic_api_key}&kind=news"
            res = requests.get(url, timeout=10)
            
            if res.status_code == 429:
                print(f"[{now}] CryptoPanic: rate limit (429), возвращаем кэш")
                return self.cryptopanic_cache if self.cryptopanic_cache else []
            
            if res.status_code != 200:
                print(f"[{now}] CryptoPanic: HTTP error {res.status_code}")
                return self.cryptopanic_cache if self.cryptopanic_cache else []
            
            data = res.json()
            self.cryptopanic_cache = data.get('results', [])[:5]
            self.cryptopanic_cache_time = now
            print(f"[{now}] CryptoPanic: загружено {len(self.cryptopanic_cache)} новостей")
            return self.cryptopanic_cache
            
        except Exception as e:
            print(f"[{now}] CryptoPanic error: {e}")
            return self.cryptopanic_cache if self.cryptopanic_cache else []

    def update_macro_context(self):
        """Обновляет макроэкономический контекст (раз в день)"""
        now = datetime.now(timezone.utc)
        
        if self.macro_last_update and now - self.macro_last_update < timedelta(days=1):
            return self.macro_context
        
        inflation = self.fred.get_inflation_data()
        rates = self.fred.get_interest_rate()
        
        self.macro_context = {
            'inflation': inflation,
            'rates': rates,
            'timestamp': now
        }
        self.macro_last_update = now
        
        if inflation:
            print(f"[{now}] 📊 Макроэкономика: Инфляция {inflation['monthly_change_pct']:.2f}% ({inflation['signal']})")
        if rates:
            print(f"[{now}] 📊 Ставка ФРС: {rates['rate']}% ({rates['environment']})")
        
        return self.macro_context

    def get_macro_signal(self):
        """Возвращает макроэкономический сигнал для трейдинга"""
        if not self.macro_context:
            return 'NEUTRAL'
        
        inflation = self.macro_context.get('inflation', {})
        rates = self.macro_context.get('rates', {})
        
        # Комбинируем сигналы
        if inflation.get('signal') == 'BEARISH' and rates.get('environment') == 'HIGH_RATE':
            return 'BEARISH'
        elif inflation.get('signal') == 'BULLISH' and rates.get('environment') == 'LOW_RATE':
            return 'BULLISH'
        else:
            return 'NEUTRAL'

    def get_ai_filter(self, symbol, df, signal, orderbook, coinglass, news):
        """Смягченный AI фильтр с подробным промптом"""
        if not self.deepseek_api_key:
            return True
        try:
            last = df.iloc[-1]
            news_text = "\n".join(n.get('title', '') for n in news[:3])
            
            # Добавляем макроэкономический контекст
            macro = self.get_macro_signal()
            
            # Определяем состояние рынка для промпта
            rsi_state = 'oversold' if last['rsi'] < 30 else 'overbought' if last['rsi'] > 70 else 'neutral'
            adx_state = 'trending' if last['adx'] > 25 else 'ranging'
            vwap_state = 'above' if last['close'] > last['vwap'] else 'below'
            ema_state = 'BULLISH' if last['ema_20'] > last['ema_50'] else 'BEARISH'
            
            # Определяем позицию относительно Bollinger
            if last['close'] >= last['bb_upper']:
                bb_state = 'UPPER (overbought)'
            elif last['close'] <= last['bb_lower']:
                bb_state = 'LOWER (oversold)'
            else:
                bb_state = 'MIDDLE'
            
            # Определяем доминирование в стакане
            order_flow = 'buyers' if orderbook['bid_ratio'] > 50 else 'sellers'
            
            prompt = f"""Analyze this {signal} scalp trade for {symbol}:

Price: ${last['close']:.2f}

TECHNICAL ANALYSIS:
• RSI: {last['rsi']:.1f} ({rsi_state})
• ADX: {last['adx']:.1f} ({adx_state})
• VWAP: price {vwap_state} VWAP (${last['vwap']:.2f})
• EMA20/50: {ema_state} (20: ${last['ema_20']:.2f}, 50: ${last['ema_50']:.2f})
• Bollinger: price at {bb_state} (upper: ${last['bb_upper']:.2f}, lower: ${last['bb_lower']:.2f})
• ATR: ${last['atr']:.2f} (volatility measure)

ORDER FLOW:
• Bid/Ask Ratio: {orderbook['bid_ratio']:.1f}% ({order_flow} dominate)

MARKET SENTIMENT:
• Coinglass L/S: {coinglass.get('longShortRatio', 'N/A')}
• Macro Outlook: {macro}
• News: {news_text[:150]}...

SCALP TRADING CONTEXT:
- Target: 0.5-1% profit
- Stop loss: tight (1.2x ATR)
- Holding time: minutes to hours
- We don't need perfect setups, just decent probability

QUESTION: Based on ALL available data, would you take this {signal} scalp trade?
Reply with ONLY "YES" or "NO"."""
            
            print(f"[{datetime.now(timezone.utc)}] 🤔 Asking DeepSeek about {symbol} {signal}...")
            
            res = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.3,
                    'max_tokens': 10
                },
                timeout=15
            ).json()
            
            answer = res['choices'][0]['message']['content'].strip().upper()
            print(f"[{datetime.now(timezone.utc)}] 🤖 DeepSeek verdict: {answer}")
            
            # Проверяем наличие положительного ответа
            positive = any(word in answer for word in ['YES', 'SURE', 'GOOD', 'OK', 'TAKE', 'YEP'])
            
            if positive:
                print(f"✅ DeepSeek APPROVED {symbol} {signal}")
            else:
                print(f"❌ DeepSeek REJECTED {symbol} {signal}")
            
            return positive
            
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] ⚠️ AI error: {e}")
            return True

    def calculate_indicators(self, df):
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=14)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'], period=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        adx, di_plus, di_minus = TechnicalIndicators.adx(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        df['stoch_k'], df['stoch_d'] = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.macd(df['close'])
        return df

    def check_daily_loss_limit(self):
        now = datetime.now(timezone.utc)
        current_day = now.date()

        if self.last_day != current_day:
            try:
                bal = self.exchange.fetch_balance()
                # Улучшенный парсинг баланса с fallback
                if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                    equity = float(bal['info']['result']['list'][0]['totalEquity'])
                else:
                    equity = float(bal['USDT']['total']) if 'USDT' in bal and 'total' in bal['USDT'] else 100.0
                
                self.day_start_equity = equity
                self.last_day = current_day
                self.trading_paused_until = None
                print(f"[{now}] Новый день UTC. Депозит на начало: {equity:.2f} USDT")
                self.send_telegram(f"Новый день UTC. Баланс на старте: {equity:.2f} USDT")
            except Exception as e:
                print(f"Не удалось получить equity для лимита: {e}")
                return True

        if self.trading_paused_until and now < self.trading_paused_until:
            print(f"[{now}] Торговля остановлена до {self.trading_paused_until} из-за лимита убытков")
            return False

        if self.day_start_equity is None:
            return True

        try:
            bal = self.exchange.fetch_balance()
            if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                current_equity = float(bal['info']['result']['list'][0]['totalEquity'])
            else:
                current_equity = float(bal['USDT']['total']) if 'USDT' in bal and 'total' in bal['USDT'] else 100.0
                
            pnl_pct = (current_equity - self.day_start_equity) / self.day_start_equity * 100
            print(f"[{now}] Текущий PnL дня: {pnl_pct:.2f}% (начало: {self.day_start_equity:.2f}, сейчас: {current_equity:.2f})")

            if pnl_pct <= self.daily_loss_limit_pct:
                self.trading_paused_until = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                msg = f"🚨 Дневной лимит убытков -{self.daily_loss_limit_pct}% достигнут! Торговля остановлена до {self.trading_paused_until.strftime('%Y-%m-%d %H:%M UTC')}"
                print(msg)
                self.send_telegram(msg)
                return False
            return True
        except Exception as e:
            print(f"Ошибка проверки лимита: {e}")
            return True

    def sideways_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        rsi = last['rsi']
        stoch_k = last['stoch_k']
        macd_hist = last['macd_hist']
        bb_lower = last['bb_lower']
        bb_upper = last['bb_upper']
        bid_ratio = ob['bid_ratio']

        # LONG сигнал
        if price <= bb_lower and rsi < 35 and stoch_k < 20 and bid_ratio > 55:
            strength = 0.9 if rsi < 30 and stoch_k < 15 and bid_ratio > 65 else 0.6
            return 'LONG', strength
        
        # SHORT сигнал
        if price >= bb_upper and rsi > 65 and macd_hist > 0 and bid_ratio < 45:
            strength = 0.9 if rsi > 70 and macd_hist > 0.5 and bid_ratio < 35 else 0.6
            return 'SHORT', strength
        
        return None, 0

    def trend_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        vwap = last['vwap']
        ema20 = last['ema_20']
        ema50 = last['ema_50']
        rsi = last['rsi']
        bid_ratio = ob['bid_ratio']

        # LONG сигнал
        if price > vwap and ema20 > ema50 and rsi > 35 and bid_ratio > 55:
            strength = 0.9 if rsi > 45 and bid_ratio > 65 else 0.6
            return 'LONG', strength
        
        # SHORT сигнал  
        if price < vwap and ema20 < ema50 and rsi < 65 and bid_ratio < 45:
            strength = 0.9 if rsi < 55 and bid_ratio < 35 else 0.6
            return 'SHORT', strength
        
        return None, 0

    def detect_signal(self, symbol, df):
        if not self.check_daily_loss_limit():
            return None, None, None

        # Обновляем макроэкономические данные (раз в день)
        self.update_macro_context()
        
        # Обновляем контекст старших ТФ (не чаще чем раз в 5 минут)
        now = datetime.now(timezone.utc)
        if (symbol not in self.mtf_last_update or 
            now - self.mtf_last_update.get(symbol, now) > timedelta(minutes=5)):
            
            self.mtf_context[symbol] = self.mtf_analyzer.get_trend_context(symbol)
            self.mtf_last_update[symbol] = now
            
            # Выводим информацию о глобальном тренде
            print(f"[{now}] 🌍 {symbol} MTF: {self.mtf_context[symbol]['description']} | Согласованность: {self.mtf_context[symbol]['alignment']}")
        
        # Получаем текущий контекст
        context = self.mtf_context.get(symbol, {'trend': 'NEUTRAL', 'strength': 0, 'alignment': 'NEUTRAL'})
        macro_signal = self.get_macro_signal()
        
        last = df.iloc[-1]
        adx = last['adx']
        ob = self.fetch_orderbook_data(symbol)

        # Добавляем bid_ratio в last для лога
        last['bid_ratio'] = ob['bid_ratio']

        side_sig, side_strength = self.sideways_strategy(df, ob)
        trend_sig, trend_strength = self.trend_strategy(df, ob)

        final_signal = None
        final_strength = 0

        if adx < 25:
            if side_sig:
                final_signal = side_sig
                final_strength = side_strength
            elif trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength * 0.6
        elif adx > 30:
            if trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength
            elif side_sig:
                final_signal = side_sig
                final_strength = side_strength * 0.6
        else:
            if side_strength > trend_strength:
                final_signal = side_sig
                final_strength = side_strength
            else:
                final_signal = trend_sig
                final_strength = trend_strength

        # Если есть сигнал, применяем корректировку
        if final_signal:
            original_strength = final_strength
            
            # 1. Корректировка по глобальному тренду
            if context['trend'] == 'BULL' and final_signal == 'LONG':
                boost = min(0.2, context['strength'] * 0.3)
                final_strength = min(1.0, final_strength + boost)
                print(f"📈 Лонг по бычьему тренду: +{boost:.2f} к силе")
                
            elif context['trend'] == 'BEAR' and final_signal == 'SHORT':
                boost = min(0.2, context['strength'] * 0.3)
                final_strength = min(1.0, final_strength + boost)
                print(f"📉 Шорт по медвежьему тренду: +{boost:.2f} к силе")
                
            elif context['trend'] == 'BULL' and final_signal == 'SHORT':
                penalty = min(0.3, context['strength'] * 0.4)
                final_strength = max(0, final_strength - penalty)
                print(f"⚠️ Шорт против бычьего тренда: -{penalty:.2f} к силе")
                
            elif context['trend'] == 'BEAR' and final_signal == 'LONG':
                penalty = min(0.3, context['strength'] * 0.4)
                final_strength = max(0, final_strength - penalty)
                print(f"⚠️ Лонг против медвежьего тренда: -{penalty:.2f} к силе")
            
            # 2. Корректировка по согласованности ТФ
            if context['alignment'] == 'STRONG_BULL' and final_signal == 'LONG':
                final_strength = min(1.0, final_strength + 0.1)
                print(f"💪 Сильная бычья согласованность: +0.1")
            elif context['alignment'] == 'STRONG_BEAR' and final_signal == 'SHORT':
                final_strength = min(1.0, final_strength + 0.1)
                print(f"💪 Сильная медвежья согласованность: +0.1")
            elif context['alignment'] == 'MIXED':
                final_strength = max(0, final_strength - 0.05)
                print(f"🔄 Разнонаправленные ТФ: -0.05")
            
            # 3. Корректировка по макроэкономике
            if macro_signal == 'BULLISH' and final_signal == 'LONG':
                final_strength = min(1.0, final_strength + 0.05)
                print(f"📊 Бычий макрофон: +0.05")
            elif macro_signal == 'BEARISH' and final_signal == 'SHORT':
                final_strength = min(1.0, final_strength + 0.05)
                print(f"📊 Медвежий макрофон: +0.05")
            elif macro_signal == 'BEARISH' and final_signal == 'LONG':
                final_strength = max(0, final_strength - 0.1)
                print(f"⚠️ Лонг при медвежьем макрофоне: -0.1")
            
            if final_strength != original_strength:
                print(f"🔄 Сила сигнала скорректирована: {original_strength:.2f} → {final_strength:.2f}")

        # Порог силы сигнала 0.35 (после корректировки)
        if final_signal and final_strength >= 0.35:
            base = symbol.split('/')[0]
            cg = self.fetch_coinglass_data(base)
            news = self.fetch_cryptopanic_news()

            if not self.get_ai_filter(symbol, df, final_signal, ob, cg, news):
                print(f"[{datetime.now(timezone.utc)}] AI отклонил сигнал {final_signal} для {symbol}")
                return None, None, None

            entry = last['close']
            fee_adj = entry * self.taker_fee
            atr = last['atr']
            if final_signal == 'LONG':
                sl = entry - (self.sl_atr_multiplier * atr) - fee_adj
                tp = entry + (self.tp_atr_multiplier * atr) + fee_adj
            else:
                sl = entry + (self.sl_atr_multiplier * atr) + fee_adj
                tp = entry - (self.tp_atr_multiplier * atr) - fee_adj

            print(f"[{datetime.now(timezone.utc)}] СИГНАЛ! {final_signal} (сила {final_strength:.2f}) для {symbol}")
            return final_signal, "Scalp", {'entry': entry, 'stop_loss': sl, 'take_profit': tp}

        print(f"[{datetime.now(timezone.utc)}] Нет сильного сигнала (сила {final_strength:.2f}) для {symbol}")
        return None, None, None

    def log_trade(self, symbol, side, entry, exit_price, size, pnl, pnl_pct, df_last, hold_time):
        timestamp = datetime.now(timezone.utc).isoformat()
        hold_minutes = hold_time.total_seconds() / 60 if hold_time else 0
        row = [
            timestamp, symbol, side, entry, exit_price, size, pnl, pnl_pct,
            df_last['rsi'], df_last['adx'], df_last['vwap'], df_last['ema_20'], df_last['ema_50'],
            df_last['atr'], df_last['bb_upper'], df_last['bb_lower'],
            df_last['stoch_k'], df_last['stoch_d'], df_last['macd_hist'], df_last.get('bid_ratio', 50),
            round(hold_minutes, 1)
        ]
        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[{timestamp}] Сделка записана в {self.trade_log_file}")

    def get_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                equity = float(bal['info']['result']['list'][0]['totalEquity'])
                print(f"[{datetime.now(timezone.utc)}] Баланс: totalEquity = {equity:.2f} USDT")
                return equity
            elif 'USDT' in bal and 'free' in bal['USDT']:
                usdt_free = float(bal['USDT']['free'])
                print(f"[{datetime.now(timezone.utc)}] USDT free balance: {usdt_free}")
                return usdt_free
            else:
                print(f"[{datetime.now(timezone.utc)}] USDT не найден в ответе баланса")
                return 100.0
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] BALANCE FETCH FAILED: {str(e)}")
            return 100.0

    def place_order(self, symbol, signal, params):
        try:
            balance = self.get_balance()
            
            # Проверка минимального баланса
            if balance < self.min_balance_for_trading:
                print(f"[{datetime.now(timezone.utc)}] Баланс {balance:.2f} ниже минимального {self.min_balance_for_trading}")
                self.send_telegram(f"⚠️ Баланс {balance:.2f} ниже минимального. Торговля приостановлена.")
                return
            
            # Динамический риск в зависимости от баланса
            if balance < 200:
                risk_pct = 0.005  # 0.5% при малом балансе
            else:
                risk_pct = 0.01    # 1% при нормальном балансе
                
            risk = balance * risk_pct
            size = risk / abs(params['entry'] - params['stop_loss'])
            
            # Минимальные размеры для Bybit
            min_sizes = {
                'BTC/USDT:USDT': 0.001,
                'ETH/USDT:USDT': 0.01
            }
            
            if symbol.startswith('BTC'):
                size = round(size, 3)
                if size < min_sizes[symbol]:
                    print(f"Размер {size} меньше минимального {min_sizes[symbol]}, используем минимум")
                    size = min_sizes[symbol]
            else:
                size = round(size, 2)
                if size < min_sizes[symbol]:
                    print(f"Размер {size} меньше минимального {min_sizes[symbol]}, используем минимум")
                    size = min_sizes[symbol]

            if size <= 0:
                print(f"[{datetime.now(timezone.utc)}] Размер позиции слишком мал: {size}")
                return

            msg = (
                f"📉 *Сигнал: {symbol}*\n"
                f"{signal} ({params['entry']:.2f})\n"
                f"SL: {params['stop_loss']:.2f}\n"
                f"TP: {params['take_profit']:.2f}\n"
                f"Размер: {size}\n"
                f"Риск: {risk_pct*100:.1f}%"
            )
            self.send_telegram(msg)

            if signal == 'LONG':
                order = self.exchange.create_market_buy_order(symbol, size)
            else:
                order = self.exchange.create_market_sell_order(symbol, size)

            actual_entry = order.get('average') or params['entry']
            params['entry'] = actual_entry

            self.positions[symbol] = {
                'side': signal,
                'entry': params['entry'],
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'size': size,
                'open_time': datetime.now(timezone.utc),
                'breakeven_activated': False,
                'trailing_activated': False
            }
            print(f"[{datetime.now(timezone.utc)}] Order placed: {signal} {size} for {symbol}")
            self.send_telegram(f"✅ Ордер исполнен: {signal} {size} {symbol} по {actual_entry:.2f}")

        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Order error for {symbol}: {e}")
            self.send_telegram(f"❌ Ошибка ордера {symbol}: {str(e)[:100]}")

    def manage_position(self, symbol, df):
        pos = self.positions.get(symbol)
        if not pos:
            return

        curr = df.iloc[-1]['close']
        side = pos['side']
        entry = pos['entry']
        sl = pos['stop_loss']
        tp = pos['take_profit']
        
        # Проверка времени удержания
        hold_time = datetime.now(timezone.utc) - pos['open_time']
        
        if side == 'LONG':
            pnl_pct = ((curr - entry) / entry) * 100
        else:
            pnl_pct = ((entry - curr) / entry) * 100
        
        # 1. Временной стоп (если позиция висит слишком долго)
        if hold_time > self.max_hold_time:
            print(f"⏰ Время вышло! Позиция {symbol} держится {hold_time}")
            
            if pnl_pct > 0:
                # Если в плюсе - закрываем
                self.close_position(symbol, curr, 'Time Exit (Profit)', df, hold_time)
            elif pnl_pct < -0.1:
                # Если в небольшом минусе - тоже закрываем (лучше, чем SL)
                self.close_position(symbol, curr, 'Time Exit (Stop)', df, hold_time)
            else:
                # Если около нуля - уменьшаем TP и ждем еще немного
                pos['take_profit'] = entry * (1 + (tp/entry - 1) * 0.7)
                print(f"🎯 TP уменьшен из-за времени: {pos['take_profit']:.2f}")
            return
        
        # 2. Динамический трейлинг-стоп
        if pnl_pct > self.min_profit_for_breakeven and not pos.get('breakeven_activated'):
            pos['stop_loss'] = entry  # Безубыток
            pos['breakeven_activated'] = True
            self.send_telegram(f'🔒 {symbol} в безубытке')
        
        if pnl_pct > self.trailing_activation and not pos.get('trailing_activated'):
            pos['trailing_activated'] = True
            self.send_telegram(f'🔝 {symbol} трейлинг активирован')
        
        if pos.get('trailing_activated'):
            if side == 'LONG':
                new_sl = curr * (1 - self.trailing_distance / 100)
                if new_sl > pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    print(f"🔝 Трейлинг стоп поднят до {new_sl:.2f}")
            else:
                new_sl = curr * (1 + self.trailing_distance / 100)
                if new_sl < pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    print(f"🔝 Трейлинг стоп опущен до {new_sl:.2f}")
        
        # 3. Стандартные проверки SL/TP
        if (side == 'LONG' and curr <= sl) or (side == 'SHORT' and curr >= sl):
            self.close_position(symbol, curr, 'SL Hit', df, hold_time)
        elif (side == 'LONG' and curr >= tp) or (side == 'SHORT' and curr <= tp):
            self.close_position(symbol, curr, 'TP Hit', df, hold_time)

        print(f"[{datetime.now(timezone.utc)}] Position checked for {symbol}, PNL %: {pnl_pct:.2f}, hold time: {hold_time}")

    def close_position(self, symbol, price, reason, df, hold_time):
        pos = self.positions.get(symbol)
        if not pos:
            return

        if pos['side'] == 'LONG':
            pnl = (price - pos['entry']) * pos['size']
            pnl_pct = ((price - pos['entry']) / pos['entry']) * 100
        else:
            pnl = (pos['entry'] - price) * pos['size']
            pnl_pct = ((pos['entry'] - price) / pos['entry']) * 100

        # Логируем сделку
        self.log_trade(symbol, pos['side'], pos['entry'], price, pos['size'], pnl, pnl_pct, df.iloc[-1], hold_time)

        try:
            if pos['side'] == 'LONG':
                self.exchange.create_market_sell_order(symbol, pos['size'])
            else:
                self.exchange.create_market_buy_order(symbol, pos['size'])
            
            msg = (
                f"🔴 *Закрыта {symbol}*\n"
                f"Причина: {reason}\n"
                f"Время удержания: {hold_time}\n"
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
            )
            self.send_telegram(msg)
            print(f"[{datetime.now(timezone.utc)}] Position closed for {symbol}: {reason}, P&L: ${pnl:.2f}")
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Close order error for {symbol}: {e}")
            self.send_telegram(f"❌ Ошибка закрытия {symbol}: {str(e)[:100]}")

        self.positions[symbol] = None

    def run(self):
        while True:
            print(f"[{datetime.now(timezone.utc)}] Starting new cycle")
            self.check_daily_loss_limit()
            self.get_balance()
            
            # Обновляем макроэкономику (раз в день)
            self.update_macro_context()
            
            for symbol in self.symbols:
                try:
                    df = self.fetch_ohlcv(symbol)
                    if df is None:
                        print(f"[{datetime.now(timezone.utc)}] Skipping {symbol} - no data")
                        continue
                    df = self.calculate_indicators(df)

                    if self.positions.get(symbol):
                        self.manage_position(symbol, df)
                    else:
                        signal, s_type, params = self.detect_signal(symbol, df)
                        if signal:
                            self.place_order(symbol, signal, params)
                except Exception as e:
                    print(f"[{datetime.now(timezone.utc)}] Error for {symbol}: {e}")
            print(f"[{datetime.now(timezone.utc)}] Cycle finished, sleeping 30s")
            time.sleep(30)


if __name__ == "__main__":
    bot = BybitScalpingBot()
    bot.run()
