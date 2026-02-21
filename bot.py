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
            if datetime.now() - time < timedelta(days=1):
                return data
        
        if not self.api_key:
            return None
            
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': 'CPIAUCSL',
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 2
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
        if change_pct > 0.5:
            return 'BEARISH'
        elif change_pct > 0.2:
            return 'CAUTION'
        elif change_pct < -0.2:
            return 'BULLISH'
        else:
            return 'NEUTRAL'
    
    def get_interest_rate(self):
        cache_key = 'interest_rate'
        if cache_key in self.cache:
            data, time = self.cache[cache_key]
            if datetime.now() - time < timedelta(days=1):
                return data
                
        if not self.api_key:
            return None
            
        try:
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


class TradingSessions:
    """Анализ торговых сессий - торгуем всегда, но с разными стратегиями"""
    
    SESSIONS = {
        'asia': {
            'name': '🇯🇵 Азиатская',
            'start': 0,
            'end': 8,
            'volatility': 'medium',
            'description': 'Спокойное накопление',
            'strategy': 'range',
            'color': '🟡',
            'trade_multiplier': 0.8
        },
        'london': {
            'name': '🇬🇧 Лондонская',
            'start': 8,
            'end': 16,
            'volatility': 'high',
            'description': 'Трендовое движение',
            'strategy': 'trend',
            'color': '🔵',
            'trade_multiplier': 1.0
        },
        'ny': {
            'name': '🇺🇸 Нью-Йоркская',
            'start': 13,
            'end': 21,
            'volatility': 'very_high',
            'description': 'Активная торговля',
            'strategy': 'breakout',
            'color': '🔴',
            'trade_multiplier': 1.2
        },
        'london_ny_overlap': {
            'name': '🇬🇧🇺🇸 Перекрытие',
            'start': 13,
            'end': 16,
            'volatility': 'extreme',
            'description': 'Пик активности',
            'strategy': 'momentum',
            'color': '⚡',
            'trade_multiplier': 1.5
        },
        'weekend': {
            'name': '🎯 Выходные',
            'volatility': 'special',
            'description': 'Низкая ликвидность, ложные движения',
            'strategy': 'fade',
            'color': '💫',
            'trade_multiplier': 0.8  # УВЕЛИЧЕНО С 0.6 ДО 0.8
        },
        'sunday_open': {
            'name': '📊 Открытие недели',
            'start': 21,
            'end': 0,
            'volatility': 'high',
            'description': 'Контртренд на открытии',
            'strategy': 'counter_trend',
            'color': '🌟',
            'trade_multiplier': 0.9
        }
    }
    
    @staticmethod
    def get_current_session():
        """Определяет текущую сессию"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()
        
        # Воскресенье 21:00 - 00:00 (особое окно)
        if weekday == 6 and hour >= 21:
            return 'sunday_open', TradingSessions.SESSIONS['sunday_open']
        
        # Выходные (но не воскресное открытие)
        if weekday >= 5:
            return 'weekend', TradingSessions.SESSIONS['weekend']
        
        # Перекрытие Лондон-Нью-Йорк
        if 13 <= hour < 16:
            return 'london_ny_overlap', TradingSessions.SESSIONS['london_ny_overlap']
        # Нью-Йорк
        elif 13 <= hour < 21:
            return 'ny', TradingSessions.SESSIONS['ny']
        # Лондон
        elif 8 <= hour < 16:
            return 'london', TradingSessions.SESSIONS['london']
        # Азия
        else:
            return 'asia', TradingSessions.SESSIONS['asia']
    
    @staticmethod
    def get_session_info():
        session_key, session = TradingSessions.get_current_session()
        
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        
        if session_key not in ['weekend', 'sunday_open'] and 'end' in session:
            end_hour = session['end']
            if end_hour <= hour:
                end_hour += 24
            
            minutes_left = (end_hour - hour) * 60 - minute
            hours_left = minutes_left // 60
            mins_left = minutes_left % 60
            
            time_left = f"{hours_left}ч {mins_left}м" if hours_left > 0 else f"{mins_left}м"
        else:
            time_left = "переменная"
        
        return {
            'key': session_key,
            'name': session['name'],
            'volatility': session['volatility'],
            'description': session['description'],
            'strategy': session['strategy'],
            'color': session['color'],
            'time_left': time_left,
            'trade_multiplier': session['trade_multiplier']
        }


class MultiTimeframeAnalyzer:
    """Анализирует все таймфреймы от 15m до 1d"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.timeframes = {
            '15m': {'weight': 0.15, 'name': '15-минутный', 'cache_ttl': 5},
            '30m': {'weight': 0.20, 'name': '30-минутный', 'cache_ttl': 10},
            '1h': {'weight': 0.25, 'name': 'Часовой', 'cache_ttl': 15},
            '4h': {'weight': 0.25, 'name': '4-часовой', 'cache_ttl': 60},
            '1d': {'weight': 0.15, 'name': 'Дневной', 'cache_ttl': 240},
        }
        self.cache = {}
        
    def get_trend_context(self, symbol):
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
            
            if avg_score > 0.2:
                context['trend'] = 'BULL'
                context['strength'] = avg_score
                context['description'] = f"⬆️ Бычий тренд (сила {avg_score:.2f})"
            elif avg_score < -0.2:
                context['trend'] = 'BEAR'
                context['strength'] = abs(avg_score)
                context['description'] = f"⬇️ Медвежий тренд (сила {abs(avg_score):.2f})"
            
            bull_count = sum(1 for d in directions if d == 'BULL')
            bear_count = sum(1 for d in directions if d == 'BEAR')
            
            if bull_count >= 4:
                context['alignment'] = 'STRONG_BULL'
            elif bear_count >= 4:
                context['alignment'] = 'STRONG_BEAR'
            elif bull_count >= 3:
                context['alignment'] = 'BULL'
            elif bear_count >= 3:
                context['alignment'] = 'BEAR'
            else:
                context['alignment'] = 'MIXED'
        
        return context
    
    def _get_cached_data(self, symbol, timeframe, cache_ttl_minutes):
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
        last = df.iloc[-1]
        prev = df.iloc[-5]
        
        score = 0
        reasons = []
        
        if last['ema_20'] > last['ema_50']:
            score += 0.4
            reasons.append("EMA20 > EMA50")
        else:
            score -= 0.4
            reasons.append("EMA20 < EMA50")
        
        if last['close'] > last['ema_20']:
            score += 0.3
            reasons.append("Цена выше EMA20")
        else:
            score -= 0.3
            reasons.append("Цена ниже EMA20")
        
        if last['rsi'] > 50:
            score += 0.2
            reasons.append(f"RSI {last['rsi']:.1f} > 50")
        else:
            score -= 0.2
            reasons.append(f"RSI {last['rsi']:.1f} < 50")
        
        if last['close'] > prev['close']:
            score += 0.1
            reasons.append("Цена растет")
        else:
            score -= 0.1
            reasons.append("Цена падает")
        
        if score > 0.3:
            trend = 'BULL'
            desc = f"⬆️ Бычий"
        elif score < -0.3:
            trend = 'BEAR'
            desc = f"⬇️ Медвежий"
        else:
            trend = 'NEUTRAL'
            desc = f"↔️ Нейтральный"
        
        return trend, score, desc


class GlobalLevels:
    """Анализ глобальных уровней Open/Close"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.cache = {}
        
    def get_daily_levels(self, symbol):
        cache_key = f"{symbol}_daily"
        now = datetime.now(timezone.utc)
        
        if cache_key in self.cache:
            data, time = self.cache[cache_key]
            if now - time < timedelta(hours=1):
                return data
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=5)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            today = df.iloc[-1]
            yesterday = df.iloc[-2] if len(df) > 1 else None
            
            levels = {
                'today_open': today['open'],
                'today_high': today['high'],
                'today_low': today['low'],
                'today_close': today['close'],
                'yesterday_close': yesterday['close'] if yesterday else None,
                'position': self._get_position(today['close'], today['open'], today['high'], today['low'])
            }
            
            levels['psychological'] = self._get_psychological_levels(today['close'])
            
            self.cache[cache_key] = (levels, now)
            return levels
            
        except Exception as e:
            return None
    
    def _get_position(self, price, open_, high, low):
        range_size = high - low
        if range_size == 0:
            return 'MIDDLE'
        
        position = ((price - low) / range_size) * 100
        
        if position < 25:
            return 'LOW'
        elif position < 50:
            return 'LOWER_MID'
        elif position < 75:
            return 'UPPER_MID'
        else:
            return 'HIGH'
    
    def _get_psychological_levels(self, price):
        levels = []
        
        if price > 1000:
            base = round(price / 1000) * 1000
            for i in [-2, -1, 0, 1, 2]:
                levels.append(base + i * 1000)
        elif price > 100:
            base = round(price / 100) * 100
            for i in [-2, -1, 0, 1, 2]:
                levels.append(base + i * 100)
        elif price > 10:
            base = round(price / 10) * 10
            for i in [-2, -1, 0, 1, 2]:
                levels.append(base + i * 10)
        else:
            base = round(price)
            for i in [-2, -1, 0, 1, 2]:
                levels.append(base + i)
        
        return sorted(levels)
    
    def get_signal_from_levels(self, price, levels, side):
        if not levels:
            return 0
        
        boost = 0
        
        if abs(price - levels['today_open']) / levels['today_open'] < 0.002:
            if side == 'LONG' and price > levels['today_open']:
                boost += 0.1
            elif side == 'SHORT' and price < levels['today_open']:
                boost += 0.1
        
        if price >= levels['today_high'] * 0.998:
            if side == 'SHORT':
                boost += 0.15
        elif price <= levels['today_low'] * 1.002:
            if side == 'LONG':
                boost += 0.15
        
        for psych_level in levels.get('psychological', []):
            if abs(price - psych_level) / psych_level < 0.001:
                if side == 'LONG' and price > psych_level:
                    boost += 0.2
                elif side == 'SHORT' and price < psych_level:
                    boost += 0.2
                break
        
        return min(boost, 0.5)


class BybitScalpingBot:
    def __init__(self):
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.cryptopanic_api_plan = os.getenv('CRYPTOPANIC_API_PLAN', 'developer')
        self.fred_api_key = os.getenv('FRED_API_KEY')

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

        # Расширенный список топовых пар
        self.symbols = [
            'BTC/USDT:USDT', 
            'ETH/USDT:USDT',
            'SOL/USDT:USDT',
            'XRP/USDT:USDT'
        ]
        
        # Индивидуальные настройки для каждой пары - ВСЕ С ПЛЕЧОМ 5X
        self.symbol_config = {
            'BTC/USDT:USDT': {
                'risk_pct': 0.005,
                'min_size': 0.001,
                'leverage': 5,
                'volatility_factor': 1.0,
                'max_positions_per_day': 5
            },
            'ETH/USDT:USDT': {
                'risk_pct': 0.005,
                'min_size': 0.01,
                'leverage': 5,
                'volatility_factor': 1.0,
                'max_positions_per_day': 5
            },
            'SOL/USDT:USDT': {
                'risk_pct': 0.003,
                'min_size': 0.1,
                'leverage': 5,
                'volatility_factor': 0.8,
                'max_positions_per_day': 4
            },
            'XRP/USDT:USDT': {
                'risk_pct': 0.004,
                'min_size': 10,
                'leverage': 5,
                'volatility_factor': 0.9,
                'max_positions_per_day': 4
            }
        }

        # Установка плеча для каждой пары
        for symbol in self.symbols:
            try:
                leverage = self.symbol_config[symbol]['leverage']
                self.exchange.set_margin_mode('cross', symbol)
                self.exchange.set_leverage(leverage, symbol)
                print(f"[{datetime.now(timezone.utc)}] Leverage {leverage}x and cross for {symbol}")
            except Exception as e:
                print(f"Note for {symbol}: {e}")

        self.bot = telebot.TeleBot(self.telegram_token)
        self.timeframe = '5m'
        self.positions = {s: None for s in self.symbols}

        # Общие настройки
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5
        self.taker_fee = 0.0006

        self.max_hold_time = timedelta(hours=2)
        self.min_profit_for_breakeven = 0.3
        self.trailing_activation = 0.5
        self.trailing_distance = 0.3
        self.min_balance_for_trading = 50

        # Лимиты
        self.daily_loss_limit_pct = -4.2
        self.max_concurrent_positions = 3
        self.last_day = None
        self.day_start_equity = None
        self.trading_paused_until = None
        
        # Счетчики сделок по парам
        self.pair_trades_today = {s: 0 for s in self.symbols}
        self.last_reset_date = datetime.now(timezone.utc).date()

        # Лог файл с расширенными полями
        self.trade_log_file = "trade_log.csv"
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'entry', 'exit', 'size', 'pnl', 'pnl_pct',
                    'rsi_entry', 'adx_entry', 'vwap_entry', 'ema_20_entry', 'ema_50_entry',
                    'atr_entry', 'bb_upper_entry', 'bb_lower_entry', 'stoch_k_entry', 
                    'stoch_d_entry', 'macd_hist_entry', 'bid_ratio_entry',
                    'exit_reason', 'hold_duration_seconds', 'hold_duration_minutes',
                    'session', 'global_level_boost', 'ai_approved'
                ])

        # Кэши для оптимизации запросов
        self.ohlcv_cache = {}
        self.orderbook_cache = {}
        self.balance_cache = {'value': None, 'time': None}
        self.coinglass_cache = {}
        
        # Настройки кэширования
        self.ohlcv_cache_ttl = 15
        self.orderbook_cache_ttl = 5
        self.balance_cache_ttl = 60
        self.coinglass_cache_ttl = 300
        
        # Rate limiting для Bybit
        self.last_bybit_request = {}
        self.min_request_interval = 0.5

        # Анализаторы
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.exchange)
        self.mtf_context = {}
        self.mtf_last_update = {}

        self.fred = FREDAnalyzer(self.fred_api_key)
        self.macro_context = {}
        self.macro_last_update = None

        self.global_levels = GlobalLevels(self.exchange)
        self.levels_cache = {}

        self.session_trades = {}
        self.current_session = None
        self.last_session_message = None

        # Статистика AI фильтра
        self.ai_stats = {
            'total': 0,
            'approved': 0,
            'rejected': 0,
            'random_taken': 0
        }

        # Проверка дневного лимита при старте
        self.check_daily_loss_limit()

        print(f"[{datetime.now(timezone.utc)}] Bot initialized for {len(self.symbols)} symbols: {self.symbols}")
        print(f"[{datetime.now(timezone.utc)}] REAL TRADING MODE ACTIVE")
        print(f"[{datetime.now(timezone.utc)}] Max concurrent positions: {self.max_concurrent_positions}")
        print(f"[{datetime.now(timezone.utc)}] AI FILTER: ENABLED with simple prompt and 50% random factor")
        self.send_telegram(f"🤖 *Bot started - REAL TRADING*\nSymbols: {', '.join(self.symbols)}\nTimeframe: {self.timeframe}\nMax positions: {self.max_concurrent_positions}\n🤖 AI filter: ENABLED (50% random factor)")

    def _check_rate_limit(self, endpoint):
        now = time.time()
        if endpoint in self.last_bybit_request:
            elapsed = now - self.last_bybit_request[endpoint]
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                time.sleep(sleep_time)
        self.last_bybit_request[endpoint] = time.time()

    def fetch_ohlcv_cached(self, symbol, limit=1000):
        cache_key = f"{symbol}_{self.timeframe}"
        now = datetime.now(timezone.utc)
        
        if cache_key in self.ohlcv_cache:
            data, timestamp = self.ohlcv_cache[cache_key]
            if now - timestamp < timedelta(seconds=self.ohlcv_cache_ttl):
                return data
        
        self._check_rate_limit('ohlcv')
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            self.ohlcv_cache[cache_key] = (df, now)
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_orderbook_cached(self, symbol):
        now = time.time()
        cache_key = f"{symbol}_ob"
        
        if cache_key in self.orderbook_cache:
            data, timestamp = self.orderbook_cache[cache_key]
            if now - timestamp < self.orderbook_cache_ttl:
                return data
        
        self._check_rate_limit('orderbook')
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=50)
            total_bids = sum(bid[1] for bid in orderbook['bids'])
            total_asks = sum(ask[1] for ask in orderbook['asks'])
            total = total_bids + total_asks
            bid_ratio = (total_bids / total) * 100 if total > 0 else 50
            
            result = {'bid_ratio': bid_ratio, 'total_volume': total}
            self.orderbook_cache[cache_key] = (result, now)
            return result
        except Exception as e:
            return {'bid_ratio': 50, 'total_volume': 0}

    def get_balance_cached(self):
        now = time.time()
        
        if self.balance_cache['value'] and self.balance_cache['time']:
            if now - self.balance_cache['time'] < self.balance_cache_ttl:
                return self.balance_cache['value']
        
        self._check_rate_limit('balance')
        try:
            bal = self.exchange.fetch_balance()
            if 'info' in bal and 'result' in bal['info'] and 'list' in bal['info']['result']:
                equity = float(bal['info']['result']['list'][0]['totalEquity'])
            else:
                equity = float(bal['USDT']['total']) if 'USDT' in bal and 'total' in bal['USDT'] else 100.0
            
            self.balance_cache['value'] = equity
            self.balance_cache['time'] = now
            return equity
        except Exception as e:
            print(f"Error getting balance: {e}")
            return self.balance_cache['value'] or 100.0

    def fetch_coinglass_cached(self, symbol_base):
        if not self.coinglass_api_key:
            return {}
        
        now = time.time()
        cache_key = f"coinglass_{symbol_base}"
        
        if cache_key in self.coinglass_cache:
            data, timestamp = self.coinglass_cache[cache_key]
            if now - timestamp < self.coinglass_cache_ttl:
                return data
        
        try:
            headers = {'cg-api-key': self.coinglass_api_key}
            url = f"https://open-api.coinglass.com/public/v2/long_short?symbol={symbol_base}&time_type=h1"
            res = requests.get(url, headers=headers, timeout=10).json()
            data = res.get('data', [])[0] if res.get('success') else {}
            
            self.coinglass_cache[cache_key] = (data, now)
            return data
        except Exception as e:
            print(f"CoinGlass error for {symbol_base}: {e}")
            return {}

    def send_telegram(self, message):
        try:
            # Экранируем специальные символы для Markdown
            message = message.replace('.', '\\.').replace('-', '\\-').replace('_', '\\_')
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            # Если Markdown не работает, отправляем без форматирования
            try:
                self.bot.send_message(self.telegram_chat_id, message)
            except Exception as e2:
                print(f"Telegram error: {e2}")

    def get_session_info(self):
        return TradingSessions.get_session_info()

    def update_session(self):
        session_info = self.get_session_info()
        
        if self.current_session != session_info['key']:
            self.current_session = session_info['key']
            
            now = datetime.now(timezone.utc)
            if not self.last_session_message or now - self.last_session_message > timedelta(minutes=30):
                msg = (
                    f"{session_info['color']} *Новая сессия*\n"
                    f"{session_info['name']}\n"
                    f"Волатильность: {session_info['volatility']}\n"
                    f"Стратегия: {session_info['strategy']}\n"
                    f"Описание: {session_info['description']}\n"
                    f"Множитель риска: {session_info['trade_multiplier']:.1f}x"
                )
                self.send_telegram(msg)
                self.last_session_message = now
            
            print(f"[{now}] 🕐 Сессия: {session_info['name']} | {session_info['description']} | x{session_info['trade_multiplier']}")
        
        return session_info

    def _is_false_breakout(self, df):
        if len(df) < 20:
            return False
        
        last = df.iloc[-1]
        high_20 = df['high'].iloc[-20:].max()
        low_20 = df['low'].iloc[-20:].min()
        
        if last['high'] > high_20 and last['close'] < high_20:
            return True
        if last['low'] < low_20 and last['close'] > low_20:
            return True
        
        return False

    def _is_counter_trend(self, df, signal):
        if len(df) < 50:
            return False
        
        last = df.iloc[-1]
        ema_50 = last['ema_50']
        
        if signal == 'LONG' and last['close'] < ema_50 * 0.98:
            return True
        if signal == 'SHORT' and last['close'] > ema_50 * 1.02:
            return True
        
        return False

    def calculate_indicators(self, df):
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=14)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'], period=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        adx, _, _ = TechnicalIndicators.adx(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        df['stoch_k'], df['stoch_d'] = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['macd'], _, df['macd_hist'] = TechnicalIndicators.macd(df['close'])
        return df

    def can_take_new_position(self, symbol):
        open_positions = sum(1 for pos in self.positions.values() if pos is not None)
        
        if open_positions >= self.max_concurrent_positions:
            print(f"⚠️ Достигнут лимит одновременных позиций ({self.max_concurrent_positions})")
            return False
        
        today = datetime.now(timezone.utc).date()
        if self.last_reset_date != today:
            self.pair_trades_today = {s: 0 for s in self.symbols}
            self.last_reset_date = today
            print(f"📅 Сброс счетчиков сделок на новый день")
        
        max_per_pair = self.symbol_config[symbol]['max_positions_per_day']
        if self.pair_trades_today[symbol] >= max_per_pair:
            print(f"⚠️ Лимит сделок для {symbol} на сегодня ({max_per_pair})")
            return False
        
        return True

    def check_daily_loss_limit(self):
        now = datetime.now(timezone.utc)
        current_day = now.date()

        if self.last_day != current_day:
            try:
                equity = self.get_balance_cached()
                self.day_start_equity = equity
                self.last_day = current_day
                self.trading_paused_until = None
                print(f"[{now}] Новый день UTC. Депозит на начало: {equity:.2f} USDT")
            except Exception as e:
                print(f"Error checking daily loss limit: {e}")
                return True

        if self.trading_paused_until and now < self.trading_paused_until:
            print(f"Trading paused until {self.trading_paused_until}")
            return False

        if self.day_start_equity is None:
            return True

        try:
            current_equity = self.get_balance_cached()
            pnl_pct = (current_equity - self.day_start_equity) / self.day_start_equity * 100

            if pnl_pct <= self.daily_loss_limit_pct:
                self.trading_paused_until = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                msg = f"🚨 Дневной лимит убытков -{self.daily_loss_limit_pct}% достигнут! Пауза до {self.trading_paused_until.strftime('%Y-%m-%d %H:%M UTC')}"
                print(msg)
                self.send_telegram(msg)
                return False
            return True
        except Exception as e:
            print(f"Error checking PnL: {e}")
            return True

    def fetch_cryptopanic_news(self):
        if not self.cryptopanic_api_key:
            return []

        now = datetime.now(timezone.utc)
        
        if self.cryptopanic_cache and self.cryptopanic_cache_time:
            if now - self.cryptopanic_cache_time < self.cryptopanic_cache_duration:
                return self.cryptopanic_cache

        try:
            url = f"https://cryptopanic.com/api/{self.cryptopanic_api_plan}/v2/posts/?auth_token={self.cryptopanic_api_key}&kind=news"
            res = requests.get(url, timeout=10)
            
            if res.status_code == 429:
                return self.cryptopanic_cache if self.cryptopanic_cache else []
            
            if res.status_code != 200:
                return self.cryptopanic_cache if self.cryptopanic_cache else []
            
            data = res.json()
            self.cryptopanic_cache = data.get('results', [])[:5]
            self.cryptopanic_cache_time = now
            return self.cryptopanic_cache
            
        except Exception as e:
            return self.cryptopanic_cache if self.cryptopanic_cache else []

    def update_macro_context(self):
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
        if not self.macro_context:
            return 'NEUTRAL'
        
        inflation = self.macro_context.get('inflation', {})
        rates = self.macro_context.get('rates', {})
        
        if inflation.get('signal') == 'BEARISH' and rates.get('environment') == 'HIGH_RATE':
            return 'BEARISH'
        elif inflation.get('signal') == 'BULLISH' and rates.get('environment') == 'LOW_RATE':
            return 'BULLISH'
        else:
            return 'NEUTRAL'

    def _calculate_signal_strength(self, df, signal, orderbook):
        """Расчет силы сигнала на основе технических факторов"""
        last = df.iloc[-1]
        strength = 0.5
        
        if signal == 'LONG':
            if 45 < last['rsi'] < 65:
                strength += 0.1
            if last['ema_20'] > last['ema_50']:
                strength += 0.1
            if last['close'] > last['vwap']:
                strength += 0.1
            if orderbook['bid_ratio'] > 55:
                strength += 0.1
            if last['macd_hist'] > 0:
                strength += 0.1
            if last['stoch_k'] > 20 and last['stoch_k'] < 80:
                strength += 0.05
        else:
            if 35 < last['rsi'] < 55:
                strength += 0.1
            if last['ema_20'] < last['ema_50']:
                strength += 0.1
            if last['close'] < last['vwap']:
                strength += 0.1
            if orderbook['bid_ratio'] < 45:
                strength += 0.1
            if last['macd_hist'] < 0:
                strength += 0.1
            if last['stoch_k'] > 20 and last['stoch_k'] < 80:
                strength += 0.05
        
        if last['rsi'] > 75 or last['rsi'] < 25:
            strength -= 0.2
        if last['adx'] < 20:
            strength -= 0.1
        if last['adx'] > 40:
            strength += 0.1
        
        return min(max(strength, 0), 1.0)

    def get_ai_filter(self, symbol, df, signal, orderbook, coinglass, news):
        """Упрощенный AI фильтр с высоким random фактором"""
        if not self.deepseek_api_key:
            return True
            
        try:
            last = df.iloc[-1]
            
            # Очень простой промпт
            prompt = f"""Analyze {signal} signal for {symbol} at ${last['close']:.2f}
RSI: {last['rsi']:.1f}, ADX: {last['adx']:.1f}, MACD: {last['macd_hist']:.2f}
Bid Ratio: {orderbook['bid_ratio']:.1f}%

Should we take this scalp trade? Reply YES or NO."""

            print(f"🤖 Отправка запроса к DeepSeek для {symbol} {signal}...")
            
            res = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.7,  # Повышено для разнообразия
                    'max_tokens': 10
                },
                timeout=15
            ).json()
            
            answer = res['choices'][0]['message']['content'].strip().upper()
            positive = answer == 'YES'
            
            self.ai_stats['total'] += 1
            if positive:
                self.ai_stats['approved'] += 1
                print(f"🤖 ✅ DeepSeek ОДОБРИЛ сигнал для {symbol} {signal}")
            else:
                self.ai_stats['rejected'] += 1
                print(f"🤖 ❌ DeepSeek ОТКЛОНИЛ сигнал для {symbol} {signal}")
            
            # УВЕЛИЧЕНО ДО 50% случайный фактор для отклоненных сигналов
            if not positive and np.random.random() < 0.5:
                self.ai_stats['random_taken'] += 1
                print(f"🎲 Случайный фактор (50%): берем сделку несмотря на NO")
                return True
            
            # Статистика
            if self.ai_stats['total'] % 10 == 0:
                approval_rate = (self.ai_stats['approved'] / self.ai_stats['total']) * 100
                random_rate = (self.ai_stats['random_taken'] / self.ai_stats['total']) * 100
                print(f"📊 AI Stats: {self.ai_stats['approved']}/{self.ai_stats['total']} одобрено ({approval_rate:.1f}%), случайно взято: {random_rate:.1f}%")
                
            return positive
            
        except Exception as e:
            print(f"🤖 AI filter error: {e}, разрешаем сделку")
            return True

    def sideways_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        rsi = last['rsi']
        stoch_k = last['stoch_k']
        bb_lower = last['bb_lower']
        bb_upper = last['bb_upper']
        bid_ratio = ob['bid_ratio']

        if price <= bb_lower and rsi < 40 and stoch_k < 30 and bid_ratio > 52:
            strength = 0.6 if rsi < 35 and stoch_k < 25 else 0.4
            return 'LONG', strength
        
        if price >= bb_upper and rsi > 60 and stoch_k > 70 and bid_ratio < 48:
            strength = 0.6 if rsi > 65 and stoch_k > 75 else 0.4
            return 'SHORT', strength
        
        return None, 0

    def trend_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        vwap = last['vwap']
        ema20 = last['ema_20']
        ema50 = last['ema_50']
        rsi = last['rsi']
        stoch_k = last['stoch_k']
        macd_hist = last['macd_hist']
        bid_ratio = ob['bid_ratio']

        if (price > vwap and ema20 > ema50 and 
            rsi > 45 and rsi < 70 and 
            macd_hist > 0 and
            stoch_k < 80 and
            bid_ratio > 52):
            
            strength = 0.7
            if rsi > 50 and stoch_k > 50 and bid_ratio > 58:
                strength = 0.8
            return 'LONG', strength
        
        if (price < vwap and ema20 < ema50 and 
            rsi < 55 and rsi > 30 and 
            macd_hist < 0 and
            stoch_k > 20 and
            bid_ratio < 48):
            
            strength = 0.7
            if rsi < 50 and stoch_k < 50 and bid_ratio < 42:
                strength = 0.8
            return 'SHORT', strength
        
        return None, 0

    def basic_strategy(self, df, ob):
        last = df.iloc[-1]
        price = last['close']
        ema20 = last['ema_20']
        rsi = last['rsi']
        
        if price > ema20 and rsi > 40 and rsi < 70:
            strength = 0.35
            return 'LONG', strength
        
        if price < ema20 and rsi < 60 and rsi > 30:
            strength = 0.35
            return 'SHORT', strength
        
        return None, 0

    def detect_signal(self, symbol, df):
        if not self.check_daily_loss_limit():
            print(f"[{datetime.now(timezone.utc)}] Daily loss limit active, skipping signals")
            return None, None, None

        if not self.can_take_new_position(symbol):
            return None, None, None

        session_info = self.update_session()
        self.update_macro_context()
        
        now = datetime.now(timezone.utc)
        if (symbol not in self.mtf_last_update or 
            now - self.mtf_last_update.get(symbol, now) > timedelta(minutes=5)):
            
            self.mtf_context[symbol] = self.mtf_analyzer.get_trend_context(symbol)
            self.mtf_last_update[symbol] = now
            
            print(f"[{now}] 🌍 {symbol} MTF: {self.mtf_context[symbol]['description']} | Согласованность: {self.mtf_context[symbol]['alignment']}")
        
        context = self.mtf_context.get(symbol, {'trend': 'NEUTRAL', 'strength': 0, 'alignment': 'NEUTRAL'})
        daily_levels = self.global_levels.get_daily_levels(symbol)
        
        last = df.iloc[-1]
        adx = last['adx']
        ob = self.fetch_orderbook_cached(symbol)
        last['bid_ratio'] = ob['bid_ratio']

        side_sig, side_strength = self.sideways_strategy(df, ob)
        trend_sig, trend_strength = self.trend_strategy(df, ob)
        basic_sig, basic_strength = self.basic_strategy(df, ob)

        final_signal = None
        final_strength = 0
        level_boost = 0

        if adx < 25:
            if side_sig:
                final_signal = side_sig
                final_strength = side_strength
                print(f"[{now}] 📊 Флэт: сигнал от sideways стратегии (сила {side_strength})")
            elif basic_sig:
                final_signal = basic_sig
                final_strength = basic_strength * 0.8
                print(f"[{now}] 📊 Флэт: сигнал от базовой стратегии (сила {basic_strength})")
                
        elif adx > 30:
            if trend_sig:
                final_signal = trend_sig
                final_strength = trend_strength
                print(f"[{now}] 📊 Тренд: сигнал от trend стратегии (сила {trend_strength})")
            elif basic_sig:
                final_signal = basic_sig
                final_strength = basic_strength
                print(f"[{now}] 📊 Тренд: сигнал от базовой стратегии (сила {basic_strength})")
                
        else:
            signals = []
            if side_sig:
                signals.append(('sideways', side_strength, side_sig))
            if trend_sig:
                signals.append(('trend', trend_strength, trend_sig))
            if basic_sig:
                signals.append(('basic', basic_strength, basic_sig))
            
            if signals:
                best_signal = max(signals, key=lambda x: x[1])
                final_signal = best_signal[2]
                final_strength = best_signal[1]
                print(f"[{now}] 📊 Смешанный: выбран {best_signal[0]} сигнал (сила {best_signal[1]})")

        if not final_signal:
            if last['close'] > last['ema_20']:
                final_signal = 'LONG'
                final_strength = 0.25
                print(f"[{now}] 📊 Запасной сигнал LONG (цена выше EMA20)")
            elif last['close'] < last['ema_20']:
                final_signal = 'SHORT'
                final_strength = 0.25
                print(f"[{now}] 📊 Запасной сигнал SHORT (цена ниже EMA20)")

        if final_signal:
            # ДОБАВЛЕНО: приоритет LONG в бычьем тренде
            if context['trend'] == 'BULL' and final_signal == 'LONG':
                final_strength += 0.2
                print(f"[{now}] 📈 Дополнительный буст для LONG в бычьем тренде: +0.2")
            
            if context['trend'] == 'BULL' and final_signal == 'LONG':
                boost = min(0.2, context['strength'] * 0.3)
                final_strength = min(1.0, final_strength + boost)
                print(f"[{now}] 📈 MTF буст: +{boost:.2f}")
            elif context['trend'] == 'BEAR' and final_signal == 'SHORT':
                boost = min(0.2, context['strength'] * 0.3)
                final_strength = min(1.0, final_strength + boost)
                print(f"[{now}] 📉 MTF буст: +{boost:.2f}")
            elif context['trend'] == 'BULL' and final_signal == 'SHORT':
                penalty = min(0.3, context['strength'] * 0.4)
                final_strength = max(0, final_strength - penalty)
                print(f"[{now}] ⚠️ MTF штраф: -{penalty:.2f}")
            elif context['trend'] == 'BEAR' and final_signal == 'LONG':
                penalty = min(0.3, context['strength'] * 0.4)
                final_strength = max(0, final_strength - penalty)
                print(f"[{now}] ⚠️ MTF штраф: -{penalty:.2f}")
            
            level_boost = self.global_levels.get_signal_from_levels(
                last['close'], daily_levels, final_signal
            )
            if level_boost > 0:
                final_strength = min(1.0, final_strength + level_boost)
                print(f"[{now}] 📊 Уровневый буст: +{level_boost:.2f}")
            
            session_mult = session_info['trade_multiplier']
            final_strength *= session_mult
            if session_mult != 1.0:
                print(f"[{now}] 🕐 Сессия {session_info['name']}: x{session_mult}")
            
            if session_info['key'] == 'weekend' and self._is_false_breakout(df):
                final_strength += 0.2
                print("🎯 Обнаружен ложный пробой на выходных! +0.2")
            
            if session_info['key'] == 'sunday_open' and self._is_counter_trend(df, final_signal):
                final_strength += 0.15
                print("📊 Контртренд на открытии недели! +0.15")

            print(f"[{now}] 📊 Итоговая сила сигнала: {final_strength:.2f}")

        # ИЗМЕНЕНО: динамический порог в зависимости от сессии
        threshold = 0.20
        if session_info['key'] == 'weekend':
            threshold = 0.15  # Пониженный порог на выходных
            print(f"[{now}] 📊 Выходные: пониженный порог {threshold}")

        if final_signal and final_strength >= threshold:
            base = symbol.split('/')[0]
            cg = self.fetch_coinglass_cached(base)
            news = self.fetch_cryptopanic_news()

            ai_approved = self.get_ai_filter(symbol, df, final_signal, ob, cg, news)
            
            if not ai_approved:
                print(f"[{now}] 🤖 AI фильтр отклонил сигнал")
                return None, None, None
            
            print(f"[{now}] 🤖 AI фильтр ОДОБРИЛ сигнал")

            entry = last['close']
            fee_adj = entry * self.taker_fee
            atr = last['atr']
            
            volatility_factor = self.symbol_config[symbol]['volatility_factor']
            adjusted_atr = atr * volatility_factor
            
            if final_signal == 'LONG':
                sl = entry - (self.sl_atr_multiplier * adjusted_atr) - fee_adj
                tp = entry + (self.tp_atr_multiplier * adjusted_atr) + fee_adj
            else:
                sl = entry + (self.sl_atr_multiplier * adjusted_atr) + fee_adj
                tp = entry - (self.tp_atr_multiplier * adjusted_atr) - fee_adj

            print(f"[{now}] 🎯 СИГНАЛ! {final_signal} (сила {final_strength:.2f}) для {symbol}")
            print(f"    Сессия: {session_info['name']}, Уровни: +{level_boost:.2f}")
            
            return final_signal, "Scalp", {
                'entry': entry, 
                'stop_loss': sl, 
                'take_profit': tp,
                'level_boost': level_boost,
                'session': session_info['name']
            }
        else:
            if final_signal:
                print(f"[{now}] ⚠️ Сигнал {final_signal} отклонен: сила {final_strength:.2f} < {threshold}")
            else:
                print(f"[{now}] ❌ Нет сигнала: side={side_sig}, trend={trend_sig}, basic={basic_sig}, adx={adx:.1f}")

        return None, None, None

    def log_trade(self, symbol, side, entry, exit_price, size, pnl, pnl_pct, 
                  df_entry, hold_duration, exit_reason, session, level_boost, ai_approved=True):
        timestamp = datetime.now(timezone.utc).isoformat()
        hold_seconds = hold_duration.total_seconds()
        hold_minutes = hold_seconds / 60
        
        row = [
            timestamp, symbol, side, entry, exit_price, size, pnl, pnl_pct,
            df_entry['rsi'], df_entry['adx'], df_entry['vwap'], 
            df_entry['ema_20'], df_entry['ema_50'],
            df_entry['atr'], df_entry['bb_upper'], df_entry['bb_lower'],
            df_entry['stoch_k'], df_entry['stoch_d'], df_entry['macd_hist'], 
            df_entry.get('bid_ratio', 50),
            exit_reason, round(hold_seconds, 1), round(hold_minutes, 1),
            session, level_boost, 'YES' if ai_approved else 'NO'
        ]
        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def place_order(self, symbol, signal, params):
        try:
            balance = self.get_balance_cached()
            
            if balance < self.min_balance_for_trading:
                print(f"Баланс {balance:.2f} ниже минимального ({self.min_balance_for_trading})")
                return

            risk_pct = self.symbol_config[symbol]['risk_pct']
            risk = balance * risk_pct
            size = risk / abs(params['entry'] - params['stop_loss'])
            
            min_size = self.symbol_config[symbol]['min_size']
            
            if symbol.startswith('SOL'):
                size = round(size, 2)
            elif symbol.startswith('XRP'):
                size = round(size, 1)
            elif symbol.startswith('BTC'):
                size = round(size, 3)
            else:
                size = round(size, 2)
            
            if size < min_size:
                size = min_size
                print(f"Размер увеличен до минимального: {size}")

            if size <= 0:
                print("Размер ордера <= 0, отмена")
                return

            self.pair_trades_today[symbol] += 1

            msg = (
                f"📉 *НОВЫЙ СИГНАЛ: {symbol}*\n"
                f"{'🟢 LONG' if signal == 'LONG' else '🔴 SHORT'}\n"
                f"Цена входа: {params['entry']:.4f}\n"
                f"Stop Loss: {params['stop_loss']:.4f}\n"
                f"Take Profit: {params['take_profit']:.4f}\n"
                f"Размер: {size}\n"
                f"Риск: {risk_pct*100}%\n"
                f"Плечо: 5x\n"
                f"Сессия: {params.get('session', 'N/A')}\n"
                f"Сделок сегодня: {self.pair_trades_today[symbol]}/{self.symbol_config[symbol]['max_positions_per_day']}"
            )
            self.send_telegram(msg)

            print(f"Размещение ордера: {signal} {size} {symbol}")
            
            if signal == 'LONG':
                order = self.exchange.create_market_buy_order(symbol, size)
            else:
                order = self.exchange.create_market_sell_order(symbol, size)

            actual_entry = order.get('average') or params['entry']
            print(f"Ордер исполнен по цене: {actual_entry}")

            self.positions[symbol] = {
                'side': signal,
                'entry': actual_entry,
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'size': size,
                'open_time': datetime.now(timezone.utc),
                'breakeven_activated': False,
                'trailing_activated': False,
                'level_boost': params.get('level_boost', 0),
                'session': params.get('session', 'N/A')
            }
            
            print(f"✅ Позиция открыта: {signal} {size} {symbol}")

        except Exception as e:
            print(f"Ошибка ордера: {e}")
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
        
        hold_time = datetime.now(timezone.utc) - pos['open_time']
        
        if side == 'LONG':
            pnl_pct = ((curr - entry) / entry) * 100
        else:
            pnl_pct = ((entry - curr) / entry) * 100
        
        if hold_time > self.max_hold_time:
            if pnl_pct > 0:
                self.close_position(symbol, curr, 'TIME_EXIT_PROFIT', df, hold_time, pos)
            elif pnl_pct < -0.1:
                self.close_position(symbol, curr, 'TIME_EXIT_STOP', df, hold_time, pos)
            else:
                pos['take_profit'] = entry * (1 + (tp/entry - 1) * 0.7)
                print(f"⏱️ Время вышло, TP сужен до {pos['take_profit']:.2f}")
            return
        
        if pnl_pct > self.min_profit_for_breakeven and not pos.get('breakeven_activated'):
            pos['stop_loss'] = entry
            pos['breakeven_activated'] = True
            print(f"🔒 Breakeven активирован для {symbol}")
        
        if pnl_pct > self.trailing_activation and not pos.get('trailing_activated'):
            pos['trailing_activated'] = True
            print(f"🏃 Трейлинг стоп активирован для {symbol}")
        
        if pos.get('trailing_activated'):
            if side == 'LONG':
                new_sl = curr * (1 - self.trailing_distance / 100)
                if new_sl > pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    print(f"📈 Трейлинг SL обновлен до {new_sl:.2f}")
            else:
                new_sl = curr * (1 + self.trailing_distance / 100)
                if new_sl < pos['stop_loss']:
                    pos['stop_loss'] = new_sl
                    print(f"📉 Трейлинг SL обновлен до {new_sl:.2f}")
        
        if (side == 'LONG' and curr <= sl) or (side == 'SHORT' and curr >= sl):
            self.close_position(symbol, curr, 'SL_HIT', df, hold_time, pos)
        elif (side == 'LONG' and curr >= tp) or (side == 'SHORT' and curr <= tp):
            self.close_position(symbol, curr, 'TP_HIT', df, hold_time, pos)

    def close_position(self, symbol, price, reason, df, hold_time, pos):
        if not pos:
            return

        if pos['side'] == 'LONG':
            pnl = (price - pos['entry']) * pos['size']
            pnl_pct = ((price - pos['entry']) / pos['entry']) * 100
        else:
            pnl = (pos['entry'] - price) * pos['size']
            pnl_pct = ((pos['entry'] - price) / pos['entry']) * 100

        df_entry = df.iloc[-1].copy()

        self.log_trade(
            symbol, pos['side'], pos['entry'], price, pos['size'], 
            pnl, pnl_pct, df_entry, hold_time, reason,
            pos.get('session', 'N/A'), pos.get('level_boost', 0),
            ai_approved=True
        )

        try:
            if pos['side'] == 'LONG':
                self.exchange.create_market_sell_order(symbol, pos['size'])
            else:
                self.exchange.create_market_buy_order(symbol, pos['size'])
            
            emoji = '✅' if pnl > 0 else '🔴'
            msg = (
                f"{emoji} *Закрыта {symbol}*\n"
                f"Причина: {reason}\n"
                f"Время удержания: {str(hold_time).split('.')[0]}\n"
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
            )
            self.send_telegram(msg)
            print(f"✅ Позиция закрыта: {symbol} {reason} P&L: {pnl_pct:.2f}%")
            
        except Exception as e:
            print(f"Ошибка закрытия: {e}")
            self.send_telegram(f"❌ Ошибка закрытия {symbol}: {str(e)[:100]}")

        self.positions[symbol] = None

    def run(self):
        cycle_count = 0
        
        print(f"\n{'='*50}")
        print(f"🚀 Бот запущен в РЕАЛЬНОМ режиме")
        print(f"📊 Торгуемые пары ({len(self.symbols)}):")
        for symbol in self.symbols:
            config = self.symbol_config[symbol]
            print(f"   • {symbol}: риск {config['risk_pct']*100}%, плечо {config['leverage']}x, макс {config['max_positions_per_day']}/день")
        print(f"⏱️  Таймфрейм: {self.timeframe}")
        print(f"💰 Мин. баланс: {self.min_balance_for_trading} USDT")
        print(f"📈 Макс. одновременных позиций: {self.max_concurrent_positions}")
        print(f"📉 Дневной лимит: {self.daily_loss_limit_pct}%")
        print(f"📝 Лог сделок: {self.trade_log_file}")
        print(f"🤖 AI FILTER: ENABLED with simple prompt and 50% random factor")
        print(f"🎯 Weekend threshold: 0.15 (было 0.20)")
        print(f"📈 LONG priority in BULL trend: +0.2 boost")
        print(f"{'='*50}\n")
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                cycle_count += 1
                
                full_analysis = (cycle_count % 4 == 0)
                
                print(f"\n[{now}] 🔄 Цикл #{cycle_count} {'(полный анализ)' if full_analysis else '(быстрый)'}")
                
                if not self.check_daily_loss_limit():
                    print(f"Торговля приостановлена до завтра")
                    time.sleep(300)
                    continue
                
                balance = self.get_balance_cached()
                print(f"[{now}] 💰 Баланс: {balance:.2f} USDT")
                
                session_info = self.update_session()
                
                if full_analysis:
                    self.update_macro_context()
                
                open_positions = sum(1 for pos in self.positions.values() if pos)
                print(f"[{now}] 📊 Открыто позиций: {open_positions}/{self.max_concurrent_positions}")
                
                for symbol in self.symbols:
                    try:
                        print(f"\n[{now}] 🔍 {symbol}...")
                        
                        df = self.fetch_ohlcv_cached(symbol)
                        if df is None:
                            print(f"[{now}] ❌ Нет данных")
                            continue
                        
                        df = self.calculate_indicators(df)
                        last = df.iloc[-1]
                        print(f"[{now}] 📊 Цена: {last['close']:.4f}, RSI: {last['rsi']:.1f}, ADX: {last['adx']:.1f}")

                        if self.positions.get(symbol):
                            print(f"[{now}] 📈 Управление позицией")
                            self.manage_position(symbol, df)
                        elif full_analysis:
                            print(f"[{now}] 🔎 Поиск сигнала")
                            signal, s_type, params = self.detect_signal(symbol, df)
                            if signal:
                                print(f"[{now}] ✅ НАЙДЕН СИГНАЛ: {signal}")
                                self.place_order(symbol, signal, params)
                            else:
                                print(f"[{now}] ⏸️ Сигналов нет")
                                
                    except Exception as e:
                        print(f"❌ Ошибка для {symbol}: {e}")
                        continue
                
                sleep_time = 30 if full_analysis else 15
                print(f"\n[{datetime.now(timezone.utc)}] ✅ Цикл завершен, ожидание {sleep_time}с")
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Критическая ошибка: {e}")
                print("Ожидание 60с перед перезапуском...")
                time.sleep(60)


if __name__ == "__main__":
    bot = BybitScalpingBot()
    bot.run()
