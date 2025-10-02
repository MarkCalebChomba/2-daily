import ccxt
import pandas as pd
import numpy as np
import math
import ta
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import csv
import os
import time

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OKXFuturesBot')

class RateLimiter:
    """Rate limiter to prevent API errors"""
    def __init__(self, max_calls_per_second=5):
        self.max_calls = max_calls_per_second
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        async with self.lock:
            now = time.time()
            self.calls = [c for c in self.calls if now - c < 1.0]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = 1.0 - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.calls = []
            
            self.calls.append(time.time())

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(close, period=14):
        return ta.momentum.RSIIndicator(close=close, window=period).rsi()
    
    @staticmethod
    def calculate_macd(close):
        macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()
    
    @staticmethod
    def calculate_bollinger_bands(close, period=20, std_dev=2):
        bb_indicator = ta.volatility.BollingerBands(close=close, window=period, window_dev=std_dev)
        return bb_indicator.bollinger_hband(), bb_indicator.bollinger_mavg(), bb_indicator.bollinger_lband()
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        stoch_indicator = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=k_period, smooth_window=d_period)
        return stoch_indicator.stoch(), stoch_indicator.stoch_signal()
    
    @staticmethod
    def calculate_vwap_with_bands(high, low, close, volume, period=20, std_dev=2):
        try:
            typical_price = (high + low + close) / 3
            tpv = typical_price * volume
            vwap = tpv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            price_diff = typical_price - vwap
            variance = (price_diff ** 2).rolling(window=period).mean()
            std = np.sqrt(variance)
            upper_band = vwap + (std_dev * std)
            lower_band = vwap - (std_dev * std)
            return vwap, upper_band, lower_band, std
        except:
            typical_price = (high + low + close) / 3
            cumulative_tpv = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()
            vwap = cumulative_tpv / cumulative_volume
            rolling_std = typical_price.rolling(window=period).std()
            return vwap, vwap + (std_dev * rolling_std), vwap - (std_dev * rolling_std), rolling_std
    
    @staticmethod
    def calculate_ema(close, period=20):
        return ta.trend.EMAIndicator(close=close, window=period).ema_indicator()
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        return ta.trend.ADXIndicator(high=high, low=low, close=close, window=period).adx()
    
    @staticmethod
    def calculate_keltner_channels(high, low, close, period=20, atr_period=10, multiplier=2):
        """Calculate Keltner Channels"""
        ema = ta.trend.EMAIndicator(close=close, window=period).ema_indicator()
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=atr_period).average_true_range()
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        return upper_channel, ema, lower_channel
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Calculate Average True Range"""
        return ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
    
    @staticmethod
    def calculate_volume_profile_signal(close, volume, period=20):
        """Calculate Volume Profile signal - simplified for real-time use"""
        try:
            # Calculate VWAP as volume-weighted price
            typical_price = close
            vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
            
            # High volume areas act as support/resistance
            volume_ma = volume.rolling(window=period).mean()
            current_volume = volume.iloc[-1]
            
            # Volume above average at current price suggests importance
            volume_strength = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            return vwap, volume_strength
        except:
            return close.rolling(window=period).mean(), 1.0

class SessionTracker:
    def __init__(self, csv_filename="okx_trading_sessions.csv"):
        self.csv_filename = csv_filename
        self.session_start_time = datetime.now()
        self.session_start_balance = 0
        self.trades_data = []
        self.peak_balance = 0
        self.max_drawdown = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.total_trades = 0
        self.create_csv_if_not_exists()
    
    def create_csv_if_not_exists(self):
        if not os.path.exists(self.csv_filename):
            headers = ['session_start', 'session_end', 'session_duration_minutes', 'start_balance', 
                      'end_balance', 'session_pnl', 'session_pnl_percentage', 'trades_total', 
                      'trades_won', 'trades_lost', 'win_rate_percentage', 'max_drawdown_percentage', 
                      'peak_balance', 'final_drawdown_percentage', 'trades_details']
            with open(self.csv_filename, 'w', newline='') as file:
                csv.writer(file).writerow(headers)
    
    def start_session(self, initial_balance):
        self.session_start_time = datetime.now()
        self.session_start_balance = initial_balance
        self.peak_balance = initial_balance
        self.trades_data = []
        self.max_drawdown = 0
        self.trades_won = self.trades_lost = self.total_trades = 0
        logger.info(f"Session started: ${initial_balance:.2f}")
    
    def update_balance(self, current_balance):
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        else:
            drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def add_trade(self, symbol, direction, entry_price, exit_price, pnl, pnl_percentage, status):
        self.trades_data.append({
            'symbol': symbol, 'direction': direction, 'entry_price': entry_price,
            'exit_price': exit_price, 'pnl': pnl, 'pnl_percentage': pnl_percentage,
            'status': status, 'timestamp': datetime.now().isoformat()
        })
        self.total_trades += 1
        if status == 'won':
            self.trades_won += 1
        elif status == 'lost':
            self.trades_lost += 1
    
    def end_session(self, final_balance, reason="manual_close"):
        duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        pnl = final_balance - self.session_start_balance
        pnl_pct = (pnl / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
        win_rate = (self.trades_won / self.total_trades) * 100 if self.total_trades > 0 else 0
        final_dd = ((self.peak_balance - final_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        
        data = [self.session_start_time.isoformat(), datetime.now().isoformat(), round(duration, 2),
                self.session_start_balance, final_balance, pnl, round(pnl_pct, 2), self.total_trades,
                self.trades_won, self.trades_lost, round(win_rate, 2), round(self.max_drawdown, 2),
                self.peak_balance, round(final_dd, 2), json.dumps(self.trades_data)]
        
        with open(self.csv_filename, 'a', newline='') as file:
            csv.writer(file).writerow(data)
        
        logger.info(f"Session ended ({reason}): PnL ${pnl:.2f} ({pnl_pct:.2f}%), Win rate {win_rate:.1f}%")

class OKXFuturesBot:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, test_mode: bool = False):
        self.exchange = ccxt.okx({
            'apiKey': api_key, 'secret': api_secret, 'password': passphrase,
            'sandbox': test_mode, 'enableRateLimit': True,
        })
        self.rate_limiter = RateLimiter(max_calls_per_second=5)
        self.indicators = TechnicalIndicators()
        self.positions = {'long': {}, 'short': {}}  # Track separately
        self.leverage = 20
        self.max_longs = 10
        self.max_shorts = 10
        self.order_reset_time = datetime.now()
        self.session_tracker = SessionTracker()
        self.session_active = False
        self.session_start_balance = 0
        self.profit_target_percentage = 50
        self.logger = logger

    async def rate_limited_call(self, func, *args, **kwargs):
        """Execute API call with rate limiting"""
        await self.rate_limiter.wait_if_needed()
        return func(*args, **kwargs)

    def check_profit_target(self, current_balance):
        if not self.session_active or self.session_start_balance <= 0:
            return False
        profit_pct = ((current_balance - self.session_start_balance) / self.session_start_balance) * 100
        if profit_pct >= self.profit_target_percentage:
            self.logger.info(f"PROFIT TARGET REACHED: {profit_pct:.3f}%")
            return True
        return False

    async def close_all_positions_and_orders(self, reason="profit_target"):
        try:
            self.logger.info(f"Closing all positions - Reason: {reason}")
            
            # Cancel orders
            open_orders = await self.rate_limited_call(self.exchange.fetch_open_orders)
            for order in open_orders:
                if order['status'] == 'open':
                    try:
                        await self.rate_limited_call(self.exchange.cancel_order, order['id'], order['symbol'])
                    except Exception as e:
                        self.logger.error(f"Cancel failed: {e}")
            
            # Close positions
            positions = await self.rate_limited_call(self.exchange.fetch_positions)
            for pos in positions:
                if pos.get('contracts', 0) > 0:
                    symbol = pos['symbol']
                    size = pos['contracts']
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    
                    try:
                        await self.rate_limited_call(
                            self.exchange.create_order,
                            symbol=symbol, type='market', side=side, amount=size,
                            params={'posSide': pos['side'], 'tdMode': 'cross', 'reduceOnly': True}
                        )
                        
                        current_price = (await self.rate_limited_call(self.exchange.fetch_ticker, symbol))['last']
                        entry_price = pos.get('entryPrice', 0)
                        pnl = pos.get('unrealizedPnl', 0)
                        pnl_pct = ((current_price - entry_price) / entry_price * 100) if pos['side'] == 'long' else ((entry_price - current_price) / entry_price * 100)
                        status = 'won' if pnl > 0 else ('lost' if pnl < 0 else 'breakeven')
                        
                        self.session_tracker.add_trade(symbol, pos['side'], entry_price, current_price, pnl, pnl_pct, status)
                        self.logger.info(f"Closed {symbol} {pos['side']}: ${pnl:.2f} ({pnl_pct:.2f}%)")
                    except Exception as e:
                        self.logger.error(f"Close failed {symbol}: {e}")
            
            self.positions = {'long': {}, 'short': {}}
            return True
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
            return False

    def start_new_session(self, initial_balance):
        self.session_start_balance = initial_balance
        self.session_active = True
        self.session_tracker.start_session(initial_balance)
        self.order_reset_time = datetime.now()

    async def get_available_margin(self):
        try:
            balance = await self.rate_limited_call(self.exchange.fetch_balance)
            return balance['USDT']['free'] if 'USDT' in balance else 0
        except Exception as e:
            self.logger.error(f"Error fetching margin: {e}")
            return 0

    def calculate_equal_position_size(self, symbol: str, price: float, total_margin: float, num_positions: int, account_balance: float):
        try:
            market = self.exchange.market(symbol)
            contract_size = market.get('contractSize', 1)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 1)
            
            margin_per_position = total_margin / num_positions
            max_margin_per_trade = account_balance * 0.1
            
            if margin_per_position > max_margin_per_trade:
                margin_per_position = max_margin_per_trade
            
            position_value = margin_per_position * self.leverage
            raw_amount = (position_value / (price * contract_size)) / (self.leverage / 2)
            
            amount_precision = market.get('precision', {}).get('amount', 0)
            amount = math.floor(raw_amount * (10 ** amount_precision)) / (10 ** amount_precision) if amount_precision > 0 else math.floor(raw_amount)
            
            if amount < min_amount:
                return 0, 0
            
            actual_margin = (amount * price * contract_size) / self.leverage
            return amount, actual_margin
        except Exception as e:
            self.logger.error(f"Size calc error {symbol}: {e}")
            return 0, 0

    async def get_futures_symbols(self):
        try:
            markets = self.exchange.load_markets()
            futures_with_volume = []
            
            for symbol, market in markets.items():
                if market.get('swap') and market.get('quote') == 'USDT':
                    try:
                        ticker = await self.rate_limited_call(self.exchange.fetch_ticker, symbol)
                        if ticker.get('baseVolume'):
                            market['volume'] = ticker['baseVolume']
                            market['last'] = ticker['last']
                            futures_with_volume.append(market)
                    except:
                        continue

            futures_with_volume.sort(key=lambda x: x.get('volume', 0), reverse=True)
            return futures_with_volume[:100]
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
            return []

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        try:
            ohlcv = await self.rate_limited_call(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"OHLCV error {symbol}: {e}")
            return pd.DataFrame()

    def analyze_symbol(self, df: pd.DataFrame) -> Dict:
        """Analyze with INVERTED signals using 10 indicators"""
        if df.empty or len(df) < 50:
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None}
        
        long_signals = short_signals = 0
        
        try:
            vwap, upper_band, lower_band, std = self.indicators.calculate_vwap_with_bands(
                df['high'], df['low'], df['close'], df['volume']
            )
            
            if vwap.isna().all():
                return {'score': 0, 'direction': 'neutral', 'vwap_data': None}
            
            current_price = df['close'].iloc[-1]
            vwap_data = {
                'vwap': vwap.iloc[-1], 'upper_band': upper_band.iloc[-1],
                'lower_band': lower_band.iloc[-1], 'std': std.iloc[-1] if not std.isna().all() else None
            }
            
            # 1. RSI - INVERTED: Overbought = long, Oversold = short
            rsi = self.indicators.calculate_rsi(df['close'])
            if rsi.iloc[-1] > 70:
                long_signals += 1
            elif rsi.iloc[-1] < 30:
                short_signals += 1
            
            # 2. MACD - INVERTED: Positive momentum = short, Negative = long
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            if hist.iloc[-1] < 0:
                long_signals += 1
            elif hist.iloc[-1] > 0:
                short_signals += 1
            
            # 3. Bollinger Bands - INVERTED: Above upper = long, Below lower = short
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df['close'])
            if current_price > bb_upper.iloc[-1]:
                long_signals += 1
            elif current_price < bb_lower.iloc[-1]:
                short_signals += 1
            
            # 4. Stochastic - INVERTED: Overbought = long, Oversold = short
            k, d = self.indicators.calculate_stochastic(df['high'], df['low'], df['close'])
            if k.iloc[-1] > 80:
                long_signals += 1
            elif k.iloc[-1] < 20:
                short_signals += 1
            
            # 5. Price vs VWAP - INVERTED: Above VWAP = long, Below = short
            if current_price > vwap_data['vwap']:
                long_signals += 1
            else:
                short_signals += 1
            
            # 6. EMA - INVERTED: Above EMA = long, Below = short
            ema = self.indicators.calculate_ema(df['close'])
            if current_price > ema.iloc[-1]:
                long_signals += 1
            elif current_price < ema.iloc[-1]:
                short_signals += 1
            
            # 7. ADX - INVERTED: Strong uptrend = long, Strong downtrend = short
            adx = self.indicators.calculate_adx(df['high'], df['low'], df['close'])
            if not adx.isna().all() and adx.iloc[-1] > 25:
                if current_price > df['close'].iloc[-10]:
                    long_signals += 1
                elif current_price < df['close'].iloc[-10]:
                    short_signals += 1
            
            # 8. Keltner Channels - INVERTED: Above upper = long, Below lower = short
            kc_upper, kc_middle, kc_lower = self.indicators.calculate_keltner_channels(
                df['high'], df['low'], df['close']
            )
            if not kc_upper.isna().all():
                if current_price > kc_upper.iloc[-1]:
                    long_signals += 1
                elif current_price < kc_lower.iloc[-1]:
                    short_signals += 1
            
            # 9. ATR - INVERTED: High volatility with upward movement = long
            atr = self.indicators.calculate_atr(df['high'], df['low'], df['close'])
            if not atr.isna().all():
                atr_ma = atr.rolling(window=14).mean()
                if atr.iloc[-1] > atr_ma.iloc[-1]:  # High volatility
                    if current_price > df['close'].iloc[-5]:  # Recent upward movement
                        long_signals += 1
                    elif current_price < df['close'].iloc[-5]:  # Recent downward movement
                        short_signals += 1
            
            # 10. Volume Profile - INVERTED: High volume + price up = long
            vp_vwap, volume_strength = self.indicators.calculate_volume_profile_signal(
                df['close'], df['volume']
            )
            if not vp_vwap.isna().all():
                if volume_strength > 1.2:  # Above average volume
                    if current_price > vp_vwap.iloc[-1]:
                        long_signals += 1
                    elif current_price < vp_vwap.iloc[-1]:
                        short_signals += 1
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None}
        
        if long_signals > short_signals:
            return {'score': (long_signals / 10) * 100, 'direction': 'long', 'vwap_data': vwap_data}
        elif short_signals > long_signals:
            return {'score': (short_signals / 10) * 100, 'direction': 'short', 'vwap_data': vwap_data}
        else:
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None}

    async def find_trading_opportunities(self, symbols: List[str]):
        """Find opportunities with balanced long/short search"""
        longs, shorts = [], []
        
        for idx, symbol_info in enumerate(symbols[:230], 1):
            symbol = symbol_info['symbol']
            
            # Skip if already tracked in either direction
            if symbol in self.positions['long'] or symbol in self.positions['short']:
                continue
            
            df_4h = await self.fetch_ohlcv(symbol, '4h', 100)
            analysis_4h = self.analyze_symbol(df_4h)
            
            if analysis_4h['score'] >= 30 and analysis_4h['vwap_data']:
                df_15m = await self.fetch_ohlcv(symbol, '15m', 100)
                analysis_15m = self.analyze_symbol(df_15m)
                
                if analysis_15m['score'] >= 15 and analysis_15m['vwap_data'] and analysis_4h['direction'] == analysis_15m['direction']:
                    ticker = await self.rate_limited_call(self.exchange.fetch_ticker, symbol)
                    
                    trade = {
                        'symbol': symbol, 'direction': analysis_4h['direction'],
                        'score_4h': analysis_4h['score'], 'score_15m': analysis_15m['score'],
                        'combined_score': (analysis_4h['score'] + analysis_15m['score']) / 2,
                        'current_price': ticker['last'], 'vwap_data': analysis_15m['vwap_data']
                    }
                    
                    if analysis_4h['direction'] == 'long':
                        longs.append(trade)
                    else:
                        shorts.append(trade)
        
        longs.sort(key=lambda x: x['combined_score'], reverse=True)
        shorts.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.logger.info(f"Found {len(longs)} longs, {len(shorts)} shorts")
        return longs, shorts

    async def get_current_exposure(self):
        """Get balanced position counts"""
        try:
            positions = await self.rate_limited_call(self.exchange.fetch_positions)
            open_orders = await self.rate_limited_call(self.exchange.fetch_open_orders)
            
            long_count = short_count = 0
            
            for p in positions:
                if p.get('contracts', 0) > 0:
                    if p['side'] == 'long':
                        long_count += 1
                    else:
                        short_count += 1
            
            for o in open_orders:
                if o.get('status') == 'open':
                    if o['side'] == 'buy':
                        long_count += 1
                    else:
                        short_count += 1
            
            self.logger.info(f"Current: {long_count} longs, {short_count} shorts")
            return long_count, short_count
        except Exception as e:
            self.logger.error(f"Exposure check error: {e}")
            return 0, 0

    async def place_bracket_order(self, trade_info: Dict, amount: float, direction: str):
        """Place bracket order with VWAP bands"""
        symbol = trade_info['symbol']
        vwap_data = trade_info['vwap_data']
        
        try:
            await self.rate_limited_call(self.exchange.set_leverage, self.leverage, symbol)
            
            entry_price = vwap_data['vwap']
            
            if direction == 'long':
                side, pos_side = 'buy', 'long'
                sl_price = vwap_data['lower_band'] * 0.998
                risk_distance = entry_price - sl_price
                tp_price = entry_price + (2 * risk_distance)
            else:
                side, pos_side = 'sell', 'short'
                sl_price = vwap_data['upper_band'] * 1.002
                risk_distance = sl_price - entry_price
                tp_price = entry_price - (2 * risk_distance)
            
            params = {
                'posSide': pos_side, 'tdMode': 'cross',
                'takeProfit': {'triggerPrice': tp_price, 'price': tp_price, 'triggerPriceType': 'last'},
                'stopLoss': {'triggerPrice': sl_price, 'price': sl_price, 'triggerPriceType': 'last'}
            }
            
            order = await self.rate_limited_call(
                self.exchange.create_order,
                symbol=symbol, type='limit', side=side, amount=amount, price=entry_price, params=params
            )
            
            self.positions[direction][symbol] = {
                'entry_price': entry_price, 'amount': amount,
                'tp_price': tp_price, 'sl_price': sl_price, 'order_id': order['id']
            }
            
            self.logger.info(f"✓ {direction.upper()} {symbol}: Entry ${entry_price:.6f}, SL ${sl_price:.6f}, TP ${tp_price:.6f}")
            return True
        except Exception as e:
            self.logger.error(f"Order failed {symbol}: {e}")
            return False

    async def run_trading_cycle(self):
        """Main cycle with balanced 10/10 positions"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Trading Cycle - Balanced 10 Longs / 10 Shorts")
            self.logger.info("=" * 60)
            
            current_balance = await self.get_available_margin()
            
            if not self.session_active:
                self.start_new_session(current_balance)
            
            self.session_tracker.update_balance(current_balance)
            
            # Check profit target
            if self.check_profit_target(current_balance):
                await self.close_all_positions_and_orders("profit_target")
                final_balance = await self.get_available_margin()
                self.session_tracker.end_session(final_balance, "profit_target")
                await asyncio.sleep(5)
                new_balance = await self.get_available_margin()
                self.start_new_session(new_balance)
                self.logger.info("NEW SESSION STARTED")
                return
            
            # 1-hour reset
            if datetime.now() - self.order_reset_time > timedelta(hours=1):
                self.logger.info("1-hour reset")
                open_orders = await self.rate_limited_call(self.exchange.fetch_open_orders)
                for order in open_orders:
                    try:
                        await self.rate_limited_call(self.exchange.cancel_order, order['id'], order['symbol'])
                    except:
                        pass
                self.positions = {'long': {}, 'short': {}}
                self.order_reset_time = datetime.now()
                await asyncio.sleep(5)
            
            # Get current positions
            long_count, short_count = await self.get_current_exposure()
            
            long_slots = self.max_longs - long_count
            short_slots = self.max_shorts - short_count
            
            if long_slots == 0 and short_slots == 0:
                self.logger.info("At capacity (10/10)")
                return
            
            # Get opportunities
            symbols = await self.get_futures_symbols()
            if not symbols:
                return
            
            longs, shorts = await self.find_trading_opportunities(symbols)
            
            # Balance selection: take what's available up to slots
            selected_longs = longs[:long_slots] if long_slots > 0 else []
            selected_shorts = shorts[:short_slots] if short_slots > 0 else []
            
            total_selected = len(selected_longs) + len(selected_shorts)
            
            if total_selected == 0:
                self.logger.info("No balanced opportunities found")
                return
            
            self.logger.info(f"Selected: {len(selected_longs)} longs, {len(selected_shorts)} shorts")
            
            # Calculate sizing
            available_margin = await self.get_available_margin()
            balance_info = await self.rate_limited_call(self.exchange.fetch_balance)
            total_account = balance_info.get('USDT', {}).get('total', available_margin)
            
            total_margin = available_margin * 0.8
            
            # Place orders
            for trade in selected_longs:
                amount, margin = self.calculate_equal_position_size(
                    trade['symbol'], trade['vwap_data']['vwap'],
                    total_margin, total_selected, total_account
                )
                if amount > 0:
                    await self.place_bracket_order(trade, amount, 'long')
                    await asyncio.sleep(1)
            
            for trade in selected_shorts:
                amount, margin = self.calculate_equal_position_size(
                    trade['symbol'], trade['vwap_data']['vwap'],
                    total_margin, total_selected, total_account
                )
                if amount > 0:
                    await self.place_bracket_order(trade, amount, 'short')
                    await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Cycle error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    API_KEY = "5e22e924-0001-4675-b235-632c467c818e"
    API_SECRET = "811817EFE46121411C683AEB142B14CE"
    PASSPHRASE = "#Dinywa15"
    TEST_MODE = False
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   OKX Balanced Futures Bot                ║
    ║   • 10 Long / 10 Short positions          ║
    ║   • INVERTED indicator signals            ║
    ║   • 10 indicators with Keltner, ATR, VP   ║
    ║   • Rate limiting (5 calls/sec)           ║
    ║   • Auto-close at 50% profit             ║
    ║   • CSV session logging                   ║
    ╚════════════════════════════════════════════╝
    """)
    
    bot = OKXFuturesBot(API_KEY, API_SECRET, PASSPHRASE, TEST_MODE)
    
    # Test connection
    try:
        margin = await bot.get_available_margin()
        logger.info(f"Connected. Available: ${margin:.2f}")
        logger.info(f"CSV: {bot.session_tracker.csv_filename}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return
    
    while True:
        try:
            await bot.run_trading_cycle()
            logger.info("Waiting 60s...")
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if bot.session_active:
                final_balance = await bot.get_available_margin()
                bot.session_tracker.end_session(final_balance, "manual_shutdown")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
