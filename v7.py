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
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OKXFuturesBot')

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
        """Calculate VWAP with standard deviation bands"""
        try:
            # Calculate VWAP
            typical_price = (high + low + close) / 3
            tpv = typical_price * volume
            
            # Rolling VWAP calculation
            vwap = tpv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            
            # Calculate standard deviation of typical price vs VWAP
            price_diff = typical_price - vwap
            variance = (price_diff ** 2).rolling(window=period).mean()
            std = np.sqrt(variance)
            
            # Calculate bands
            upper_band_2sigma = vwap + (std_dev * std)
            lower_band_2sigma = vwap - (std_dev * std)
            
            return vwap, upper_band_2sigma, lower_band_2sigma, std
            
        except Exception as e:
            # Fallback calculation
            typical_price = (high + low + close) / 3
            cumulative_tpv = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()
            vwap = cumulative_tpv / cumulative_volume
            
            # Simple standard deviation fallback
            rolling_std = typical_price.rolling(window=period).std()
            upper_band_2sigma = vwap + (std_dev * rolling_std)
            lower_band_2sigma = vwap - (std_dev * rolling_std)
            
            return vwap, upper_band_2sigma, lower_band_2sigma, rolling_std
    
    @staticmethod
    def calculate_ema(close, period=20):
        return ta.trend.EMAIndicator(close=close, window=period).ema_indicator()
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        return ta.trend.ADXIndicator(high=high, low=low, close=close, window=period).adx()
    
    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        return ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=period).williams_r()
    
    @staticmethod
    def calculate_cci(high, low, close, period=20):
        return ta.trend.CCIIndicator(high=high, low=low, close=close, window=period).cci()
    
    @staticmethod
    def calculate_mfi(high, low, close, volume, period=14):
        return ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=period).money_flow_index()

class SessionTracker:
    def __init__(self, csv_filename="okx_trading_sessions.csv"):
        self.csv_filename = csv_filename
        self.session_start_time = datetime.now()
        self.session_start_balance = 0
        self.trades_data = []
        self.peak_balance = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.total_trades = 0
        
        # Create CSV file with headers if it doesn't exist
        self.create_csv_if_not_exists()
    
    def create_csv_if_not_exists(self):
        if not os.path.exists(self.csv_filename):
            headers = [
                'session_start', 'session_end', 'session_duration_minutes',
                'start_balance', 'end_balance', 'session_pnl', 'session_pnl_percentage',
                'trades_total', 'trades_won', 'trades_lost', 'win_rate_percentage',
                'max_drawdown_percentage', 'peak_balance', 'final_drawdown_percentage',
                'trades_details'
            ]
            
            with open(self.csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
    
    def start_session(self, initial_balance):
        self.session_start_time = datetime.now()
        self.session_start_balance = initial_balance
        self.peak_balance = initial_balance
        self.trades_data = []
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.total_trades = 0
        logger.info(f"New session started with balance: ${initial_balance:.2f}")
    
    def update_balance(self, current_balance):
        # Update peak and drawdown tracking
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def add_trade(self, symbol, direction, entry_price, exit_price, pnl, pnl_percentage, status):
        trade_data = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'status': status,  # 'won', 'lost', 'breakeven'
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades_data.append(trade_data)
        self.total_trades += 1
        
        if status == 'won':
            self.trades_won += 1
        elif status == 'lost':
            self.trades_lost += 1
    
    def end_session(self, final_balance, reason="manual_close"):
        session_end_time = datetime.now()
        session_duration = (session_end_time - self.session_start_time).total_seconds() / 60  # in minutes
        
        session_pnl = final_balance - self.session_start_balance
        session_pnl_percentage = (session_pnl / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
        
        win_rate = (self.trades_won / self.total_trades) * 100 if self.total_trades > 0 else 0
        final_drawdown = ((self.peak_balance - final_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        
        # Prepare session data
        session_data = [
            self.session_start_time.isoformat(),
            session_end_time.isoformat(),
            round(session_duration, 2),
            self.session_start_balance,
            final_balance,
            session_pnl,
            round(session_pnl_percentage, 2),
            self.total_trades,
            self.trades_won,
            self.trades_lost,
            round(win_rate, 2),
            round(self.max_drawdown, 2),
            self.peak_balance,
            round(final_drawdown, 2),
            json.dumps(self.trades_data)  # Store trades as JSON string
        ]
        
        # Write to CSV
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(session_data)
        
        logger.info(f"Session ended - Reason: {reason}")
        logger.info(f"Duration: {session_duration:.1f} minutes")
        logger.info(f"PnL: ${session_pnl:.2f} ({session_pnl_percentage:.2f}%)")
        logger.info(f"Trades: {self.total_trades} (Won: {self.trades_won}, Lost: {self.trades_lost})")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Max Drawdown: {self.max_drawdown:.2f}%")

class OKXFuturesBot:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, test_mode: bool = False):
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'sandbox': test_mode,
            'enableRateLimit': True,
        })
        
        self.indicators = TechnicalIndicators()
        self.positions = {}
        self.leverage = 20
        self.max_total_positions = 20
        self.order_reset_time = datetime.now()
        self.logger = logger
        
        # NEW FEATURES - Session management and logging
        self.session_tracker = SessionTracker()
        self.session_active = False
        self.session_start_balance = 0
        self.profit_target_percentage = 0.2  # 0.2% profit target
        self.position_history = {}  # Track position outcomes for CSV logging

    # NEW METHOD - Check if profit target reached
    def check_profit_target(self, current_balance):
        if not self.session_active:
            return False
        
        if self.session_start_balance <= 0:
            return False
        
        profit_percentage = ((current_balance - self.session_start_balance) / self.session_start_balance) * 100
        
        if profit_percentage >= self.profit_target_percentage:
            self.logger.info(f"PROFIT TARGET REACHED! {profit_percentage:.3f}% >= {self.profit_target_percentage}%")
            self.logger.info(f"Account grew from ${self.session_start_balance:.2f} to ${current_balance:.2f}")
            return True
        
        return False

    # NEW METHOD - Close all positions and orders
    async def close_all_positions_and_orders(self, reason="profit_target"):
        try:
            self.logger.info(f"Closing all positions and orders - Reason: {reason}")
            
            # Cancel all open orders first
            open_orders = self.exchange.fetch_open_orders()
            cancelled_count = 0
            
            for order in open_orders:
                if order['status'] == 'open':
                    try:
                        self.exchange.cancel_order(order['id'], order['symbol'])
                        cancelled_count += 1
                        self.logger.info(f"Cancelled order: {order['symbol']} {order['side']}")
                    except Exception as e:
                        self.logger.error(f"Failed to cancel order {order['id']}: {e}")
            
            # Close all positions at market price
            positions = self.exchange.fetch_positions()
            closed_count = 0
            
            for pos in positions:
                if pos.get('contracts', 0) > 0:
                    symbol = pos['symbol']
                    size = pos['contracts']
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    
                    try:
                        # Market order to close position
                        close_order = self.exchange.create_order(
                            symbol=symbol,
                            type='market',
                            side=side,
                            amount=size,
                            params={'posSide': pos['side'], 'tdMode': 'cross', 'reduceOnly': True}
                        )
                        
                        closed_count += 1
                        
                        # Log trade outcome for CSV
                        current_price = self.exchange.fetch_ticker(symbol)['last']
                        entry_price = pos.get('entryPrice', 0)
                        pnl = pos.get('unrealizedPnl', 0)
                        
                        if entry_price > 0:
                            if pos['side'] == 'long':
                                pnl_percentage = ((current_price - entry_price) / entry_price) * 100
                            else:
                                pnl_percentage = ((entry_price - current_price) / entry_price) * 100
                        else:
                            pnl_percentage = 0
                        
                        status = 'won' if pnl > 0 else ('lost' if pnl < 0 else 'breakeven')
                        
                        self.session_tracker.add_trade(
                            symbol=symbol,
                            direction=pos['side'],
                            entry_price=entry_price,
                            exit_price=current_price,
                            pnl=pnl,
                            pnl_percentage=pnl_percentage,
                            status=status
                        )
                        
                        self.logger.info(f"Closed position: {symbol} {pos['side']} - PnL: ${pnl:.2f} ({pnl_percentage:.2f}%)")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to close position {symbol}: {e}")
            
            self.logger.info(f"Session cleanup complete - Cancelled {cancelled_count} orders, Closed {closed_count} positions")
            self.positions.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return False

    # NEW METHOD - Start new session
    def start_new_session(self, initial_balance):
        self.session_start_balance = initial_balance
        self.session_active = True
        self.session_tracker.start_session(initial_balance)
        self.order_reset_time = datetime.now()

    def get_available_margin(self):
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free'] if 'USDT' in balance else 0
        except Exception as e:
            self.logger.error(f"Error fetching margin: {e}")
            return 0

    def calculate_equal_position_size(self, symbol: str, price: float, total_margin: float, num_positions: int, account_balance: float):
        """Calculate position size ensuring all positions use equal margin with 5% account safety limit"""
        try:
            market = self.exchange.market(symbol)
            contract_size = market.get('contractSize', 1)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 1)
            
            # Equal margin per position
            margin_per_position = total_margin / num_positions
            
            # SAFETY CHECK: Ensure no single trade exceeds 10% of account
            max_margin_per_trade = account_balance * 0.1  # 10% limit
            
            if margin_per_position > max_margin_per_trade:
                self.logger.warning(f"Trade margin ${margin_per_position:.2f} exceeds 10% limit (${max_margin_per_trade:.2f}) for {symbol}")
                margin_per_position = max_margin_per_trade
                self.logger.info(f"Adjusted to 10% limit: ${margin_per_position:.2f}")
            
            position_value = margin_per_position * self.leverage
            
            # Calculate amount
            raw_amount = (position_value / (price * contract_size)) / (self.leverage / 2)
            
            # Apply precision
            amount_precision = market.get('precision', {}).get('amount', 0)
            if amount_precision > 0:
                amount = math.floor(raw_amount * (10 ** amount_precision)) / (10 ** amount_precision)
            else:
                amount = math.floor(raw_amount)
            
            # Ensure minimum
            if amount < min_amount:
                return 0, 0
            
            # Calculate actual margin used
            actual_margin = (amount * price * contract_size) / self.leverage
            
            # Final safety check - verify the calculated margin doesn't exceed 10%
            if actual_margin > max_margin_per_trade:
                # Recalculate amount to fit within 5% limit
                max_position_value = max_margin_per_trade * self.leverage
                recalc_amount = (max_position_value / (price * contract_size)) / (self.leverage / 2)
                
                # Apply precision again
                if amount_precision > 0:
                    amount = math.floor(recalc_amount * (10 ** amount_precision)) / (10 ** amount_precision)
                else:
                    amount = math.floor(recalc_amount)
                
                # Recalculate actual margin
                actual_margin = (amount * price * contract_size) / self.leverage
                
                self.logger.info(f"Recalculated {symbol}: amount={amount}, margin=${actual_margin:.2f}")
            
            # Verify we're not risking too much
            account_risk_pct = (actual_margin / account_balance) * 100
            
            self.logger.info(f"{symbol}: ${actual_margin:.2f} margin ({account_risk_pct:.2f}% of account)")
            
            return amount, actual_margin
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0, 0

    def get_futures_symbols(self):
        try:
            markets = self.exchange.load_markets()
            futures_with_volume = []
            
            for symbol, market in markets.items():
                if market.get('swap') and market.get('quote') == 'USDT':
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        if ticker.get('baseVolume'):
                            market['volume'] = ticker['baseVolume']
                            market['last'] = ticker['last']
                            futures_with_volume.append(market)
                    except:
                        continue

            futures_with_volume.sort(key=lambda x: x.get('volume', 0), reverse=True)
            return futures_with_volume[:100]

        except Exception as e:
            self.logger.error(f"Error fetching futures: {e}")
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def analyze_symbol(self, df: pd.DataFrame) -> Dict:
        """Analyze symbol with REVERSED signals using 10 indicators"""
        if df.empty or len(df) < 50:
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None}
        
        long_signals = 0  # Will count traditionally bearish signals
        short_signals = 0  # Will count traditionally bullish signals
        
        try:
            # Calculate VWAP with bands
            vwap, upper_band, lower_band, std = self.indicators.calculate_vwap_with_bands(
                df['high'], df['low'], df['close'], df['volume']
            )
            
            if vwap.isna().all():
                return {'score': 0, 'direction': 'neutral', 'vwap_data': None}
            
            current_vwap = vwap.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            vwap_data = {
                'vwap': current_vwap,
                'upper_band': current_upper,
                'lower_band': current_lower,
                'std': std.iloc[-1] if not std.isna().all() else None
            }
            
            # 1. RSI - REVERSED: Overbought becomes long signal
            rsi = self.indicators.calculate_rsi(df['close'])
            if rsi.iloc[-1] > 70:
                long_signals += 1
            elif rsi.iloc[-1] < 30:
                short_signals += 1
            
            # 2. MACD - REVERSED: Negative momentum becomes long signal
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            if hist.iloc[-1] < 0:
                long_signals += 1
            elif hist.iloc[-1] > 0:
                short_signals += 1
            
            # 3. Bollinger Bands - REVERSED: Above upper band becomes long signal
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df['close'])
            if current_price > bb_upper.iloc[-1]:
                long_signals += 1
            elif current_price < bb_lower.iloc[-1]:
                short_signals += 1
            
            # 4. Stochastic - REVERSED: Overbought becomes long signal
            k, d = self.indicators.calculate_stochastic(df['high'], df['low'], df['close'])
            if k.iloc[-1] > 80:
                long_signals += 1
            elif k.iloc[-1] < 20:
                short_signals += 1
            
            # 5. Price vs VWAP - REVERSED: Above VWAP becomes long signal
            if current_price > current_vwap:
                long_signals += 1
            else:
                short_signals += 1
            
            # 6. EMA - REVERSED: Price above EMA becomes long signal
            ema = self.indicators.calculate_ema(df['close'])
            if current_price > ema.iloc[-1]:
                long_signals += 1
            elif current_price < ema.iloc[-1]:
                short_signals += 1
            
            # 7. ADX - REVERSED: Strong trend up becomes long signal
            adx = self.indicators.calculate_adx(df['high'], df['low'], df['close'])
            if not adx.isna().all() and adx.iloc[-1] > 25:
                if current_price > df['close'].iloc[-10]:
                    long_signals += 1
                elif current_price < df['close'].iloc[-10]:
                    short_signals += 1
            
            # 8. Williams %R - REVERSED: Overbought becomes long signal
            williams_r = self.indicators.calculate_williams_r(df['high'], df['low'], df['close'])
            if not williams_r.isna().all():
                if williams_r.iloc[-1] > -20:
                    long_signals += 1
                elif williams_r.iloc[-1] < -80:
                    short_signals += 1
            
            # 9. CCI - REVERSED: Above 100 becomes long signal
            cci = self.indicators.calculate_cci(df['high'], df['low'], df['close'])
            if not cci.isna().all():
                if cci.iloc[-1] > 100:
                    long_signals += 1
                elif cci.iloc[-1] < -100:
                    short_signals += 1
            
            # 10. MFI - REVERSED: Overbought becomes long signal
            mfi = self.indicators.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
            if not mfi.isna().all():
                if mfi.iloc[-1] > 80:
                    long_signals += 1
                elif mfi.iloc[-1] < 20:
                    short_signals += 1
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None}
        
        if long_signals > short_signals:
            direction = 'long'
            score = (long_signals / 10) * 100
        elif short_signals > long_signals:
            direction = 'short' 
            score = (short_signals / 10) * 100
        else:
            direction = 'neutral'
            score = 0
        
        return {
            'score': score,
            'direction': direction,
            'vwap_data': vwap_data,
            'long_signals': long_signals,
            'short_signals': short_signals
        }

    def find_trading_opportunities(self, symbols: List[str]):
        """Find trading opportunities with reversed signals"""
        longs = []
        shorts = []
        
        for idx, symbol_info in enumerate(symbols[:230], 1):
            symbol = symbol_info['symbol']
            
            # Skip symbols already in our tracking
            if symbol in self.positions:
                continue
                
            self.logger.info(f"Analyzing {symbol} ({idx}/230)...")
            
            # Analyze 4h timeframe
            df_4h = self.fetch_ohlcv(symbol, '4h', 100)
            analysis_4h = self.analyze_symbol(df_4h)
            
            if analysis_4h['score'] >= 30 and analysis_4h['vwap_data'] is not None:
                # Confirm with 15m timeframe  
                df_15m = self.fetch_ohlcv(symbol, '15m', 100)
                analysis_15m = self.analyze_symbol(df_15m)
                
                if analysis_15m['score'] >= 15 and analysis_15m['vwap_data'] is not None and analysis_4h['direction'] == analysis_15m['direction']:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    trade_candidate = {
                        'symbol': symbol,
                        'direction': analysis_4h['direction'],
                        'score_4h': analysis_4h['score'],
                        'score_15m': analysis_15m['score'],
                        'combined_score': (analysis_4h['score'] + analysis_15m['score']) / 2,
                        'current_price': current_price,
                        'vwap_data': analysis_15m['vwap_data']
                    }
                    
                    if analysis_4h['direction'] == 'long':
                        longs.append(trade_candidate)
                    else:
                        shorts.append(trade_candidate)
                    
                    self.logger.info(f"{symbol} qualified for {analysis_4h['direction'].upper()} (4h: {analysis_4h['score']:.1f}%, 15m: {analysis_15m['score']:.1f}%)")

        # Sort by combined score
        longs.sort(key=lambda x: x['combined_score'], reverse=True)
        shorts.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.logger.info(f"Found {len(longs)} long candidates, {len(shorts)} short candidates")
        
        return longs, shorts

    def get_current_exposure(self, max_retries=3):
        """Get current positions and pending orders count with retry logic"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching current exposure (attempt {attempt + 1}/{max_retries})...")
                
                # Fetch positions with retry
                positions = None
                for pos_retry in range(2):
                    try:
                        positions = self.exchange.fetch_positions()
                        break
                    except Exception as e:
                        if pos_retry == 0:
                            self.logger.warning(f"Position fetch failed, retrying: {e}")
                            import time
                            time.sleep(2)
                        else:
                            raise e
                
                # Fetch orders with retry
                open_orders = None
                for order_retry in range(2):
                    try:
                        open_orders = self.exchange.fetch_open_orders()
                        break
                    except Exception as e:
                        if order_retry == 0:
                            self.logger.warning(f"Orders fetch failed, retrying: {e}")
                            import time
                            time.sleep(2)
                        else:
                            raise e
                
                # Count active positions
                active_positions = 0
                position_symbols = []
                if positions:
                    for p in positions:
                        if p.get('contracts', 0) > 0:
                            active_positions += 1
                            position_symbols.append(p['symbol'])
                
                # Count pending orders
                pending_orders = 0
                order_symbols = []
                if open_orders:
                    for o in open_orders:
                        if o.get('status') == 'open':
                            pending_orders += 1
                            order_symbols.append(f"{o['symbol']}({o['side']})")
                
                total_exposure = active_positions + pending_orders
                
                # Update internal tracking with actual exchange state
                self.sync_positions_from_exchange(positions, open_orders)
                
                self.logger.info(f"✓ Exchange sync complete:")
                self.logger.info(f"  Active positions ({active_positions}): {position_symbols}")
                self.logger.info(f"  Pending orders ({pending_orders}): {order_symbols}")
                self.logger.info(f"  Total exposure: {total_exposure}/20")
                
                return total_exposure, active_positions, pending_orders
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive backoff: 5s, 10s, 15s
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    import time
                    time.sleep(wait_time)
                else:
                    self.logger.error("All attempts failed - returning zero exposure (DANGEROUS!)")
                    return 0, 0, 0

    def sync_positions_from_exchange(self, positions=None, open_orders=None):
        """Sync internal position tracking with actual exchange state"""
        try:
            # Clear current tracking
            self.positions.clear()
            
            # Sync from positions
            if positions:
                for pos in positions:
                    if pos.get('contracts', 0) > 0:
                        symbol = pos['symbol']
                        self.positions[symbol] = {
                            'direction': 'long' if pos['side'] == 'long' else 'short',
                            'entry_price': pos.get('entryPrice', 0),
                            'amount': pos.get('contracts', 0),
                            'type': 'position',
                            'pnl': pos.get('unrealizedPnl', 0)
                        }
            
            # Sync from open orders
            if open_orders:
                for order in open_orders:
                    if order.get('status') == 'open':
                        symbol = order['symbol']
                        # Don't overwrite if position already exists
                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'direction': 'long' if order['side'] == 'buy' else 'short',
                                'entry_price': order.get('price', 0),
                                'amount': order.get('amount', 0),
                                'type': 'order',
                                'order_id': order.get('id')
                            }
            
            self.logger.info(f"Internal tracking synced: {len(self.positions)} instruments")
            
        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")

    async def get_exchange_status(self):
        """Check if exchange is accessible"""
        try:
            # Simple test call
            balance = self.exchange.fetch_balance()
            return True
        except Exception as e:
            self.logger.error(f"Exchange not accessible: {e}")
            return False

    async def cancel_all_orders(self):
        """Cancel all open orders with retry logic"""
        """Cancel all open orders"""
        try:
            open_orders = self.exchange.fetch_open_orders()
            cancelled_count = 0
            
            for order in open_orders:
                if order['status'] == 'open':
                    try:
                        self.exchange.cancel_order(order['id'], order['symbol'])
                        cancelled_count += 1
                    except:
                        pass
            
            if cancelled_count > 0:
                self.logger.info(f"Cancelled {cancelled_count} orders for reset")
                
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")

    async def place_bracket_order(self, trade_info: Dict, amount: float):
        """Place bracket order using VWAP deviation bands for SL"""
        symbol = trade_info['symbol']
        direction = trade_info['direction']
        vwap_data = trade_info['vwap_data']
        
        try:
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            
            # Entry at VWAP
            entry_price = vwap_data['vwap']
            
            if direction == 'long':
                side = 'buy'
                pos_side = 'long'
                # Stop loss just below -2σ band
                sl_price = vwap_data['lower_band'] * 0.998  # 0.2% buffer below lower band
                # Take profit is 2x the risk
                risk_distance = entry_price - sl_price
                tp_price = entry_price + (2 * risk_distance)
            else:
                side = 'sell'
                pos_side = 'short'
                # Stop loss just above +2σ band
                sl_price = vwap_data['upper_band'] * 1.002  # 0.2% buffer above upper band
                # Take profit is 2x the risk
                risk_distance = sl_price - entry_price
                tp_price = entry_price - (2 * risk_distance)
            
            # OKX bracket order parameters
            params = {
                'posSide': pos_side,
                'tdMode': 'cross',
                'takeProfit': {
                    'triggerPrice': tp_price,
                    'price': tp_price,
                    'triggerPriceType': 'last',
                },
                'stopLoss': {
                    'triggerPrice': sl_price,
                    'price': sl_price,
                    'triggerPriceType': 'last',
                }
            }
            
            # Place bracket order
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=entry_price,
                params=params
            )
            
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'amount': amount,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'order_id': order['id']
            }
            
            risk_pct = (abs(entry_price - sl_price) / entry_price) * 100
            reward_pct = (abs(tp_price - entry_price) / entry_price) * 100
            
            self.logger.info(f"✓ {direction.upper()} bracket order placed for {symbol}")
            self.logger.info(f"  Entry: ${entry_price:.6f} | SL: ${sl_price:.6f} (at -2σ) | TP: ${tp_price:.6f}")
            self.logger.info(f"  Risk: {risk_pct:.2f}% | Reward: {reward_pct:.2f}% | R:R = 1:{reward_pct/risk_pct:.1f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to place bracket order for {symbol}: {e}")
            # Try simple limit order as fallback
            try:
                simple_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=amount,
                    price=entry_price,
                    params={'posSide': pos_side}
                )
                
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'amount': amount,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'order_id': simple_order['id']
                }
                
                self.logger.info(f"✓ Simple {direction.upper()} limit order placed for {symbol}")
                return True
                
            except Exception as e2:
                self.logger.error(f"Failed to place any order for {symbol}: {e2}")
                return False

    async def run_trading_cycle(self):
        """Main trading cycle with 20 position limit and 1-hour order reset"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting trading cycle - VWAP Deviation Bands SL")
            self.logger.info("MAX 20 TOTAL POSITIONS/ORDERS")
            self.logger.info("=" * 60)
            
            # Check exchange connectivity first
            if not await self.get_exchange_status():
                self.logger.error("Exchange not accessible - skipping cycle")
                return
            
            # NEW FEATURE - Check current balance and profit target
            current_balance = self.get_available_margin()
            
            # If no session is active, start one
            if not self.session_active:
                self.start_new_session(current_balance)
            
            # Update session tracker with current balance
            self.session_tracker.update_balance(current_balance)
            
            # Check if profit target is reached
            if self.check_profit_target(current_balance):
                # Close all positions and orders
                success = await self.close_all_positions_and_orders("profit_target")
                
                if success:
                    # End current session and log to CSV
                    final_balance = self.get_available_margin()
                    self.session_tracker.end_session(final_balance, "profit_target")
                    
                    # Start new session
                    await asyncio.sleep(5)  # Wait for balance to update
                    new_balance = self.get_available_margin()
                    self.start_new_session(new_balance)
                    
                    self.logger.info("NEW SESSION STARTED AFTER PROFIT TARGET")
                    return
            
            # Check if 1 hour passed - reset all orders
            if datetime.now() - self.order_reset_time > timedelta(hours=1):
                self.logger.info("1 hour passed - resetting all orders")
                await self.cancel_all_orders()
                self.positions.clear()
                self.order_reset_time = datetime.now()
                await asyncio.sleep(5)  # Wait for cancellations to process
            
            # Check current exposure with retries
            total_exposure, active_positions, pending_orders = self.get_current_exposure()
            
            # Safety check - if we got zero but expected more, wait and retry once
            if total_exposure == 0 and len(self.positions) > 0:
                self.logger.warning("Exposure mismatch detected - waiting and retrying...")
                await asyncio.sleep(10)
                total_exposure, active_positions, pending_orders = self.get_current_exposure()
            
            if total_exposure >= self.max_total_positions:
                self.logger.info("At maximum capacity - waiting for fills or exits")
                return
            
            # Calculate how many more positions we can add
            remaining_slots = self.max_total_positions - total_exposure
            
            # Get available margin and total account balance
            available_margin = self.get_available_margin()
            if available_margin < 10:
                self.logger.error(f"Insufficient margin: ${available_margin:.2f}")
                return
            
            # Get total account balance for safety calculations
            try:
                balance_info = self.exchange.fetch_balance()
                total_account_balance = balance_info.get('USDT', {}).get('total', available_margin)
                if total_account_balance <= 0:
                    total_account_balance = available_margin * 1.25  # Estimate if can't get total
            except:
                total_account_balance = available_margin * 1.25  # Conservative estimate
            
            self.logger.info(f"Available: ${available_margin:.2f}, Total account: ${total_account_balance:.2f}")
            self.logger.info(f"Session balance: ${self.session_start_balance:.2f} -> ${current_balance:.2f}")
            profit_pct = ((current_balance - self.session_start_balance) / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
            self.logger.info(f"Session profit: {profit_pct:.3f}% (Target: {self.profit_target_percentage}%)")
            self.logger.info(f"Remaining slots: {remaining_slots}")
            
            # Verify we have sufficient margin for 5% per trade rule
            min_required_balance = remaining_slots * (total_account_balance * 0.05)
            if available_margin < min_required_balance * 0.5:  # Need at least 50% of theoretical minimum
                self.logger.warning(f"May not have enough margin for {remaining_slots} trades at 5% each")
                self.logger.warning(f"Theoretical minimum needed: ${min_required_balance:.2f}, Available: ${available_margin:.2f}")
            
            # Get symbols and find opportunities
            symbols = self.get_futures_symbols()
            if not symbols:
                return
                
            longs, shorts = self.find_trading_opportunities(symbols)
            
            if not longs and not shorts:
                self.logger.info("No opportunities found")
                return
            
            # Select best opportunities to fill remaining slots
            all_opportunities = []
            
            # Add longs
            for trade in longs[:remaining_slots]:
                all_opportunities.append(trade)
            
            # Add shorts
            remaining_after_longs = remaining_slots - len(all_opportunities)
            for trade in shorts[:remaining_after_longs]:
                all_opportunities.append(trade)
            
            # Sort by combined score and take top remaining slots
            all_opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
            selected_trades = all_opportunities[:remaining_slots]
            
            if not selected_trades:
                return
            
            self.logger.info(f"Selected {len(selected_trades)} trades to fill remaining slots")
            
            # Calculate position sizes
            total_margin_to_use = available_margin * 0.8
            successful_trades = 0
            
            # Place orders
            for i, trade in enumerate(selected_trades, 1):
                self.logger.info(f"Placing order {i}/{len(selected_trades)}: {trade['symbol']} {trade['direction'].upper()}")
                
                amount, margin_used = self.calculate_equal_position_size(
                    trade['symbol'], 
                    trade['vwap_data']['vwap'], 
                    total_margin_to_use, 
                    len(selected_trades),
                    total_account_balance  # Pass account balance for 5% safety check
                )
                
                if amount > 0:
                    success = await self.place_bracket_order(trade, amount)
                    if success:
                        successful_trades += 1
                    await asyncio.sleep(1)  # Rate limiting
            
            self.logger.info(f"Successfully placed {successful_trades}/{len(selected_trades)} orders")
            
            # Final sync after placing orders
            if successful_trades > 0:
                await asyncio.sleep(3)
                final_exposure, final_positions, final_orders = self.get_current_exposure()
                self.logger.info(f"Post-order sync: {final_exposure}/20 total exposure")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()

async def main():
    # Replace with your real credentials
    API_KEY = "516d19c9-a842-412c-8677-322867c8d9a6"
    API_SECRET = "243D09084EDFDC7CE01189F1439DCEF2"
    PASSPHRASE = "#Dinywa15"
    TEST_MODE = False
    
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   OKX Futures Bot - VWAP Deviation Bands     ║
    ║   • Stop Loss at ±2σ VWAP bands              ║
    ║   • Take Profit at 2x risk distance          ║
    ║   • No manual monitoring - pure limit orders ║
    ║   • 20 position/order limit                  ║
    ║   • 1-hour order reset cycle                 ║
    ║   • AUTO CLOSE at 0.2% account profit        ║
    ║   • Full trade logging to CSV                ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    bot = OKXFuturesBot(API_KEY, API_SECRET, PASSPHRASE, TEST_MODE)
    
    # Test connection
    try:
        margin = bot.get_available_margin()
        logger.info(f"Connected successfully. Available: ${margin:.2f}")
        logger.info(f"CSV logging to: {bot.session_tracker.csv_filename}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return
    
    while True:
        try:
            await bot.run_trading_cycle()
            logger.info("Waiting 60 seconds before next cycle...")
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            # End current session before shutdown
            if bot.session_active:
                final_balance = bot.get_available_margin()
                bot.session_tracker.end_session(final_balance, "manual_shutdown")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
