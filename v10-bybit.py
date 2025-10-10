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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BybitFuturesBot')

class EmailNotifier:
    def __init__(self, sender_email="markcalebchomba@gmail.com", 
                 receiver_email="achiverscollege6@gmail.com",
                 password="leug erco myri ncxv"):
        self.sender = sender_email
        self.receiver = receiver_email
        self.password = password
        self.last_email_time = datetime.now() - timedelta(hours=1)
        self.last_report_time = datetime.now()
    
    def send_email(self, subject, body, attachments=None):
        try:
            if (datetime.now() - self.last_email_time).total_seconds() < 600:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = self.receiver
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            if attachments:
                for filepath in attachments:
                    if os.path.exists(filepath):
                        with open(filepath, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(filepath)}")
                        msg.attach(part)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.sender, self.password)
                server.send_message(msg)
            
            self.last_email_time = datetime.now()
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False
    
    def send_session_report(self, csv_file):
        subject = f"Bybit Bot Session Completed - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body = "Trading session completed. See attached CSV for details."
        self.send_email(subject, body, [csv_file] if os.path.exists(csv_file) else None)
    
    def send_periodic_report(self, stats: dict):
        try:
            if (datetime.now() - self.last_report_time).total_seconds() < 43200:
                return False
            
            subject = f"Bybit Bot 12-Hour Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            body = f"""
12-Hour Trading Report
======================

Current Balance: ${stats.get('current_balance', 0):.2f}
Session Start Balance: ${stats.get('session_start_balance', 0):.2f}
Session P&L: ${stats.get('session_pnl', 0):.2f} ({stats.get('session_pnl_pct', 0):.2f}%)

Active Positions: {stats.get('active_positions', 0)}
  - Longs: {stats.get('active_longs', 0)}
  - Shorts: {stats.get('active_shorts', 0)}
  - At Breakeven: {stats.get('breakeven_positions', 0)}

Trades Completed: {stats.get('trades_completed', 0)}
  - Won: {stats.get('trades_won', 0)}
  - Lost: {stats.get('trades_lost', 0)}
  - Win Rate: {stats.get('win_rate', 0):.1f}%

Peak Portfolio P&L: ${stats.get('peak_pnl', 0):.2f}
Max Drawdown: {stats.get('max_drawdown', 0):.2f}%

Bot Status: Running normally
Next report in 12 hours.
"""
            
            result = self.send_email(subject, body)
            if result:
                self.last_report_time = datetime.now()
            return result
            
        except Exception as e:
            logger.error(f"Periodic report failed: {e}")
            return False
    
    def send_error_alert(self, error_msg):
        subject = f"Bybit Bot ERROR - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body = f"Critical error occurred:\n\n{error_msg}\n\nBot will attempt to recover."
        self.send_email(subject, body)

class RateLimiter:
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
            return vwap, vwap + (std_dev * std), vwap - (std_dev * std), std
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
    def calculate_williams_r(high, low, close, period=14):
        return ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=period).williams_r()
    
    @staticmethod
    def calculate_cci(high, low, close, period=20):
        return ta.trend.CCIIndicator(high=high, low=low, close=close, window=period).cci()
    
    @staticmethod
    def calculate_mfi(high, low, close, volume, period=14):
        return ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=period).money_flow_index()

class SessionTracker:
    def __init__(self, csv_filename="bybit_trading_sessions.csv"):
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
                      'peak_balance', 'final_drawdown_percentage', 'avg_indicators_passed_percent', 'trades_details']
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
    
    def add_trade(self, symbol, direction, entry_price, exit_price, pnl, pnl_percentage, status, indicators_passed_pct, exit_reason=""):
        self.trades_data.append({
            'symbol': symbol, 'direction': direction, 'entry_price': entry_price,
            'exit_price': exit_price, 'pnl': pnl, 'pnl_percentage': pnl_percentage,
            'status': status, 'indicators_passed_percent': indicators_passed_pct,
            'exit_reason': exit_reason, 'timestamp': datetime.now().isoformat()
        })
        self.total_trades += 1
        if status == 'won':
            self.trades_won += 1
        elif status == 'lost':
            self.trades_lost += 1
    
    def end_session(self, final_balance, reason="manual_shutdown"):
        duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        pnl = final_balance - self.session_start_balance
        pnl_pct = (pnl / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
        win_rate = (self.trades_won / self.total_trades) * 100 if self.total_trades > 0 else 0
        final_dd = ((self.peak_balance - final_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        
        avg_indicators = 0
        if self.trades_data:
            indicator_pcts = [t.get('indicators_passed_percent', 0) for t in self.trades_data]
            avg_indicators = sum(indicator_pcts) / len(indicator_pcts)
        
        data = [self.session_start_time.isoformat(), datetime.now().isoformat(), round(duration, 2),
                self.session_start_balance, final_balance, pnl, round(pnl_pct, 2), self.total_trades,
                self.trades_won, self.trades_lost, round(win_rate, 2), round(self.max_drawdown, 2),
                self.peak_balance, round(final_dd, 2), round(avg_indicators, 2), json.dumps(self.trades_data)]
        
        with open(self.csv_filename, 'a', newline='') as file:
            csv.writer(file).writerow(data)
        
        logger.info(f"Session ended ({reason}): PnL ${pnl:.2f} ({pnl_pct:.2f}%), Win rate {win_rate:.1f}%, Avg indicators {avg_indicators:.1f}%")
        return self.csv_filename

class BybitFuturesBot:
    def __init__(self, api_key: str, api_secret: str, test_mode: bool = False):
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,  # Added recvWindow
                'position_mode': 'hedge'
            },
            'timeout': 30000,
            'rateLimit': 100,
        })
        
        # Synchronize timestamp
        self.exchange.load_time_difference()
        self.exchange.load_markets()
        
        self.rate_limiter = RateLimiter(max_calls_per_second=5)
        self.indicators = TechnicalIndicators()
        self.positions = {}
        self.leverage = 20
        self.max_longs = 10
        self.max_shorts = 10
        self.max_total_positions = 20
        self.max_trades_per_symbol = 3
        self.order_reset_time = datetime.now()
        self.session_tracker = SessionTracker()
        self.email_notifier = EmailNotifier()
        self.session_active = False
        self.session_start_balance = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.logger = logger
        self.monitoring_active = False
        self.portfolio_peak_pnl = 0
        self.breakeven_positions = set()

    async def rate_limited_call(self, func, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.wait_if_needed()
                result = func(*args, **kwargs)
                self.consecutive_errors = 0
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    self.logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    self.consecutive_errors += 1
                    raise e

    async def close_position(self, symbol: str, position_data: dict, reason: str):
        """Close a single position - Bybit version"""
        try:
            side = 'sell' if position_data['direction'] == 'long' else 'buy'
            position_idx = 1 if position_data['direction'] == 'long' else 2
            
            await self.rate_limited_call(
                self.exchange.create_order,
                symbol=symbol, 
                type='market', 
                side=side, 
                amount=position_data['amount'],
                params={
                    'positionIdx': position_idx,
                    'reduceOnly': True
                }
            )
            
            current_price = (await self.rate_limited_call(self.exchange.fetch_ticker, symbol))['last']
            entry_price = position_data['entry_price']
            
            if position_data['direction'] == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # USDT-based PnL calculation
            pnl = (position_data['amount'] * abs(current_price - entry_price)) * (1 if pnl_pct > 0 else -1)
            status = 'won' if pnl > 0 else ('lost' if pnl < 0 else 'breakeven')
            
            indicators_pct = position_data.get('indicators_passed_percent', 0)
            
            self.session_tracker.add_trade(
                symbol, position_data['direction'], entry_price, current_price, 
                pnl, pnl_pct, status, indicators_pct, reason
            )
            
            self.logger.info(f"✗ Closed {symbol} {position_data['direction'].upper()}: ${pnl:.2f} ({pnl_pct:.2f}%) - {reason}")
            
            if symbol in self.positions:
                del self.positions[symbol]
            
            if symbol in self.breakeven_positions:
                self.breakeven_positions.remove(symbol)
            
            return True
        except Exception as e:
            self.logger.error(f"Close failed {symbol}: {e}")
            return False

    async def close_all_positions_and_orders(self, reason="manual_close"):
        try:
            self.logger.info(f"Closing all positions - Reason: {reason}")
            
            # Cancel open orders
            open_orders = await self.rate_limited_call(self.exchange.fetch_open_orders)
            for order in open_orders:
                if order['status'] == 'open':
                    try:
                        await self.rate_limited_call(self.exchange.cancel_order, order['id'], order['symbol'])
                    except:
                        pass
            
            # Close positions
            positions = await self.rate_limited_call(self.exchange.fetch_positions)
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    symbol = pos['symbol']
                    if symbol in self.positions:
                        await self.close_position(symbol, self.positions[symbol], reason)
                    else:
                        size = float(pos['contracts'])
                        side = 'sell' if pos['side'] == 'long' else 'buy'
                        position_idx = 1 if pos['side'] == 'long' else 2
                        try:
                            await self.rate_limited_call(
                                self.exchange.create_order,
                                symbol=symbol, 
                                type='market', 
                                side=side, 
                                amount=size,
                                params={
                                    'positionIdx': position_idx,
                                    'reduceOnly': True
                                }
                            )
                        except:
                            pass
            
            self.positions.clear()
            self.portfolio_peak_pnl = 0
            self.breakeven_positions.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
            return False

    def start_new_session(self, initial_balance):
        self.session_start_balance = initial_balance
        self.session_active = True
        self.session_tracker.start_session(initial_balance)
        self.order_reset_time = datetime.now()

    def get_available_margin(self):
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free']) if 'USDT' in balance else 0
        except Exception as e:
            self.logger.error(f"Error fetching margin: {e}")
            return 0

    def calculate_equal_position_size(self, symbol: str, price: float, account_balance: float):
        """USDT-based position sizing - 0.1% of account as margin per trade"""
        try:
            market = self.exchange.market(symbol)
            min_amount = float(market.get('limits', {}).get('amount', {}).get('min', 0.01))
            
            # Use 0.1% of account balance as margin per trade
            margin_per_trade = account_balance * 0.001
            
            # Calculate position value with leverage
            position_value_usdt = margin_per_trade * self.leverage
            
            # Calculate amount in base currency
            amount = position_value_usdt / price
            
            # Apply precision
            amount_precision = market.get('precision', {}).get('amount', 3)
            amount = math.floor(amount * (10 ** amount_precision)) / (10 ** amount_precision)
            
            if amount < min_amount:
                return 0, 0
            
            # Actual margin used
            actual_margin = (amount * price) / self.leverage
            
            return amount, actual_margin
        except Exception as e:
            self.logger.error(f"Size calc error {symbol}: {e}")
            return 0, 0

    def get_futures_symbols(self):
        try:
            markets = self.exchange.load_markets()
            futures_with_volume = []
            
            for symbol, market in markets.items():
                if market.get('swap') and market.get('quote') == 'USDT' and market.get('settle') == 'USDT':
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
            self.logger.error(f"Error fetching symbols: {e}")
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
            self.logger.error(f"OHLCV error {symbol}: {e}")
            return pd.DataFrame()

    def analyze_symbol(self, df: pd.DataFrame) -> Dict:
        """INVERTED signals with detailed indicator tracking"""
        if df.empty or len(df) < 50:
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None, 'indicators': {}}
        
        long_signals = short_signals = 0
        indicators_status = {}
        
        try:
            vwap, upper_band, lower_band, std = self.indicators.calculate_vwap_with_bands(
                df['high'], df['low'], df['close'], df['volume']
            )
            
            if vwap.isna().all():
                return {'score': 0, 'direction': 'neutral', 'vwap_data': None, 'indicators': {}}
            
            current_price = df['close'].iloc[-1]
            vwap_data = {
                'vwap': vwap.iloc[-1], 'upper_band': upper_band.iloc[-1],
                'lower_band': lower_band.iloc[-1], 'std': std.iloc[-1] if not std.isna().all() else None
            }
            
            # RSI
            rsi = self.indicators.calculate_rsi(df['close'])
            if rsi.iloc[-1] > 70:
                long_signals += 1
                indicators_status['RSI'] = 'long'
            elif rsi.iloc[-1] < 30:
                short_signals += 1
                indicators_status['RSI'] = 'short'
            else:
                indicators_status['RSI'] = 'neutral'
            
            # MACD
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            if hist.iloc[-1] < 0:
                long_signals += 1
                indicators_status['MACD'] = 'long'
            elif hist.iloc[-1] > 0:
                short_signals += 1
                indicators_status['MACD'] = 'short'
            else:
                indicators_status['MACD'] = 'neutral'
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df['close'])
            if current_price > bb_upper.iloc[-1]:
                long_signals += 1
                indicators_status['BB'] = 'long'
            elif current_price < bb_lower.iloc[-1]:
                short_signals += 1
                indicators_status['BB'] = 'short'
            else:
                indicators_status['BB'] = 'neutral'
            
            # Stochastic
            k, d = self.indicators.calculate_stochastic(df['high'], df['low'], df['close'])
            if k.iloc[-1] > 80:
                long_signals += 1
                indicators_status['STOCH'] = 'long'
            elif k.iloc[-1] < 20:
                short_signals += 1
                indicators_status['STOCH'] = 'short'
            else:
                indicators_status['STOCH'] = 'neutral'
            
            # VWAP
            if current_price > vwap_data['vwap']:
                long_signals += 1
                indicators_status['VWAP'] = 'long'
            else:
                short_signals += 1
                indicators_status['VWAP'] = 'short'
            
            # EMA
            ema = self.indicators.calculate_ema(df['close'])
            if current_price > ema.iloc[-1]:
                long_signals += 1
                indicators_status['EMA'] = 'long'
            elif current_price < ema.iloc[-1]:
                short_signals += 1
                indicators_status['EMA'] = 'short'
            else:
                indicators_status['EMA'] = 'neutral'
            
            # ADX
            adx = self.indicators.calculate_adx(df['high'], df['low'], df['close'])
            if not adx.isna().all() and adx.iloc[-1] > 25:
                if current_price > df['close'].iloc[-10]:
                    long_signals += 1
                    indicators_status['ADX'] = 'long'
                elif current_price < df['close'].iloc[-10]:
                    short_signals += 1
                    indicators_status['ADX'] = 'short'
                else:
                    indicators_status['ADX'] = 'neutral'
            else:
                indicators_status['ADX'] = 'neutral'
            
            # Williams %R
            williams_r = self.indicators.calculate_williams_r(df['high'], df['low'], df['close'])
            if not williams_r.isna().all():
                if williams_r.iloc[-1] > -20:
                    long_signals += 1
                    indicators_status['WILL'] = 'long'
                elif williams_r.iloc[-1] < -80:
                    short_signals += 1
                    indicators_status['WILL'] = 'short'
                else:
                    indicators_status['WILL'] = 'neutral'
            else:
                indicators_status['WILL'] = 'neutral'
            
            # CCI
            cci = self.indicators.calculate_cci(df['high'], df['low'], df['close'])
            if not cci.isna().all():
                if cci.iloc[-1] > 100:
                    long_signals += 1
                    indicators_status['CCI'] = 'long'
                elif cci.iloc[-1] < -100:
                    short_signals += 1
                    indicators_status['CCI'] = 'short'
                else:
                    indicators_status['CCI'] = 'neutral'
            else:
                indicators_status['CCI'] = 'neutral'
            
            # MFI
            mfi = self.indicators.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
            if not mfi.isna().all():
                if mfi.iloc[-1] > 80:
                    long_signals += 1
                    indicators_status['MFI'] = 'long'
                elif mfi.iloc[-1] < 20:
                    short_signals += 1
                    indicators_status['MFI'] = 'short'
                else:
                    indicators_status['MFI'] = 'neutral'
            else:
                indicators_status['MFI'] = 'neutral'
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None, 'indicators': {}}
        
        total_signals = long_signals + short_signals
        indicators_passed_pct = (total_signals / 10) * 100
        
        if long_signals > short_signals:
            return {
                'score': (long_signals / 10) * 100, 
                'direction': 'long', 
                'vwap_data': vwap_data,
                'indicators': indicators_status,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'indicators_passed_percent': indicators_passed_pct
            }
        elif short_signals > long_signals:
            return {
                'score': (short_signals / 10) * 100, 
                'direction': 'short', 
                'vwap_data': vwap_data,
                'indicators': indicators_status,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'indicators_passed_percent': indicators_passed_pct
            }
        else:
            return {'score': 0, 'direction': 'neutral', 'vwap_data': None, 'indicators': indicators_status, 'indicators_passed_percent': 0}

    def find_trading_opportunities(self, symbols: List[str]):
        """Find opportunities with symbol limit enforcement"""
        longs, shorts = [], []
        symbol_trade_counts = {}
        
        for pos_symbol in self.positions.keys():
            base_symbol = pos_symbol.split('/')[0]
            symbol_trade_counts[base_symbol] = symbol_trade_counts.get(base_symbol, 0) + 1
        
        for idx, symbol_info in enumerate(symbols[:230], 1):
            symbol = symbol_info['symbol']
            base_symbol = symbol.split('/')[0]
            
            if symbol_trade_counts.get(base_symbol, 0) >= self.max_trades_per_symbol:
                continue
            
            if symbol in self.positions:
                continue
            
            df_4h = self.fetch_ohlcv(symbol, '4h', 100)
            analysis_4h = self.analyze_symbol(df_4h)
            
            if analysis_4h['score'] >= 30 and analysis_4h['vwap_data']:
                df_15m = self.fetch_ohlcv(symbol, '15m', 100)
                analysis_15m = self.analyze_symbol(df_15m)
                
                if analysis_15m['score'] >= 15 and analysis_15m['vwap_data'] and analysis_4h['direction'] == analysis_15m['direction']:
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    trade = {
                        'symbol': symbol,
                        'direction': analysis_4h['direction'],
                        'score_4h': analysis_4h['score'],
                        'score_15m': analysis_15m['score'],
                        'combined_score': (analysis_4h['score'] + analysis_15m['score']) / 2,
                        'current_price': ticker['last'],
                        'vwap_data': analysis_15m['vwap_data'],
                        'indicators_15m': analysis_15m['indicators'],
                        'indicators_passed_percent': analysis_15m.get('indicators_passed_percent', 0),
                        'entry_indicators': analysis_15m['indicators'].copy()
                    }
                    
                    if analysis_4h['direction'] == 'long':
                        longs.append(trade)
                    else:
                        shorts.append(trade)
                    
                    indicators_str = ", ".join([f"{k}:{v}" for k, v in analysis_15m['indicators'].items()])
                    self.logger.info(f"{symbol} {analysis_4h['direction'].upper()} - 4H:{analysis_4h['score']:.0f}% 15M:{analysis_15m['score']:.0f}% | {indicators_str}")
        
        longs.sort(key=lambda x: x['combined_score'], reverse=True)
        shorts.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.logger.info(f"Found {len(longs)} longs, {len(shorts)} shorts")
        return longs, shorts

    def get_current_exposure(self):
        """Get balanced long/short counts excluding breakeven positions"""
        try:
            positions = self.exchange.fetch_positions()
            open_orders = self.exchange.fetch_open_orders()
            
            long_count = short_count = 0
            position_symbols = {'long': [], 'short': []}
            
            for p in positions:
                if float(p.get('contracts', 0)) > 0:
                    symbol = p['symbol']
                    
                    if symbol in self.breakeven_positions:
                        continue
                    
                    if p['side'] == 'long':
                        long_count += 1
                        position_symbols['long'].append(symbol)
                    else:
                        short_count += 1
                        position_symbols['short'].append(symbol)
            
            for o in open_orders:
                if o.get('status') == 'open':
                    symbol = o['symbol']
                    
                    if symbol in self.breakeven_positions:
                        continue
                    
                    if o['side'] == 'buy' and symbol not in position_symbols['long']:
                        long_count += 1
                    elif o['side'] == 'sell' and symbol not in position_symbols['short']:
                        short_count += 1
            
            self.logger.info(f"Current exposure: {long_count} longs, {short_count} shorts (Total: {long_count + short_count}/20, Breakeven: {len(self.breakeven_positions)})")
            return long_count, short_count
        except Exception as e:
            self.logger.error(f"Exposure check error: {e}")
            return 0, 0

    def sync_positions_from_exchange(self, positions=None, open_orders=None):
        """Sync with exchange state"""
        try:
            if positions:
                for pos in positions:
                    if float(pos.get('contracts', 0)) > 0:
                        symbol = pos['symbol']
                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'direction': 'long' if pos['side'] == 'long' else 'short',
                                'entry_price': float(pos.get('entryPrice', 0)),
                                'amount': float(pos.get('contracts', 0)),
                                'type': 'position',
                                'max_profit_alpha': 0,
                                'trailing_tier': 0,
                                'entry_indicators': {}
                            }
                        else:
                            self.positions[symbol]['entry_price'] = float(pos.get('entryPrice', self.positions[symbol]['entry_price']))
                            self.positions[symbol]['amount'] = float(pos.get('contracts', self.positions[symbol]['amount']))
            
            if open_orders:
                for order in open_orders:
                    if order.get('status') == 'open':
                        symbol = order['symbol']
                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'direction': 'long' if order['side'] == 'buy' else 'short',
                                'entry_price': float(order.get('price', 0)),
                                'amount': float(order.get('amount', 0)),
                                'type': 'order',
                                'order_id': order.get('id'),
                                'max_profit_alpha': 0,
                                'trailing_tier': 0,
                                'entry_indicators': {}
                            }
        except Exception as e:
            self.logger.error(f"Sync error: {e}")

    async def cancel_all_orders(self):
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
                self.logger.info(f"Cancelled {cancelled_count} orders")
        except Exception as e:
            self.logger.error(f"Cancel error: {e}")

    async def update_stop_loss(self, symbol: str, new_sl_price: float, position_data: dict):
        """Update stop loss for existing position - Bybit version"""
        try:
            position_idx = 1 if position_data['direction'] == 'long' else 2
            
            # Bybit uses set_trading_stop for updating SL/TP
            params = {
                'symbol': symbol,
                'positionIdx': position_idx,
                'stopLoss': str(new_sl_price)
            }
            
            self.exchange.private_post_v5_position_trading_stop(params)
            
            self.logger.info(f"↑ Updated SL {symbol}: ${new_sl_price:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"SL update failed {symbol}: {e}")
            return False

    async def check_signal_reversal(self, symbol: str, position_data: dict) -> bool:
        """Check if 50% of entry indicators have flipped"""
        try:
            entry_indicators = position_data.get('entry_indicators', {})
            if not entry_indicators:
                return False
            
            df_15m = self.fetch_ohlcv(symbol, '15m', 100)
            current_analysis = self.analyze_symbol(df_15m)
            
            if not current_analysis or current_analysis['direction'] == 'neutral':
                return False
            
            current_indicators = current_analysis.get('indicators', {})
            
            flipped_count = 0
            total_indicators = 0
            
            for indicator_name, entry_signal in entry_indicators.items():
                if entry_signal in ['long', 'short']:
                    total_indicators += 1
                    current_signal = current_indicators.get(indicator_name, 'neutral')
                    
                    if entry_signal == 'long' and current_signal == 'short':
                        flipped_count += 1
                    elif entry_signal == 'short' and current_signal == 'long':
                        flipped_count += 1
            
            if total_indicators == 0:
                return False
            
            flip_percentage = (flipped_count / total_indicators) * 100
            
            if flip_percentage >= 50:
                self.logger.info(f"⚠ Signal reversal {symbol}: {flip_percentage:.0f}% indicators flipped")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Signal check error {symbol}: {e}")
            return False

    async def monitor_positions(self):
        """Monitor all positions for trailing stops and signal reversals"""
        self.monitoring_active = True
        self.logger.info("Position monitoring started")
        
        while self.monitoring_active:
            try:
                if not self.positions:
                    await asyncio.sleep(10)
                    continue
                
                positions = await self.rate_limited_call(self.exchange.fetch_positions)
                portfolio_total_pnl = 0
                
                for pos in positions:
                    if float(pos.get('contracts', 0)) <= 0:
                        continue
                    
                    symbol = pos['symbol']
                    if symbol not in self.positions:
                        continue
                    
                    position_data = self.positions[symbol]
                    entry_price = position_data['entry_price']
                    current_price = float(pos.get('markPrice', pos.get('lastPrice', entry_price)))
                    
                    if position_data['direction'] == 'long':
                        price_diff = current_price - entry_price
                    else:
                        price_diff = entry_price - current_price
                    
                    sl_price = position_data.get('sl_price', 0)
                    if sl_price > 0:
                        alpha = abs(entry_price - sl_price)
                    else:
                        alpha = abs(entry_price * 0.02)
                    
                    profit_alpha = price_diff / alpha if alpha > 0 else 0
                    
                    max_profit = position_data.get('max_profit_alpha', 0)
                    if profit_alpha > max_profit:
                        position_data['max_profit_alpha'] = profit_alpha
                        max_profit = profit_alpha
                    
                    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                    portfolio_total_pnl += unrealized_pnl
                    
                    should_close = False
                    close_reason = ""
                    
                    signal_flipped = await self.check_signal_reversal(symbol, position_data)
                    if signal_flipped:
                        should_close = True
                        close_reason = "signal_reversal"
                    
                    if max_profit >= 2.0 and profit_alpha < 0:
                        should_close = True
                        close_reason = "trailing_stop_breakeven"
                    
                    elif max_profit >= 3.0 and profit_alpha < (max_profit - 1.0):
                        should_close = True
                        close_reason = "trailing_stop_tier2"
                    
                    elif max_profit >= 3.5 and profit_alpha < 2.0:
                        should_close = True
                        close_reason = "trailing_stop_tier3"
                    
                    elif max_profit >= 2.0:
                        giveback = max_profit - profit_alpha
                        giveback_pct = (giveback / max_profit) * 100 if max_profit > 0 else 0
                        if giveback_pct >= 33:
                            should_close = True
                            close_reason = f"reversal_33%_from_peak"
                    
                    if should_close:
                        await self.close_position(symbol, position_data, close_reason)
                        continue
                    
                    current_tier = position_data.get('trailing_tier', 0)
                    new_sl_price = None
                    
                    if profit_alpha >= 3.5 and current_tier < 3:
                        if position_data['direction'] == 'long':
                            new_sl_price = entry_price + (2.0 * alpha)
                        else:
                            new_sl_price = entry_price - (2.0 * alpha)
                        position_data['trailing_tier'] = 3
                    
                    elif profit_alpha >= 3.0 and current_tier < 2:
                        if position_data['direction'] == 'long':
                            new_sl_price = entry_price + (1.5 * alpha)
                        else:
                            new_sl_price = entry_price - (1.5 * alpha)
                        position_data['trailing_tier'] = 2
                    
                    elif profit_alpha >= 2.0 and current_tier < 1:
                        new_sl_price = entry_price
                        position_data['trailing_tier'] = 1
                        
                        if symbol not in self.breakeven_positions:
                            self.breakeven_positions.add(symbol)
                            self.logger.info(f"🔓 {symbol} at breakeven - slot freed for new trade")
                    
                    if new_sl_price:
                        await self.update_stop_loss(symbol, new_sl_price, position_data)
                
                if portfolio_total_pnl > self.portfolio_peak_pnl:
                    self.portfolio_peak_pnl = portfolio_total_pnl
                
                if self.portfolio_peak_pnl > 0:
                    giveback = self.portfolio_peak_pnl - portfolio_total_pnl
                    giveback_pct = (giveback / self.portfolio_peak_pnl) * 100
                    
                    if self.portfolio_peak_pnl > 100 and giveback_pct >= 40:
                        self.logger.warning(f"Portfolio reversal: ${portfolio_total_pnl:.2f} from peak ${self.portfolio_peak_pnl:.2f}")
                        await self.close_all_positions_and_orders("portfolio_reversal_40%")
                        self.portfolio_peak_pnl = 0
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(10)

    async def place_bracket_order(self, trade_info: Dict, amount: float):
        """Place bracket order with SL/TP - Bybit version"""
        symbol = trade_info['symbol']
        direction = trade_info['direction']
        vwap_data = trade_info['vwap_data']
        
        try:
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol, params={
                'buyLeverage': str(self.leverage),
                'sellLeverage': str(self.leverage)
            })
            
            entry_price = vwap_data['vwap']
            
            if direction == 'long':
                side, position_idx = 'buy', 1
                sl_price = vwap_data['lower_band'] * 0.998
                risk_distance = entry_price - sl_price
                tp_price = entry_price + (2 * risk_distance)
            else:
                side, position_idx = 'sell', 2
                sl_price = vwap_data['upper_band'] * 1.002
                risk_distance = sl_price - entry_price
                tp_price = entry_price - (2 * risk_distance)
            
            params = {
                'positionIdx': position_idx,
                'stopLoss': str(sl_price),
                'takeProfit': str(tp_price)
            }
            
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
                'order_id': order['id'],
                'indicators_passed_percent': trade_info.get('indicators_passed_percent', 0),
                'entry_indicators': trade_info.get('entry_indicators', {}),
                'max_profit_alpha': 0,
                'trailing_tier': 0,
                'type': 'order'
            }
            
            risk_pct = (abs(entry_price - sl_price) / entry_price) * 100
            reward_pct = (abs(tp_price - entry_price) / entry_price) * 100
            
            self.logger.info(f"✓ {direction.upper()} {symbol}: Entry ${entry_price:.6f}, SL ${sl_price:.6f}, TP ${tp_price:.6f}")
            self.logger.info(f"  Risk {risk_pct:.2f}%, Reward {reward_pct:.2f}%, R:R 1:{reward_pct/risk_pct:.1f}, Indicators {trade_info.get('indicators_passed_percent', 0):.0f}%")
            return True
            
        except Exception as e:
            self.logger.error(f"Bracket order failed {symbol}: {e}")
            try:
                simple_order = self.exchange.create_order(
                    symbol=symbol, 
                    type='limit', 
                    side=side, 
                    amount=amount,
                    price=entry_price, 
                    params={'positionIdx': position_idx}
                )
                
                self.positions[symbol] = {
                    'direction': direction, 'entry_price': entry_price, 'amount': amount,
                    'tp_price': tp_price, 'sl_price': sl_price, 'order_id': simple_order['id'],
                    'indicators_passed_percent': trade_info.get('indicators_passed_percent', 0),
                    'entry_indicators': trade_info.get('entry_indicators', {}),
                    'max_profit_alpha': 0, 'trailing_tier': 0, 'type': 'order'
                }
                
                self.logger.info(f"✓ {direction.upper()} {symbol} (simple order)")
                return True
            except Exception as e2:
                self.logger.error(f"All orders failed {symbol}: {e2}")
                return False

    async def run_trading_cycle(self):
        """Main cycle with balanced 10/10 enforcement"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Trading Cycle - Balanced 10 Longs / 10 Shorts")
            self.logger.info("=" * 60)
            
            current_balance = self.get_available_margin()
            if current_balance <= 0:
                self.logger.warning("Balance 0, retrying...")
                await asyncio.sleep(5)
                current_balance = self.get_available_margin()
                if current_balance <= 0:
                    self.logger.error("Unable to fetch balance")
                    return
            
            if not self.session_active:
                self.start_new_session(current_balance)
            
            self.session_tracker.update_balance(current_balance)
            
            # Send 12-hour report
            if (datetime.now() - self.email_notifier.last_report_time).total_seconds() >= 43200:
                positions = self.exchange.fetch_positions()
                active_longs = sum(1 for p in positions if float(p.get('contracts', 0)) > 0 and p['side'] == 'long')
                active_shorts = sum(1 for p in positions if float(p.get('contracts', 0)) > 0 and p['side'] == 'short')
                
                session_pnl = current_balance - self.session_start_balance
                session_pnl_pct = (session_pnl / self.session_start_balance * 100) if self.session_start_balance > 0 else 0
                win_rate = (self.session_tracker.trades_won / self.session_tracker.total_trades * 100) if self.session_tracker.total_trades > 0 else 0
                
                stats = {
                    'current_balance': current_balance,
                    'session_start_balance': self.session_start_balance,
                    'session_pnl': session_pnl,
                    'session_pnl_pct': session_pnl_pct,
                    'active_positions': active_longs + active_shorts,
                    'active_longs': active_longs,
                    'active_shorts': active_shorts,
                    'breakeven_positions': len(self.breakeven_positions),
                    'trades_completed': self.session_tracker.total_trades,
                    'trades_won': self.session_tracker.trades_won,
                    'trades_lost': self.session_tracker.trades_lost,
                    'win_rate': win_rate,
                    'peak_pnl': self.portfolio_peak_pnl,
                    'max_drawdown': self.session_tracker.max_drawdown
                }
                
                self.email_notifier.send_periodic_report(stats)
            
            if datetime.now() - self.order_reset_time > timedelta(hours=1):
                self.logger.info("1-hour reset")
                await self.cancel_all_orders()
                self.order_reset_time = datetime.now()
                await asyncio.sleep(5)
            
            long_count, short_count = self.get_current_exposure()
            
            long_slots = self.max_longs - long_count
            short_slots = self.max_shorts - short_count
            
            if long_slots == 0 and short_slots == 0:
                self.logger.info(f"At capacity (10/10) - {len(self.breakeven_positions)} at breakeven don't count")
                return
            
            symbols = self.get_futures_symbols()
            if not symbols:
                return
            
            longs, shorts = self.find_trading_opportunities(symbols)
            
            selected_longs = longs[:long_slots] if long_slots > 0 else []
            selected_shorts = shorts[:short_slots] if short_slots > 0 else []
            
            total_selected = len(selected_longs) + len(selected_shorts)
            
            if total_selected == 0:
                self.logger.info("No qualified opportunities")
                return
            
            self.logger.info(f"Selected: {len(selected_longs)} longs, {len(selected_shorts)} shorts")
            
            available_margin = self.get_available_margin()
            balance_info = self.exchange.fetch_balance()
            total_account = float(balance_info.get('USDT', {}).get('total', available_margin * 1.25))
            
            total_margin = available_margin * 0.8
            
            placed = 0
            for trade in selected_longs:
                amount, margin = self.calculate_equal_position_size(
                    trade['symbol'], trade['vwap_data']['vwap'],
                    total_margin, total_selected, total_account
                )
                if amount > 0:
                    if await self.place_bracket_order(trade, amount):
                        placed += 1
                    await asyncio.sleep(1)
            
            for trade in selected_shorts:
                amount, margin = self.calculate_equal_position_size(
                    trade['symbol'], trade['vwap_data']['vwap'],
                    total_margin, total_selected, total_account
                )
                if amount > 0:
                    if await self.place_bracket_order(trade, amount):
                        placed += 1
                    await asyncio.sleep(1)
            
            self.logger.info(f"Placed {placed}/{total_selected} orders")
            
            if placed > 0:
                await asyncio.sleep(3)
                positions = self.exchange.fetch_positions()
                open_orders = self.exchange.fetch_open_orders()
                self.sync_positions_from_exchange(positions, open_orders)
            
        except Exception as e:
            self.logger.error(f"Cycle error: {e}")
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.max_consecutive_errors:
                error_msg = f"CRITICAL: {self.consecutive_errors} consecutive errors. Last: {str(e)}"
                self.logger.error(error_msg)
                self.email_notifier.send_error_alert(error_msg)
                
                try:
                    await self.close_all_positions_and_orders("error_recovery")
                    self.consecutive_errors = 0
                except:
                    pass
            
            import traceback
            traceback.print_exc()

async def main():
    API_KEY = "wTECac16kuNJyGJp15"
    API_SECRET = "jcVUfJmJki0j95DKp6JV6P9TM6y6N39fvI8p"
    TEST_MODE = False
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   Bybit Futures Bot - ENHANCED             ║
    ║   • 10 Long / 10 Short (balanced)         ║
    ║   • USDT-based position sizing            ║
    ║   • Breakeven positions free up slots     ║
    ║   • Trailing stops (3-tier system)        ║
    ║   • Signal reversal detection (50%)       ║
    ║   • Max 3 trades per symbol               ║
    ║   • Portfolio drawdown protection (40%)   ║
    ║   • Real-time position monitoring         ║
    ║   • 12-hour email reports                 ║
    ║   • Continuous operation                  ║
    ║   • RecvWindow configured                 ║
    ╚════════════════════════════════════════════╝
    """)
    
    bot = BybitFuturesBot(API_KEY, API_SECRET, TEST_MODE)
    
    try:
        margin = bot.get_available_margin()
        logger.info(f"Connected: ${margin:.2f}")
        logger.info(f"CSV: {bot.session_tracker.csv_filename}")
        logger.info(f"Email: {bot.email_notifier.receiver}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        bot.email_notifier.send_error_alert(f"Startup failed: {str(e)}")
        return
    
    monitor_task = asyncio.create_task(bot.monitor_positions())
    
    try:
        while True:
            try:
                await bot.run_trading_cycle()
                logger.info("Waiting 60s...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Manual shutdown...")
                bot.monitoring_active = False
                await monitor_task
                if bot.session_active:
                    final_balance = bot.get_available_margin()
                    csv_file = bot.session_tracker.end_session(final_balance, "manual_shutdown")
                    bot.email_notifier.send_session_report(csv_file)
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                bot.email_notifier.send_error_alert(f"Loop error: {str(e)}")
                await asyncio.sleep(60)
    finally:
        bot.monitoring_active = False
        try:
            await monitor_task
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
