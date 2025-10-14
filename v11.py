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
logger = logging.getLogger('OKXFuturesBot')

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
    
    def send_periodic_report(self, stats: dict, csv_file: str):
        """Send 12-hour report with CSV attachment"""
        try:
            if (datetime.now() - self.last_report_time).total_seconds() < 43200:
                return False
            
            subject = f"OKX Bot 12-Hour Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            exit_reasons = stats.get('exit_reasons', {})
            exit_summary = "\n".join([f"  - {reason}: {count}" for reason, count in exit_reasons.items()])
            
            body = f"""
12-Hour Trading Report
======================

Performance Summary:
--------------------
Current Balance: ${stats.get('current_balance', 0):.2f}
Session Start Balance: ${stats.get('session_start_balance', 0):.2f}
Session P&L: ${stats.get('session_pnl', 0):.2f} ({stats.get('session_pnl_pct', 0):.2f}%)
Peak Portfolio P&L: ${stats.get('peak_pnl', 0):.2f}
Max Drawdown: {stats.get('max_drawdown', 0):.2f}%

Active Positions:
-----------------
Total Active: {stats.get('active_positions', 0)}
  - Longs: {stats.get('active_longs', 0)}
  - Shorts: {stats.get('active_shorts', 0)}

Trades (Last 12 Hours):
-----------------------
Completed: {stats.get('trades_last_12h', 0)}
  - Won: {stats.get('won_last_12h', 0)}
  - Lost: {stats.get('lost_last_12h', 0)}
  - Win Rate: {stats.get('win_rate_12h', 0):.1f}%

Exit Reasons (Last 12 Hours):
-----------------------------
{exit_summary if exit_summary else '  - No trades closed'}

Circuit Breaker Status:
-----------------------
12-Hour Loss: {stats.get('circuit_breaker_loss', 0):.2f}%
Status: {stats.get('circuit_breaker_status', 'ACTIVE')}
{stats.get('circuit_breaker_message', '')}

Bot Status: Running normally
Next report in 12 hours.

Detailed trade data attached as CSV.
"""
            
            result = self.send_email(subject, body, [csv_file] if os.path.exists(csv_file) else None)
            if result:
                self.last_report_time = datetime.now()
            return result
            
        except Exception as e:
            logger.error(f"Periodic report failed: {e}")
            return False
    
    def send_circuit_breaker_alert(self, loss_pct: float, balance: float):
        """Send emergency alert when circuit breaker triggers"""
        subject = f"⚠️ CIRCUIT BREAKER TRIGGERED - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body = f"""
EMERGENCY: 10% LOSS LIMIT REACHED
==================================

Current 12-Hour Loss: {loss_pct:.2f}%
Current Balance: ${balance:.2f}

Actions Taken:
- All positions closed immediately
- New trade entries DISABLED
- Bot will auto-resume in 12 hours

Time of trigger: {datetime.now().isoformat()}
Auto-resume at: {(datetime.now() + timedelta(hours=12)).isoformat()}

This is an automatic safety measure to protect your account.
"""
        self.send_email(subject, body)
    
    def send_error_alert(self, error_msg):
        subject = f"OKX Bot ERROR - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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

class TradeTracker:
    def __init__(self):
        self.trades_12h = []
        self.master_csv = "okx_all_trades.csv"
        self.create_master_csv()
    
    def create_master_csv(self):
        if not os.path.exists(self.master_csv):
            headers = ['timestamp', 'symbol', 'direction', 'entry_price', 'exit_price', 
                      'pnl_dollars', 'pnl_percent', 'exit_reason', 'indicators_passed_percent',
                      'trade_duration_minutes', 'max_profit_alpha', 'entry_signals_count', 
                      'signals_flipped_at_exit']
            with open(self.master_csv, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
    
    def add_trade(self, trade_data: dict):
        """Add trade to both 12h list and master CSV"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades_12h.append(trade_data)
        
        # Append to master CSV
        with open(self.master_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data['timestamp'], trade_data['symbol'], trade_data['direction'],
                trade_data['entry_price'], trade_data['exit_price'], trade_data['pnl_dollars'],
                trade_data['pnl_percent'], trade_data['exit_reason'], 
                trade_data['indicators_passed_percent'], trade_data['trade_duration_minutes'],
                trade_data['max_profit_alpha'], trade_data['entry_signals_count'],
                trade_data['signals_flipped_at_exit']
            ])
    
    def generate_12h_csv(self) -> str:
        """Generate CSV for last 12 hours only"""
        filename = f"okx_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        if not self.trades_12h:
            return ""
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trades_12h[0].keys())
            writer.writeheader()
            writer.writerows(self.trades_12h)
        
        return filename
    
    def clear_12h_trades(self):
        """Clear 12h trades after report sent"""
        self.trades_12h = []
    
    def get_12h_stats(self) -> dict:
        """Calculate stats for last 12 hours"""
        if not self.trades_12h:
            return {
                'trades_count': 0, 'won': 0, 'lost': 0, 'win_rate': 0,
                'exit_reasons': {}
            }
        
        won = sum(1 for t in self.trades_12h if t['pnl_dollars'] > 0)
        lost = sum(1 for t in self.trades_12h if t['pnl_dollars'] < 0)
        
        exit_reasons = {}
        for trade in self.trades_12h:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'trades_count': len(self.trades_12h),
            'won': won,
            'lost': lost,
            'win_rate': (won / len(self.trades_12h) * 100) if self.trades_12h else 0,
            'exit_reasons': exit_reasons
        }

class CircuitBreaker:
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.loss_threshold = 0.10  # 10%
        self.window_hours = 12
        self.realized_losses = []  # List of (timestamp, loss_amount)
        self.is_triggered = False
        self.trigger_time = None
    
    def add_realized_loss(self, loss_amount: float):
        """Track realized losses in 12h window"""
        if loss_amount < 0:
            self.realized_losses.append((datetime.now(), abs(loss_amount)))
    
    def get_12h_loss_percentage(self) -> float:
        """Calculate total loss % in last 12 hours"""
        cutoff_time = datetime.now() - timedelta(hours=self.window_hours)
        recent_losses = [loss for timestamp, loss in self.realized_losses if timestamp > cutoff_time]
        total_loss = sum(recent_losses)
        return (total_loss / self.starting_balance) * 100 if self.starting_balance > 0 else 0
    
    def check_trigger(self) -> bool:
        """Check if circuit breaker should trigger"""
        if self.is_triggered:
            # Check if 12 hours passed since trigger
            if datetime.now() - self.trigger_time > timedelta(hours=12):
                logger.info("Circuit breaker auto-resume: 12 hours passed")
                self.reset()
                return False
            return True
        
        loss_pct = self.get_12h_loss_percentage()
        if loss_pct >= (self.loss_threshold * 100):
            self.is_triggered = True
            self.trigger_time = datetime.now()
            logger.error(f"🚨 CIRCUIT BREAKER TRIGGERED: {loss_pct:.2f}% loss in 12h")
            return True
        
        return False
    
    def reset(self):
        """Reset circuit breaker"""
        self.is_triggered = False
        self.trigger_time = None
        self.realized_losses = []
    
    def get_status_message(self) -> str:
        if self.is_triggered:
            resume_time = self.trigger_time + timedelta(hours=12)
            remaining = (resume_time - datetime.now()).total_seconds() / 3600
            return f"PAUSED - Auto-resume in {remaining:.1f} hours"
        return "ACTIVE"

class OKXFuturesBot:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, test_mode: bool = False):
        self.exchange = ccxt.okx({
            'apiKey': api_key, 'secret': api_secret, 'password': passphrase,
            'sandbox': test_mode, 'enableRateLimit': True,
        })
        self.rate_limiter = RateLimiter(max_calls_per_second=5)
        self.indicators = TechnicalIndicators()
        self.positions = {}
        self.leverage = 20
        self.max_longs = 10
        self.max_shorts = 10
        self.max_trades_per_symbol = 3
        self.order_reset_time = datetime.now()
        self.trade_tracker = TradeTracker()
        self.email_notifier = EmailNotifier()
        self.session_active = False
        self.session_start_balance = 0
        self.circuit_breaker = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.logger = logger
        self.monitoring_active = False
        self.portfolio_peak_pnl = 0

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
        """Close a single position and track it"""
        try:
            side = 'sell' if position_data['direction'] == 'long' else 'buy'
            pos_side = position_data['direction']
            
            await self.rate_limited_call(
                self.exchange.create_order,
                symbol=symbol, type='market', side=side, amount=position_data['amount'],
                params={'posSide': pos_side, 'tdMode': 'cross', 'reduceOnly': True}
            )
            
            current_price = (await self.rate_limited_call(self.exchange.fetch_ticker, symbol))['last']
            entry_price = position_data['entry_price']
            entry_time = position_data.get('entry_time', datetime.now())
            trade_duration = (datetime.now() - entry_time).total_seconds() / 60
            
            if position_data['direction'] == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            pnl = position_data['amount'] * abs(current_price - entry_price) * (1 if pnl_pct > 0 else -1)
            
            # Track for circuit breaker
            if pnl < 0:
                self.circuit_breaker.add_realized_loss(pnl)
            
            # Record trade
            trade_data = {
                'symbol': symbol,
                'direction': position_data['direction'],
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_dollars': pnl,
                'pnl_percent': pnl_pct,
                'exit_reason': reason,
                'indicators_passed_percent': position_data.get('indicators_passed_percent', 0),
                'trade_duration_minutes': round(trade_duration, 2),
                'max_profit_alpha': position_data.get('max_profit_alpha', 0),
                'entry_signals_count': position_data.get('entry_signals_count', 0),
                'signals_flipped_at_exit': position_data.get('signals_flipped_count', 0)
            }
            
            self.trade_tracker.add_trade(trade_data)
            
            self.logger.info(f"✗ Closed {symbol} {position_data['direction'].upper()}: ${pnl:.2f} ({pnl_pct:.2f}%) - {reason}")
            
            if symbol in self.positions:
                del self.positions[symbol]
            
            return True
        except Exception as e:
            self.logger.error(f"Close failed {symbol}: {e}")
            return False

    async def close_all_positions_and_orders(self, reason="manual_close"):
        try:
            self.logger.info(f"Closing all positions - Reason: {reason}")
            
            open_orders = await self.rate_limited_call(self.exchange.fetch_open_orders)
            for order in open_orders:
                if order['status'] == 'open':
                    try:
                        await self.rate_limited_call(self.exchange.cancel_order, order['id'], order['symbol'])
                    except:
                        pass
            
            positions = await self.rate_limited_call(self.exchange.fetch_positions)
            for pos in positions:
                if pos.get('contracts', 0) > 0:
                    symbol = pos['symbol']
                    if symbol in self.positions:
                        await self.close_position(symbol, self.positions[symbol], reason)
                    else:
                        size = pos['contracts']
                        side = 'sell' if pos['side'] == 'long' else 'buy'
                        try:
                            await self.rate_limited_call(
                                self.exchange.create_order,
                                symbol=symbol, type='market', side=side, amount=size,
                                params={'posSide': pos['side'], 'tdMode': 'cross', 'reduceOnly': True}
                            )
                        except:
                            pass
            
            self.positions.clear()
            self.portfolio_peak_pnl = 0
            return True
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
            return False

    def start_new_session(self, initial_balance):
        self.session_start_balance = initial_balance
        self.session_active = True
        self.circuit_breaker = CircuitBreaker(initial_balance)
        self.order_reset_time = datetime.now()
        logger.info(f"Session started: ${initial_balance:.2f}")

    def get_available_margin(self):
        try:
            balance = self.exchange.fetch_balance()
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
            
            if actual_margin > max_margin_per_trade:
                max_position_value = max_margin_per_trade * self.leverage
                recalc_amount = (max_position_value / (price * contract_size)) / (self.leverage / 2)
                amount = math.floor(recalc_amount * (10 ** amount_precision)) / (10 ** amount_precision) if amount_precision > 0 else math.floor(recalc_amount)
                actual_margin = (amount * price * contract_size) / self.leverage
            
            return amount, actual_margin
        except Exception as e:
            self.logger.error(f"Size calc error {symbol}: {e}")
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
            
            rsi = self.indicators.calculate_rsi(df['close'])
            if rsi.iloc[-1] > 70:
                long_signals += 1
                indicators_status['RSI'] = 'long'
            elif rsi.iloc[-1] < 30:
                short_signals += 1
                indicators_status['RSI'] = 'short'
            else:
                indicators_status['RSI'] = 'neutral'
            
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            if hist.iloc[-1] < 0:
                long_signals += 1
                indicators_status['MACD'] = 'long'
            elif hist.iloc[-1] > 0:
                short_signals += 1
                indicators_status['MACD'] = 'short'
            else:
                indicators_status['MACD'] = 'neutral'
            
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df['close'])
            if current_price > bb_upper.iloc[-1]:
                long_signals += 1
                indicators_status['BB'] = 'long'
            elif current_price < bb_lower.iloc[-1]:
                short_signals += 1
                indicators_status['BB'] = 'short'
            else:
                indicators_status['BB'] = 'neutral'
            
            k, d = self.indicators.calculate_stochastic(df['high'], df['low'], df['close'])
            if k.iloc[-1] > 80:
                long_signals += 1
                indicators_status['STOCH'] = 'long'
            elif k.iloc[-1] < 20:
                short_signals += 1
                indicators_status['STOCH'] = 'short'
            else:
                indicators_status['STOCH'] = 'neutral'
            
            if current_price > vwap_data['vwap']:
                long_signals += 1
                indicators_status['VWAP'] = 'long'
            else:
                short_signals += 1
                indicators_status['VWAP'] = 'short'
            
            ema = self.indicators.calculate_ema(df['close'])
            if current_price > ema.iloc[-1]:
                long_signals += 1
                indicators_status['EMA'] = 'long'
            elif current_price < ema.iloc[-1]:
                short_signals += 1
                indicators_status['EMA'] = 'short'
            else:
                indicators_status['EMA'] = 'neutral'
            
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
            base_symbol = pos_symbol.split(':')[0]
            symbol_trade_counts[base_symbol] = symbol_trade_counts.get(base_symbol, 0) + 1
        
        for idx, symbol_info in enumerate(symbols[:230], 1):
            symbol = symbol_info['symbol']
            base_symbol = symbol.split(':')[0]
            
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
                        'entry_indicators': analysis_15m['indicators'].copy(),
                        'entry_signals_count': analysis_15m.get('long_signals', 0) if analysis_4h['direction'] == 'long' else analysis_15m.get('short_signals', 0)
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
        """Get balanced long/short counts"""
        try:
            positions = self.exchange.fetch_positions()
            open_orders = self.exchange.fetch_open_orders()
            
            long_count = short_count = 0
            position_symbols = {'long': [], 'short': []}
            
            for p in positions:
                if p.get('contracts', 0) > 0:
                    if p['side'] == 'long':
                        long_count += 1
                        position_symbols['long'].append(p['symbol'])
                    else:
                        short_count += 1
                        position_symbols['short'].append(p['symbol'])
            
            for o in open_orders:
                if o.get('status') == 'open':
                    symbol = o['symbol']
                    if o['side'] == 'buy' and symbol not in position_symbols['long']:
                        long_count += 1
                    elif o['side'] == 'sell' and symbol not in position_symbols['short']:
                        short_count += 1
            
            self.logger.info(f"Current exposure: {long_count} longs, {short_count} shorts (Total: {long_count + short_count}/20)")
            return long_count, short_count
        except Exception as e:
            self.logger.error(f"Exposure check error: {e}")
            return 0, 0

    def sync_positions_from_exchange(self, positions=None, open_orders=None):
        """Sync with exchange state"""
        try:
            if positions:
                for pos in positions:
                    if pos.get('contracts', 0) > 0:
                        symbol = pos['symbol']
                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'direction': 'long' if pos['side'] == 'long' else 'short',
                                'entry_price': pos.get('entryPrice', 0),
                                'amount': pos.get('contracts', 0),
                                'type': 'position',
                                'max_profit_alpha': 0,
                                'trailing_tier': 0,
                                'entry_indicators': {},
                                'entry_time': datetime.now(),
                                'entry_signals_count': 0,
                                'signal_reversal_mode': False
                            }
                        else:
                            self.positions[symbol]['entry_price'] = pos.get('entryPrice', self.positions[symbol]['entry_price'])
                            self.positions[symbol]['amount'] = pos.get('contracts', self.positions[symbol]['amount'])
            
            if open_orders:
                for order in open_orders:
                    if order.get('status') == 'open':
                        symbol = order['symbol']
                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'direction': 'long' if order['side'] == 'buy' else 'short',
                                'entry_price': order.get('price', 0),
                                'amount': order.get('amount', 0),
                                'type': 'order',
                                'order_id': order.get('id'),
                                'max_profit_alpha': 0,
                                'trailing_tier': 0,
                                'entry_indicators': {},
                                'entry_time': datetime.now(),
                                'entry_signals_count': 0,
                                'signal_reversal_mode': False
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

    async def update_stop_loss(self, symbol: str, new_sl_price: float):
        """Update stop loss for existing position"""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            for order in orders:
                if order.get('info', {}).get('ordType') == 'conditional':
                    try:
                        self.exchange.cancel_order(order['id'], symbol)
                        await asyncio.sleep(0.5)
                    except:
                        pass
            
            position_data = self.positions.get(symbol)
            if not position_data:
                return False
            
            side = 'sell' if position_data['direction'] == 'long' else 'buy'
            pos_side = position_data['direction']
            
            params = {
                'posSide': pos_side,
                'tdMode': 'cross',
                'reduceOnly': True,
                'stopLoss': {
                    'triggerPrice': new_sl_price,
                    'price': new_sl_price,
                    'triggerPriceType': 'last'
                }
            }
            
            self.exchange.create_order(
                symbol=symbol, type='market', side=side,
                amount=position_data['amount'], params=params
            )
            
            self.logger.info(f"↑ Updated SL {symbol}: ${new_sl_price:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"SL update failed {symbol}: {e}")
            return False

    async def check_signal_reversal(self, symbol: str, position_data: dict) -> Tuple[bool, int]:
        """Check if 70% of entry indicators have flipped - returns (flipped, flip_count)"""
        try:
            entry_indicators = position_data.get('entry_indicators', {})
            if not entry_indicators:
                return False, 0
            
            df_15m = self.fetch_ohlcv(symbol, '15m', 100)
            current_analysis = self.analyze_symbol(df_15m)
            
            if not current_analysis or current_analysis['direction'] == 'neutral':
                return False, 0
            
            current_indicators = current_analysis.get('indicators', {})
            original_direction = position_data['direction']
            
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
                return False, 0
            
            flip_percentage = (flipped_count / total_indicators) * 100
            
            if flip_percentage >= 70:
                self.logger.info(f"⚠ Signal reversal {symbol}: {flip_percentage:.0f}% indicators flipped ({flipped_count}/{total_indicators})")
                return True, flipped_count
            
            return False, flipped_count
            
        except Exception as e:
            self.logger.error(f"Signal check error {symbol}: {e}")
            return False, 0

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
                
                for pos in positions:
                    if pos.get('contracts', 0) <= 0:
                        continue
                    
                    symbol = pos['symbol']
                    if symbol not in self.positions:
                        continue
                    
                    position_data = self.positions[symbol]
                    entry_price = position_data['entry_price']
                    current_price = pos.get('markPrice', pos.get('lastPrice', entry_price))
                    
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
                    
                    should_close = False
                    close_reason = ""
                    
                    # Check signal reversal (70% threshold)
                    signal_flipped, flip_count = await self.check_signal_reversal(symbol, position_data)
                    position_data['signals_flipped_count'] = flip_count
                    
                    if signal_flipped:
                        if profit_alpha > 0:
                            # In profit - exit immediately
                            should_close = True
                            close_reason = "signal_reversal_profit_exit"
                        else:
                            # In loss - activate recovery trailing
                            if not position_data.get('signal_reversal_mode'):
                                position_data['signal_reversal_mode'] = True
                                self.logger.info(f"🔄 {symbol} entering recovery mode (loss: {profit_alpha:.2f}α)")
                            
                            # Recovery trailing logic
                            new_sl_price = None
                            if profit_alpha >= -0.5 and position_data.get('recovery_tier', 0) < 2:
                                # Improved to -0.5α, move SL to -0.75α
                                if position_data['direction'] == 'long':
                                    new_sl_price = entry_price - (0.75 * alpha)
                                else:
                                    new_sl_price = entry_price + (0.75 * alpha)
                                position_data['recovery_tier'] = 2
                                self.logger.info(f"  Recovery tier 2: SL moved to -0.75α")
                            
                            elif profit_alpha >= 0 and position_data.get('recovery_tier', 0) < 3:
                                # Reached breakeven
                                new_sl_price = entry_price
                                position_data['recovery_tier'] = 3
                                self.logger.info(f"  Recovery tier 3: Breakeven reached")
                            
                            elif profit_alpha >= 1.0 and position_data.get('recovery_tier', 0) < 4:
                                # Recovered to +1α, switch to normal trailing
                                if position_data['direction'] == 'long':
                                    new_sl_price = entry_price + (0.5 * alpha)
                                else:
                                    new_sl_price = entry_price - (0.5 * alpha)
                                position_data['recovery_tier'] = 4
                                position_data['signal_reversal_mode'] = False
                                self.logger.info(f"  Recovery complete: Switched to normal trailing at +0.5α")
                            
                            if new_sl_price:
                                await self.update_stop_loss(symbol, new_sl_price)
                    
                    # Normal trailing stop logic (if not in recovery mode)
                    if not should_close and not position_data.get('signal_reversal_mode', False):
                        current_tier = position_data.get('trailing_tier', 0)
                        new_sl_price = None
                        
                        if profit_alpha >= 4.0:
                            # Hit TP target
                            should_close = True
                            close_reason = "take_profit_hit"
                        
                        elif profit_alpha >= 3.0 and current_tier < 2:
                            # Tier 2: Move SL to +1.5α
                            if position_data['direction'] == 'long':
                                new_sl_price = entry_price + (1.5 * alpha)
                            else:
                                new_sl_price = entry_price - (1.5 * alpha)
                            position_data['trailing_tier'] = 2
                        
                        elif profit_alpha >= 2.0 and current_tier < 1:
                            # Tier 1: Move SL to breakeven
                            new_sl_price = entry_price
                            position_data['trailing_tier'] = 1
                        
                        if new_sl_price:
                            await self.update_stop_loss(symbol, new_sl_price)
                    
                    if should_close:
                        await self.close_position(symbol, position_data, close_reason)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(10)

    async def place_bracket_order(self, trade_info: Dict, amount: float):
        """Place bracket order with SL/TP (2:1 ratio with 0.1% buffer)"""
        symbol = trade_info['symbol']
        direction = trade_info['direction']
        vwap_data = trade_info['vwap_data']
        
        try:
            self.exchange.set_leverage(self.leverage, symbol)
            entry_price = vwap_data['vwap']
            
            # Calculate alpha and add 0.1% buffer to SL
            if direction == 'long':
                side, pos_side = 'buy', 'long'
                raw_sl = vwap_data['lower_band'] * 0.998
                sl_buffer = entry_price * 0.001  # 0.1% buffer
                sl_price = raw_sl - sl_buffer
                risk_distance = entry_price - sl_price
                tp_price = entry_price + (2 * risk_distance)  # 2:1 ratio
            else:
                side, pos_side = 'sell', 'short'
                raw_sl = vwap_data['upper_band'] * 1.002
                sl_buffer = entry_price * 0.001  # 0.1% buffer
                sl_price = raw_sl + sl_buffer
                risk_distance = sl_price - entry_price
                tp_price = entry_price - (2 * risk_distance)  # 2:1 ratio
            
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
            
            order = self.exchange.create_order(
                symbol=symbol, type='limit', side=side, amount=amount,
                price=entry_price, params=params
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
                'entry_signals_count': trade_info.get('entry_signals_count', 0),
                'max_profit_alpha': 0,
                'trailing_tier': 0,
                'type': 'order',
                'entry_time': datetime.now(),
                'signal_reversal_mode': False,
                'signals_flipped_count': 0
            }
            
            risk_pct = (abs(entry_price - sl_price) / entry_price) * 100
            reward_pct = (abs(tp_price - entry_price) / entry_price) * 100
            
            self.logger.info(f"✓ {direction.upper()} {symbol}: Entry ${entry_price:.6f}, SL ${sl_price:.6f} (+0.1% buffer), TP ${tp_price:.6f}")
            self.logger.info(f"  Risk {risk_pct:.2f}%, Reward {reward_pct:.2f}%, R:R 1:{reward_pct/risk_pct:.1f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Bracket order failed {symbol}: {e}")
            try:
                simple_order = self.exchange.create_order(
                    symbol=symbol, type='limit', side=side, amount=amount,
                    price=entry_price, params={'posSide': pos_side, 'tdMode': 'cross'}
                )
                
                self.positions[symbol] = {
                    'direction': direction, 'entry_price': entry_price, 'amount': amount,
                    'tp_price': tp_price, 'sl_price': sl_price, 'order_id': simple_order['id'],
                    'indicators_passed_percent': trade_info.get('indicators_passed_percent', 0),
                    'entry_indicators': trade_info.get('entry_indicators', {}),
                    'entry_signals_count': trade_info.get('entry_signals_count', 0),
                    'max_profit_alpha': 0, 'trailing_tier': 0, 'type': 'order',
                    'entry_time': datetime.now(), 'signal_reversal_mode': False,
                    'signals_flipped_count': 0
                }
                
                self.logger.info(f"✓ {direction.upper()} {symbol} (simple order)")
                return True
            except Exception as e2:
                self.logger.error(f"All orders failed {symbol}: {e2}")
                return False

    async def run_trading_cycle(self):
        """Main cycle with 10% circuit breaker"""
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
            
            # Check circuit breaker
            if self.circuit_breaker.check_trigger():
                loss_pct = self.circuit_breaker.get_12h_loss_percentage()
                self.logger.error(f"🚨 CIRCUIT BREAKER ACTIVE: {loss_pct:.2f}% loss")
                
                if len(self.positions) > 0:
                    await self.close_all_positions_and_orders("circuit_breaker_10%_loss")
                    self.email_notifier.send_circuit_breaker_alert(loss_pct, current_balance)
                
                return
            
            # Send 12-hour report
            if (datetime.now() - self.email_notifier.last_report_time).total_seconds() >= 43200:
                positions = self.exchange.fetch_positions()
                active_longs = sum(1 for p in positions if p.get('contracts', 0) > 0 and p['side'] == 'long')
                active_shorts = sum(1 for p in positions if p.get('contracts', 0) > 0 and p['side'] == 'short')
                
                session_pnl = current_balance - self.session_start_balance
                session_pnl_pct = (session_pnl / self.session_start_balance * 100) if self.session_start_balance > 0 else 0
                
                trade_stats = self.trade_tracker.get_12h_stats()
                
                stats = {
                    'current_balance': current_balance,
                    'session_start_balance': self.session_start_balance,
                    'session_pnl': session_pnl,
                    'session_pnl_pct': session_pnl_pct,
                    'active_positions': active_longs + active_shorts,
                    'active_longs': active_longs,
                    'active_shorts': active_shorts,
                    'trades_last_12h': trade_stats['trades_count'],
                    'won_last_12h': trade_stats['won'],
                    'lost_last_12h': trade_stats['lost'],
                    'win_rate_12h': trade_stats['win_rate'],
                    'exit_reasons': trade_stats['exit_reasons'],
                    'peak_pnl': self.portfolio_peak_pnl,
                    'max_drawdown': 0,
                    'circuit_breaker_loss': self.circuit_breaker.get_12h_loss_percentage(),
                    'circuit_breaker_status': self.circuit_breaker.get_status_message(),
                    'circuit_breaker_message': ''
                }
                
                csv_file = self.trade_tracker.generate_12h_csv()
                if csv_file:
                    self.email_notifier.send_periodic_report(stats, csv_file)
                    self.trade_tracker.clear_12h_trades()
            
            if datetime.now() - self.order_reset_time > timedelta(hours=1):
                self.logger.info("1-hour reset")
                await self.cancel_all_orders()
                self.order_reset_time = datetime.now()
                await asyncio.sleep(5)
            
            long_count, short_count = self.get_current_exposure()
            
            long_slots = self.max_longs - long_count
            short_slots = self.max_shorts - short_count
            
            if long_slots == 0 and short_slots == 0:
                self.logger.info("At capacity (10/10)")
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
            total_account = balance_info.get('USDT', {}).get('total', available_margin * 1.25)
            
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
    API_KEY = "5e22e924-0001-4675-b235-632c467c818e"
    API_SECRET = "811817EFE46121411C683AEB142B14CE"
    PASSPHRASE = "#Dinywa15"
    TEST_MODE = False
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   OKX Futures Bot - FINAL VERSION         ║
    ║                                            ║
    ║   Risk Management:                         ║
    ║   • 10 Long / 10 Short (balanced)         ║
    ║   • 2:1 Reward:Risk ratio                 ║
    ║   • SL with 0.1% buffer                   ║
    ║   • 10% circuit breaker (12h window)      ║
    ║                                            ║
    ║   Exit Strategy:                           ║
    ║   • 70% signal flip detection             ║
    ║   • Smart profit/loss handling            ║
    ║   • Recovery trailing for losses          ║
    ║   • 3-tier trailing stops                 ║
    ║                                            ║
    ║   Features:                                ║
    ║   • Max 3 trades per symbol               ║
    ║   • Real-time monitoring (10s)            ║
    ║   • 12-hour CSV reports via email         ║
    ║   • Detailed exit reason tracking         ║
    ║   • Auto-resume after circuit breaker     ║
    ╚════════════════════════════════════════════╝
    """)
    
    bot = OKXFuturesBot(API_KEY, API_SECRET, PASSPHRASE, TEST_MODE)
    
    try:
        margin = bot.get_available_margin()
        logger.info(f"Connected: ${margin:.2f}")
        logger.info(f"Master CSV: {bot.trade_tracker.master_csv}")
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
                
                # Generate final report
                if bot.session_active:
                    current_balance = bot.get_available_margin()
                    positions = bot.exchange.fetch_positions()
                    active_longs = sum(1 for p in positions if p.get('contracts', 0) > 0 and p['side'] == 'long')
                    active_shorts = sum(1 for p in positions if p.get('contracts', 0) > 0 and p['side'] == 'short')
                    
                    session_pnl = current_balance - bot.session_start_balance
                    session_pnl_pct = (session_pnl / bot.session_start_balance * 100) if bot.session_start_balance > 0 else 0
                    
                    trade_stats = bot.trade_tracker.get_12h_stats()
                    
                    stats = {
                        'current_balance': current_balance,
                        'session_start_balance': bot.session_start_balance,
                        'session_pnl': session_pnl,
                        'session_pnl_pct': session_pnl_pct,
                        'active_positions': active_longs + active_shorts,
                        'active_longs': active_longs,
                        'active_shorts': active_shorts,
                        'trades_last_12h': trade_stats['trades_count'],
                        'won_last_12h': trade_stats['won'],
                        'lost_last_12h': trade_stats['lost'],
                        'win_rate_12h': trade_stats['win_rate'],
                        'exit_reasons': trade_stats['exit_reasons'],
                        'peak_pnl': bot.portfolio_peak_pnl,
                        'max_drawdown': 0,
                        'circuit_breaker_loss': bot.circuit_breaker.get_12h_loss_percentage(),
                        'circuit_breaker_status': 'MANUAL_SHUTDOWN',
                        'circuit_breaker_message': 'Bot manually stopped by user'
                    }
                    
                    csv_file = bot.trade_tracker.generate_12h_csv()
                    if csv_file:
                        bot.email_notifier.send_periodic_report(stats, csv_file)
                
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
