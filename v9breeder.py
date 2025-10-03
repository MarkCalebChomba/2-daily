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
        subject = f"OKX Bot Session Completed - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body = "Trading session completed. See attached CSV for details."
        self.send_email(subject, body, [csv_file] if os.path.exists(csv_file) else None)
    
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
    
    def add_trade(self, symbol, direction, entry_price, exit_price, pnl, pnl_percentage, status, indicators_passed_pct):
        self.trades_data.append({
            'symbol': symbol, 'direction': direction, 'entry_price': entry_price,
            'exit_price': exit_price, 'pnl': pnl, 'pnl_percentage': pnl_percentage,
            'status': status, 'indicators_passed_percent': indicators_passed_pct,
            'timestamp': datetime.now().isoformat()
        })
        self.total_trades += 1
        if status == 'won':
            self.trades_won += 1
        elif status == 'lost':
            self.trades_lost += 1
    
    def end_session(self, final_balance, reason="profit_target"):
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

class OKXFuturesBot:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, test_mode: bool = False):
        self.exchange = ccxt.okx({
            'apiKey': api_key, 'secret': api_secret, 'password': passphrase,
            'sandbox': test_mode, 'enableRateLimit': True,
        })
        self.rate_limiter = RateLimiter(max_calls_per_second=5)
        self.indicators = TechnicalIndicators()
        self.positions = {}  # Single dict for all positions
        self.leverage = 20
        self.max_longs = 10
        self.max_shorts = 10
        self.max_total_positions = 20
        self.order_reset_time = datetime.now()
        self.session_tracker = SessionTracker()
        self.email_notifier = EmailNotifier()
        self.session_active = False
        self.session_start_balance = 0
        self.profit_target_percentage = 50
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.logger = logger

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
                    except:
                        pass
            
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
                        
                        # Get indicators percent if tracked
                        indicators_pct = self.positions.get(symbol, {}).get('indicators_passed_percent', 0)
                        
                        self.session_tracker.add_trade(symbol, pos['side'], entry_price, current_price, pnl, pnl_pct, status, indicators_pct)
                        self.logger.info(f"Closed {symbol} {pos['side']}: ${pnl:.2f} ({pnl_pct:.2f}%)")
                    except Exception as e:
                        self.logger.error(f"Close failed {symbol}: {e}")
            
            self.positions.clear()
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
            
            # 1. RSI - INVERTED
            rsi = self.indicators.calculate_rsi(df['close'])
            if rsi.iloc[-1] > 70:
                long_signals += 1
                indicators_status['RSI'] = f'LONG({rsi.iloc[-1]:.1f})'
            elif rsi.iloc[-1] < 30:
                short_signals += 1
                indicators_status['RSI'] = f'SHORT({rsi.iloc[-1]:.1f})'
            else:
                indicators_status['RSI'] = 'NEUTRAL'
            
            # 2. MACD - INVERTED
            macd, signal, hist = self.indicators.calculate_macd(df['close'])
            if hist.iloc[-1] < 0:
                long_signals += 1
                indicators_status['MACD'] = 'LONG'
            elif hist.iloc[-1] > 0:
                short_signals += 1
                indicators_status['MACD'] = 'SHORT'
            else:
                indicators_status['MACD'] = 'NEUTRAL'
            
            # 3. Bollinger Bands - INVERTED
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df['close'])
            if current_price > bb_upper.iloc[-1]:
                long_signals += 1
                indicators_status['BB'] = 'LONG'
            elif current_price < bb_lower.iloc[-1]:
                short_signals += 1
                indicators_status['BB'] = 'SHORT'
            else:
                indicators_status['BB'] = 'NEUTRAL'
            
            # 4. Stochastic - INVERTED
            k, d = self.indicators.calculate_stochastic(df['high'], df['low'], df['close'])
            if k.iloc[-1] > 80:
                long_signals += 1
                indicators_status['STOCH'] = f'LONG({k.iloc[-1]:.1f})'
            elif k.iloc[-1] < 20:
                short_signals += 1
                indicators_status['STOCH'] = f'SHORT({k.iloc[-1]:.1f})'
            else:
                indicators_status['STOCH'] = 'NEUTRAL'
            
            # 5. Price vs VWAP - INVERTED
            if current_price > vwap_data['vwap']:
                long_signals += 1
                indicators_status['VWAP'] = 'LONG'
            else:
                short_signals += 1
                indicators_status['VWAP'] = 'SHORT'
            
            # 6. EMA - INVERTED
            ema = self.indicators.calculate_ema(df['close'])
            if current_price > ema.iloc[-1]:
                long_signals += 1
                indicators_status['EMA'] = 'LONG'
            elif current_price < ema.iloc[-1]:
                short_signals += 1
                indicators_status['EMA'] = 'SHORT'
            else:
                indicators_status['EMA'] = 'NEUTRAL'
            
            # 7. ADX - INVERTED
            adx = self.indicators.calculate_adx(df['high'], df['low'], df['close'])
            if not adx.isna().all() and adx.iloc[-1] > 25:
                if current_price > df['close'].iloc[-10]:
                    long_signals += 1
                    indicators_status['ADX'] = f'LONG({adx.iloc[-1]:.1f})'
                elif current_price < df['close'].iloc[-10]:
                    short_signals += 1
                    indicators_status['ADX'] = f'SHORT({adx.iloc[-1]:.1f})'
                else:
                    indicators_status['ADX'] = 'NEUTRAL'
            else:
                indicators_status['ADX'] = 'WEAK'
            
            # 8. Williams %R - INVERTED
            williams_r = self.indicators.calculate_williams_r(df['high'], df['low'], df['close'])
            if not williams_r.isna().all():
                if williams_r.iloc[-1] > -20:
                    long_signals += 1
                    indicators_status['WILL'] = 'LONG'
                elif williams_r.iloc[-1] < -80:
                    short_signals += 1
                    indicators_status['WILL'] = 'SHORT'
                else:
                    indicators_status['WILL'] = 'NEUTRAL'
            else:
                indicators_status['WILL'] = 'WEAK'
            
            # 9. CCI - INVERTED
            cci = self.indicators.calculate_cci(df['high'], df['low'], df['close'])
            if not cci.isna().all():
                if cci.iloc[-1] > 100:
                    long_signals += 1
                    indicators_status['CCI'] = f'LONG({cci.iloc[-1]:.0f})'
                elif cci.iloc[-1] < -100:
                    short_signals += 1
                    indicators_status['CCI'] = f'SHORT({cci.iloc[-1]:.0f})'
                else:
                    indicators_status['CCI'] = 'NEUTRAL'
            else:
                indicators_status['CCI'] = 'WEAK'
            
            # 10. MFI - INVERTED
            mfi = self.indicators.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
            if not mfi.isna().all():
                if mfi.iloc[-1] > 80:
                    long_signals += 1
                    indicators_status['MFI'] = f'LONG({mfi.iloc[-1]:.1f})'
                elif mfi.iloc[-1] < 20:
                    short_signals += 1
                    indicators_status['MFI'] = f'SHORT({mfi.iloc[-1]:.1f})'
                else:
                    indicators_status['MFI'] = 'NEUTRAL'
            else:
                indicators_status['MFI'] = 'WEAK'
            
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
        """Find opportunities with detailed logging"""
        longs, shorts = [], []
        
        for idx, symbol_info in enumerate(symbols[:230], 1):
            symbol = symbol_info['symbol']
            
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
                        'indicators_passed_percent': analysis_15m.get('indicators_passed_percent', 0)
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
            self.positions.clear()
            
            if positions:
                for pos in positions:
                    if pos.get('contracts', 0) > 0:
                        self.positions[pos['symbol']] = {
                            'direction': 'long' if pos['side'] == 'long' else 'short',
                            'entry_price': pos.get('entryPrice', 0),
                            'amount': pos.get('contracts', 0),
                            'type': 'position'
                        }
            
            if open_orders:
                for order in open_orders:
                    if order.get('status') == 'open' and order['symbol'] not in self.positions:
                        self.positions[order['symbol']] = {
                            'direction': 'long' if order['side'] == 'buy' else 'short',
                            'entry_price': order.get('price', 0),
                            'amount': order.get('amount', 0),
                            'type': 'order',
                            'order_id': order.get('id')
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

    async def place_bracket_order(self, trade_info: Dict, amount: float):
        """Place bracket order with SL/TP on exchange (from working code)"""
        symbol = trade_info['symbol']
        direction = trade_info['direction']
        vwap_data = trade_info['vwap_data']
        
        try:
            self.exchange.set_leverage(self.leverage, symbol)
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
            
            # OKX bracket order params (proven to work)
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
                'indicators_passed_percent': trade_info.get('indicators_passed_percent', 0)
            }
            
            risk_pct = (abs(entry_price - sl_price) / entry_price) * 100
            reward_pct = (abs(tp_price - entry_price) / entry_price) * 100
            
            self.logger.info(f"✓ {direction.upper()} {symbol}: Entry ${entry_price:.6f}, SL ${sl_price:.6f}, TP ${tp_price:.6f}")
            self.logger.info(f"  Risk {risk_pct:.2f}%, Reward {reward_pct:.2f}%, R:R 1:{reward_pct/risk_pct:.1f}, Indicators {trade_info.get('indicators_passed_percent', 0):.0f}%")
            return True
            
        except Exception as e:
            self.logger.error(f"Bracket order failed {symbol}: {e}")
            # Fallback to simple limit order
            try:
                simple_order = self.exchange.create_order(
                    symbol=symbol, type='limit', side=side, amount=amount,
                    price=entry_price, params={'posSide': pos_side, 'tdMode': 'cross'}
                )
                
                self.positions[symbol] = {
                    'direction': direction, 'entry_price': entry_price, 'amount': amount,
                    'tp_price': tp_price, 'sl_price': sl_price, 'order_id': simple_order['id'],
                    'indicators_passed_percent': trade_info.get('indicators_passed_percent', 0)
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
            
            # Get balance
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
            
            # Check profit target
            if self.check_profit_target(current_balance):
                await self.close_all_positions_and_orders("profit_target")
                await asyncio.sleep(5)
                final_balance = self.get_available_margin()
                csv_file = self.session_tracker.end_session(final_balance, "profit_target")
                self.email_notifier.send_session_report(csv_file)
                
                new_balance = self.get_available_margin()
                self.start_new_session(new_balance)
                self.logger.info("NEW SESSION STARTED")
                return
            
            # 1-hour reset
            if datetime.now() - self.order_reset_time > timedelta(hours=1):
                self.logger.info("1-hour reset")
                await self.cancel_all_orders()
                self.positions.clear()
                self.order_reset_time = datetime.now()
                await asyncio.sleep(5)
            
            # Get current exposure with balanced counts
            long_count, short_count = self.get_current_exposure()
            
            long_slots = self.max_longs - long_count
            short_slots = self.max_shorts - short_count
            
            if long_slots == 0 and short_slots == 0:
                self.logger.info("At capacity (10/10)")
                return
            
            # Get opportunities
            symbols = self.get_futures_symbols()
            if not symbols:
                return
            
            longs, shorts = self.find_trading_opportunities(symbols)
            
            # BALANCED SELECTION - key logic
            selected_longs = longs[:long_slots] if long_slots > 0 else []
            selected_shorts = shorts[:short_slots] if short_slots > 0 else []
            
            total_selected = len(selected_longs) + len(selected_shorts)
            
            if total_selected == 0:
                self.logger.info("No qualified opportunities")
                return
            
            self.logger.info(f"Selected: {len(selected_longs)} longs, {len(selected_shorts)} shorts")
            
            # Calculate sizing
            available_margin = self.get_available_margin()
            balance_info = self.exchange.fetch_balance()
            total_account = balance_info.get('USDT', {}).get('total', available_margin * 1.25)
            
            total_margin = available_margin * 0.8
            
            # Place orders
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
            
            # Sync after orders
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
    ║   OKX Balanced Futures Bot - PRODUCTION   ║
    ║   • 10 Long / 10 Short (balanced)         ║
    ║   • INVERTED indicators (10 total)        ║
    ║   • Bracket orders on exchange            ║
    ║   • Rate limiting (5 calls/sec)           ║
    ║   • Auto-close at 50% profit             ║
    ║   • Email notifications                   ║
    ║   • CSV with indicator percentages        ║
    ║   • Server-ready autonomous operation     ║
    ╚════════════════════════════════════════════╝
    """)
    
    bot = OKXFuturesBot(API_KEY, API_SECRET, PASSPHRASE, TEST_MODE)
    
    try:
        margin = bot.get_available_margin()
        logger.info(f"Connected: ${margin:.2f}")
        logger.info(f"CSV: {bot.session_tracker.csv_filename}")
        logger.info(f"Email: {bot.email_notifier.receiver}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        bot.email_notifier.send_error_alert(f"Startup failed: {str(e)}")
        return
    
    while True:
        try:
            await bot.run_trading_cycle()
            logger.info("Waiting 60s...")
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Manual shutdown...")
            if bot.session_active:
                final_balance = bot.get_available_margin()
                csv_file = bot.session_tracker.end_session(final_balance, "manual_shutdown")
                bot.email_notifier.send_session_report(csv_file)
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            bot.email_notifier.send_error_alert(f"Loop error: {str(e)}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
