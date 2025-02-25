import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..brokers.mt5_adapter import MT5Adapter

class BaseStrategy(ABC):
    def __init__(self, symbol: str, timeframe: str = '5m',
                 mt5_login: Optional[int] = None,
                 mt5_password: Optional[str] = None,
                 mt5_server: str = "Exness-MT5Trial9"):
        """
        Initialize base strategy.
        
        Args:
            symbol: Trading pair symbol (e.g. 'EURUSD')
            timeframe: Trading timeframe (e.g. '1m', '5m', '1h', '1d')
            mt5_login: MT5 account login number
            mt5_password: MT5 account password
            mt5_server: MT5 server name (e.g., "Exness-MT5Trial9", "Exness-MT5Real2")
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.position_size = 0.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.total_exposure_limit = 0.06  # 6% total exposure limit
        
        # Initialize MT5 connection
        if mt5_login and mt5_password:
            self.broker = MT5Adapter(mt5_login, mt5_password, mt5_server)
        else:
            raise ValueError("MT5 login and password are required")
            
    def calculate_position_size(self, stop_loss_pips: float) -> float:
        """Calculate position size based on account risk parameters."""
        account_info = self.broker.get_account_info()
        risk_amount = float(account_info['balance']) * self.risk_per_trade
        return self.broker.calculate_position_size(
            self.symbol,
            risk_amount,
            stop_loss_pips
        )
        
    def get_market_data(self, lookback: int = 100) -> pd.DataFrame:
        """Get market data for analysis."""
        return self.broker.get_market_data(
            self.symbol,
            self.timeframe,
            lookback
        )
        
    @abstractmethod
    def generate_signals(self) -> Dict:
        """Generate trading signals based on strategy logic."""
        pass
        
    def execute_trade(self, signal: Dict) -> bool:
        """Execute trade based on generated signal."""
        if not signal['type'] or not self.check_risk_limits():
            return False
            
        try:
            # Execute with MT5
            if signal['type'] == 'MARKET':
                return self.broker.execute_market_order(
                    symbol=self.symbol,
                    side=signal['side'],
                    volume=signal['volume'],
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit')
                )
            else:  # LIMIT order
                return self.broker.execute_limit_order(
                    symbol=self.symbol,
                    side=signal['side'],
                    volume=signal['volume'],
                    price=signal['price'],
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit')
                )
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False
            
    def check_risk_limits(self) -> bool:
        """Check if current exposure is within risk limits."""
        try:
            positions = self.broker.get_positions()
            total_exposure = sum(float(p['margin']) for p in positions)
            account_info = self.broker.get_account_info()
            return total_exposure <= float(account_info['balance']) * self.total_exposure_limit
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False
            
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators."""
        # EMA calculations
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema13'] = df['close'].ewm(span=13, adjust=False).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume relative to average
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Add VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Bollinger Bands
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['stddev'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['stddev'] * 2)
        df['bb_lower'] = df['sma20'] - (df['stddev'] * 2)
        
        return df 