import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, symbol: str, timeframe: mt5.TIMEFRAME = mt5.TIMEFRAME_M5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.position_size = 0.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.total_exposure_limit = 0.06  # 6% total exposure limit
        
    def calculate_position_size(self, stop_loss_pips: float) -> float:
        """Calculate position size based on account risk parameters."""
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
            
        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError("Failed to get account info")
            
        equity = account_info.equity
        risk_amount = equity * self.risk_per_trade
        
        # Get symbol info for pip value calculation
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Failed to get symbol info for {self.symbol}")
            
        pip_value = symbol_info.trade_tick_value
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        return position_size
        
    def get_market_data(self, lookback: int = 100) -> pd.DataFrame:
        """Get market data for analysis."""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, lookback)
        if rates is None:
            raise RuntimeError(f"Failed to get market data for {self.symbol}")
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    @abstractmethod
    def generate_signals(self) -> Dict:
        """Generate trading signals based on strategy logic."""
        pass
        
    @abstractmethod
    def execute_trade(self, signal: Dict) -> bool:
        """Execute trade based on generated signal."""
        pass
        
    def check_risk_limits(self) -> bool:
        """Check if current exposure is within risk limits."""
        if not mt5.initialize():
            return False
            
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return True
            
        total_exposure = sum(pos.volume for pos in positions)
        account_info = mt5.account_info()
        
        if account_info is None:
            return False
            
        current_exposure_ratio = (total_exposure * mt5.symbol_info(self.symbol).trade_tick_value) / account_info.equity
        
        return current_exposure_ratio <= self.total_exposure_limit
        
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
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        return df 