import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

class QuickProfitCapture:
    def __init__(self):
        self.momentum_exit_threshold = 0.7  # 30% reduction at momentum slowdown
        self.stall_exit_threshold = 0.7  # 30% reduction at 3-candle stall
        self.trailing_stop_step = 5  # pips
        self.profit_retracement_exit = 0.4  # 40% profit retracement
        
    def calculate_momentum_strength(self, df: pd.DataFrame) -> float:
        """Calculate current momentum strength."""
        # Use multiple indicators for momentum strength
        # 1. Rate of change
        roc = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        # 2. Average Directional Index (ADX)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        tr = pd.DataFrame({
            'h-l': df['high'] - df['low'],
            'h-pc': abs(df['high'] - df['close'].shift()),
            'l-pc': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1] / 100
        
        # 3. Volume momentum
        volume_momentum = df['tick_volume'].iloc[-1] / df['tick_volume'].rolling(20).mean().iloc[-1]
        
        # Combine indicators
        momentum_strength = (roc + adx + (volume_momentum - 1)) / 3
        return min(max(momentum_strength, 0), 1)  # Normalize between 0 and 1
        
    def detect_price_stall(self, df: pd.DataFrame) -> bool:
        """Detect if price is stalling (3-candle pattern)."""
        last_3_candles = df.tail(3)
        
        # Check for tight price range
        price_range = (last_3_candles['high'].max() - last_3_candles['low'].min()) / df['close'].iloc[-1]
        
        # Check for decreasing volume
        volume_trend = last_3_candles['tick_volume'].is_monotonic_decreasing
        
        # Check for small body candles
        bodies = abs(last_3_candles['close'] - last_3_candles['open']) / last_3_candles['close']
        small_bodies = (bodies < 0.0005).all()  # 0.05% threshold
        
        return price_range < 0.001 and volume_trend and small_bodies
        
    def calculate_trailing_stop(self, position: Dict, current_price: float) -> Optional[float]:
        """Calculate trailing stop level."""
        if position['type'] == 'BUY':
            profit_pips = (current_price - position['entry_price']) / position['point']
            if profit_pips > self.trailing_stop_step:
                return max(
                    position['entry_price'],
                    current_price - (self.trailing_stop_step * position['point'])
                )
        else:  # SELL
            profit_pips = (position['entry_price'] - current_price) / position['point']
            if profit_pips > self.trailing_stop_step:
                return min(
                    position['entry_price'],
                    current_price + (self.trailing_stop_step * position['point'])
                )
        return None
        
    def check_profit_retracement(self, position: Dict, current_price: float, max_profit: float) -> bool:
        """Check if profit has retraced more than threshold."""
        if position['type'] == 'BUY':
            max_profit_pips = (max_profit - position['entry_price']) / position['point']
            current_profit_pips = (current_price - position['entry_price']) / position['point']
        else:  # SELL
            max_profit_pips = (position['entry_price'] - max_profit) / position['point']
            current_profit_pips = (position['entry_price'] - current_price) / position['point']
            
        if max_profit_pips <= 0:
            return False
            
        retracement = (max_profit_pips - current_profit_pips) / max_profit_pips
        return retracement >= self.profit_retracement_exit
        
    def manage_position(self, position: Dict, df: pd.DataFrame) -> Dict:
        """Manage position based on quick profit capture rules."""
        current_price = df['close'].iloc[-1]
        momentum_strength = self.calculate_momentum_strength(df)
        price_stalling = self.detect_price_stall(df)
        
        action = {
            'should_exit': False,
            'exit_size': 0.0,
            'new_stop_loss': None,
            'reason': None
        }
        
        # Check momentum slowdown
        if momentum_strength < 0.3:  # Significant momentum weakness
            action['should_exit'] = True
            action['exit_size'] = position['volume'] * self.momentum_exit_threshold
            action['reason'] = 'Momentum slowdown'
            
        # Check price stall
        elif price_stalling:
            action['should_exit'] = True
            action['exit_size'] = position['volume'] * self.stall_exit_threshold
            action['reason'] = 'Price stall'
            
        # Update trailing stop
        new_stop = self.calculate_trailing_stop(position, current_price)
        if new_stop is not None:
            action['new_stop_loss'] = new_stop
            
        # Check profit retracement
        if self.check_profit_retracement(position, current_price, position['max_profit']):
            action['should_exit'] = True
            action['exit_size'] = position['volume']  # Full exit
            action['reason'] = 'Profit retracement'
            
        return action 