from .base_strategy import BaseStrategy
import MetaTrader5 as mt5
import numpy as np
from typing import Dict, Optional
import pandas as pd

class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: mt5.TIMEFRAME = mt5.TIMEFRAME_M5):
        super().__init__(symbol, timeframe)
        self.initial_position = 0.3  # 30% initial position
        self.scale_in_levels = [(0.2, 5), (0.5, 8)]  # (size%, pips)
        self.profit_targets = [15, 30]  # pips
        self.stop_loss = 10  # pips
        
    def generate_signals(self) -> Dict:
        """Generate trading signals based on momentum breakout conditions."""
        df = self.get_market_data(lookback=100)
        df = self.calculate_indicators(df)
        
        current_price = df['close'].iloc[-1]
        current_volume_ratio = df['volume_ratio'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        
        # Check EMA alignments
        ema_aligned_bullish = (df['ema5'].iloc[-1] > df['ema8'].iloc[-1] > df['ema13'].iloc[-1])
        ema_aligned_bearish = (df['ema5'].iloc[-1] < df['ema8'].iloc[-1] < df['ema13'].iloc[-1])
        
        # Check RSI conditions
        rsi_bullish = current_rsi > 50 and df['rsi'].iloc[-2] <= 50
        rsi_bearish = current_rsi < 50 and df['rsi'].iloc[-2] >= 50
        
        # Volume confirmation
        volume_confirmed = current_volume_ratio > 1.2
        
        signal = {
            'type': None,
            'entry_price': current_price,
            'stop_loss': None,
            'targets': [],
            'position_sizes': []
        }
        
        if ema_aligned_bullish and rsi_bullish and volume_confirmed:
            signal['type'] = 'BUY'
            signal['stop_loss'] = current_price - (self.stop_loss * mt5.symbol_info(self.symbol).point)
            signal['targets'] = [
                current_price + (target * mt5.symbol_info(self.symbol).point)
                for target in self.profit_targets
            ]
            # Calculate position sizes for initial entry and scaling
            base_size = self.calculate_position_size(self.stop_loss)
            signal['position_sizes'] = [
                base_size * self.initial_position,
                base_size * self.scale_in_levels[0][0],
                base_size * self.scale_in_levels[1][0]
            ]
            
        elif ema_aligned_bearish and rsi_bearish and volume_confirmed:
            signal['type'] = 'SELL'
            signal['stop_loss'] = current_price + (self.stop_loss * mt5.symbol_info(self.symbol).point)
            signal['targets'] = [
                current_price - (target * mt5.symbol_info(self.symbol).point)
                for target in self.profit_targets
            ]
            # Calculate position sizes for initial entry and scaling
            base_size = self.calculate_position_size(self.stop_loss)
            signal['position_sizes'] = [
                base_size * self.initial_position,
                base_size * self.scale_in_levels[0][0],
                base_size * self.scale_in_levels[1][0]
            ]
            
        return signal
        
    def execute_trade(self, signal: Dict) -> bool:
        """Execute trade based on the generated signal."""
        if not signal['type'] or not self.check_risk_limits():
            return False
            
        # Prepare the trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": signal['position_sizes'][0],  # Initial position size
            "type": mt5.ORDER_TYPE_BUY if signal['type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": signal['entry_price'],
            "sl": signal['stop_loss'],
            "tp": signal['targets'][0],  # First target
            "deviation": 20,
            "magic": 234000,
            "comment": "Momentum Breakout",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send the trade
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode={result.retcode}")
            return False
            
        # Place scaled entry orders
        for i, (size_pct, pips) in enumerate(self.scale_in_levels, 1):
            scale_price = (signal['entry_price'] + 
                         (pips * mt5.symbol_info(self.symbol).point) if signal['type'] == 'BUY'
                         else signal['entry_price'] - 
                         (pips * mt5.symbol_info(self.symbol).point))
            
            scale_request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": self.symbol,
                "volume": signal['position_sizes'][i],
                "type": mt5.ORDER_TYPE_BUY_LIMIT if signal['type'] == 'BUY' 
                        else mt5.ORDER_TYPE_SELL_LIMIT,
                "price": scale_price,
                "sl": signal['stop_loss'],
                "tp": signal['targets'][1],  # Second target for scaled positions
                "deviation": 20,
                "magic": 234000 + i,
                "comment": f"Momentum Breakout Scale {i}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            scale_result = mt5.order_send(scale_request)
            if scale_result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Scale order {i} failed, retcode={scale_result.retcode}")
                
        return True 