from .base_strategy import BaseStrategy
import numpy as np
from typing import Dict, Optional
import pandas as pd

class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = '5m', exchange: str = 'binance'):
        super().__init__(symbol, timeframe, exchange)
        self.initial_position = 0.3  # 30% initial position
        self.scale_in_levels = [(0.2, 5), (0.5, 8)]  # (size%, points)
        self.profit_targets = [15, 30]  # points
        self.stop_loss = 10  # points
        
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
            signal['stop_loss'] = current_price - (self.stop_loss * self.get_point_value())
            signal['targets'] = [
                current_price + (target * self.get_point_value())
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
            signal['stop_loss'] = current_price + (self.stop_loss * self.get_point_value())
            signal['targets'] = [
                current_price - (target * self.get_point_value())
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
            
        try:
            # Prepare the order parameters
            order_type = 'market'
            side = signal['type'].lower()
            amount = signal['position_sizes'][0]
            
            # Execute the initial trade
            if self.symbol.endswith('=X'):  # Forex
                # Implementation for forex broker API would go here
                # This is a placeholder as it depends on the specific broker's API
                return False
            else:  # Crypto
                order = self.exchange_api.create_order(
                    symbol=self.symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    params={
                        'stopLoss': {
                            'type': 'stop',
                            'price': signal['stop_loss']
                        },
                        'takeProfit': {
                            'type': 'limit',
                            'price': signal['targets'][0]
                        }
                    }
                )
                
                if not order:
                    return False
                    
                # Place scaled entry orders
                for i, (size_pct, points) in enumerate(self.scale_in_levels, 1):
                    scale_price = (
                        signal['entry_price'] + (points * self.get_point_value())
                        if signal['type'] == 'BUY'
                        else signal['entry_price'] - (points * self.get_point_value())
                    )
                    
                    self.exchange_api.create_order(
                        symbol=self.symbol,
                        type='limit',
                        side=side,
                        amount=signal['position_sizes'][i],
                        price=scale_price,
                        params={
                            'stopLoss': {
                                'type': 'stop',
                                'price': signal['stop_loss']
                            },
                            'takeProfit': {
                                'type': 'limit',
                                'price': signal['targets'][min(i + 1, len(signal['targets']) - 1)]
                            }
                        }
                    )
                    
                return True
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False
            
    def get_point_value(self) -> float:
        """Get the point value for the symbol."""
        if self.symbol.endswith('=X'):  # Forex
            return 0.0001 if not self.symbol.endswith('JPY=X') else 0.01
        else:  # Crypto
            # Use a percentage of current price as point value
            current_price = self.get_market_data(lookback=1)['close'].iloc[-1]
            return current_price * 0.0001  # 0.01% of price 