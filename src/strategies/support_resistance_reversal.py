from .base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = '5m', exchange: str = 'binance'):
        super().__init__(symbol, timeframe, exchange)
        self.initial_position = 0.4  # 40% initial position
        self.scale_in_levels = [(0.3, 12)]  # (size%, points)
        self.max_hold_time = 30  # minutes
        
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels."""
        rolling_high = df['high'].rolling(window=window).max()
        rolling_low = df['low'].rolling(window=window).min()
        
        # Find areas of price congestion
        price_clusters = pd.concat([df['high'], df['low']]).value_counts()
        significant_levels = price_clusters[price_clusters > window * 0.3].index.tolist()
        
        if not significant_levels:
            return rolling_low.iloc[-1], rolling_high.iloc[-1]
            
        support = max([level for level in significant_levels if level < df['close'].iloc[-1]], default=rolling_low.iloc[-1])
        resistance = min([level for level in significant_levels if level > df['close'].iloc[-1]], default=rolling_high.iloc[-1])
        
        return support, resistance
        
    def detect_reversal_pattern(self, df: pd.DataFrame) -> str:
        """Detect candlestick reversal patterns."""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        body_size = abs(current['close'] - current['open'])
        upper_wick = current['high'] - max(current['open'], current['close'])
        lower_wick = min(current['open'], current['close']) - current['low']
        
        # Hammer pattern (bullish)
        if (lower_wick > body_size * 2 and upper_wick < body_size * 0.5 and
            current['close'] > current['open']):
            return 'BULLISH'
            
        # Shooting star pattern (bearish)
        if (upper_wick > body_size * 2 and lower_wick < body_size * 0.5 and
            current['close'] < current['open']):
            return 'BEARISH'
            
        return 'NONE'
        
    def calculate_stoch_rsi(self, df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Calculate Stochastic RSI."""
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic RSI
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k = stoch_rsi.rolling(smooth_k).mean() * 100
        d = k.rolling(smooth_d).mean()
        
        return pd.DataFrame({'k': k, 'd': d})
        
    def generate_signals(self) -> Dict:
        """Generate trading signals based on support/resistance reversals."""
        df = self.get_market_data(lookback=100)
        df = self.calculate_indicators(df)
        
        current_price = df['close'].iloc[-1]
        support, resistance = self.calculate_support_resistance(df)
        reversal_pattern = self.detect_reversal_pattern(df)
        
        # Calculate Stochastic RSI
        stoch_rsi = self.calculate_stoch_rsi(df)
        current_k = stoch_rsi['k'].iloc[-1]
        current_d = stoch_rsi['d'].iloc[-1]
        
        signal = {
            'type': None,
            'entry_price': current_price,
            'stop_loss': None,
            'targets': [],
            'position_sizes': [],
            'entry_time': datetime.now()
        }
        
        # Bullish reversal conditions
        bullish_conditions = (
            reversal_pattern == 'BULLISH' and
            abs(current_price - support) / support < 0.001 and  # Price near support
            current_k < 20 and current_k > current_d  # Oversold and crossing up
        )
        
        # Bearish reversal conditions
        bearish_conditions = (
            reversal_pattern == 'BEARISH' and
            abs(current_price - resistance) / resistance < 0.001 and  # Price near resistance
            current_k > 80 and current_k < current_d  # Overbought and crossing down
        )
        
        point_value = self.get_point_value()
        
        if bullish_conditions:
            signal['type'] = 'BUY'
            signal['stop_loss'] = support - (10 * point_value)
            signal['targets'] = [
                resistance,
                resistance + (20 * point_value)
            ]
            
            # Calculate position sizes
            base_size = self.calculate_position_size(
                (current_price - signal['stop_loss']) / point_value
            )
            signal['position_sizes'] = [
                base_size * self.initial_position,
                base_size * self.scale_in_levels[0][0]
            ]
            
        elif bearish_conditions:
            signal['type'] = 'SELL'
            signal['stop_loss'] = resistance + (10 * point_value)
            signal['targets'] = [
                support,
                support - (20 * point_value)
            ]
            
            # Calculate position sizes
            base_size = self.calculate_position_size(
                (signal['stop_loss'] - current_price) / point_value
            )
            signal['position_sizes'] = [
                base_size * self.initial_position,
                base_size * self.scale_in_levels[0][0]
            ]
            
        return signal
        
    def execute_trade(self, signal: Dict) -> bool:
        """Execute trade based on the generated signal."""
        if not signal['type'] or not self.check_risk_limits():
            return False
            
        try:
            # Check if we're within trading hours
            current_time = datetime.now()
            if current_time.hour < 2 or current_time.hour > 21:  # Example trading hours
                return False
                
            # Prepare the order parameters
            order_type = 'market'
            side = signal['type'].lower()
            amount = signal['position_sizes'][0]
            
            # Execute the trade
            if self.symbol.endswith('=X'):  # Forex
                # Implementation for forex broker API would go here
                # This is a placeholder as it depends on the specific broker's API
                return False
            else:  # Crypto
                # Initial entry
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
                    
                # Place breakout entry order
                breakout_price = (
                    signal['entry_price'] + (self.scale_in_levels[0][1] * self.get_point_value())
                    if signal['type'] == 'BUY'
                    else signal['entry_price'] - (self.scale_in_levels[0][1] * self.get_point_value())
                )
                
                expiry_time = datetime.now() + timedelta(minutes=self.max_hold_time)
                
                self.exchange_api.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side=side,
                    amount=signal['position_sizes'][1],
                    price=breakout_price,
                    params={
                        'stopLoss': {
                            'type': 'stop',
                            'price': signal['stop_loss']
                        },
                        'takeProfit': {
                            'type': 'limit',
                            'price': signal['targets'][1]
                        },
                        'timeInForce': 'GTD',
                        'expireTime': int(expiry_time.timestamp() * 1000)
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