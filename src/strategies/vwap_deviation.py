from .base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class VWAPDeviationStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str = '5m', exchange: str = 'binance'):
        super().__init__(symbol, timeframe, exchange)
        self.std_dev_threshold = 1.5  # Standard deviation threshold for entry
        self.position_scale = 0.25  # 25% position scaling
        self.max_hold_time = 120  # 2-hour maximum hold time
        self.targets = [0.5, 1.0, 1.5]  # Target levels in std dev units
        
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and standard deviation bands."""
        # Calculate VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate standard deviation
        df['price_vol'] = df['close'] * df['volume']
        df['cum_vol'] = df['volume'].cumsum()
        
        # Standard deviation calculation
        df['squared_diff'] = (df['close'] - df['vwap']) ** 2
        df['weighted_squared_diff'] = df['squared_diff'] * df['volume']
        df['vwap_std'] = np.sqrt(df['weighted_squared_diff'].cumsum() / df['cum_vol'])
        
        # Calculate standard deviation bands
        df['upper_band_1'] = df['vwap'] + (df['vwap_std'] * self.std_dev_threshold)
        df['lower_band_1'] = df['vwap'] - (df['vwap_std'] * self.std_dev_threshold)
        
        return df
        
    def detect_rsi_divergence(self, df: pd.DataFrame) -> str:
        """Detect RSI divergence patterns."""
        # Get last 5 candles for divergence check
        recent_df = df.tail(5)
        
        price_higher_high = recent_df['high'].iloc[-1] > recent_df['high'].iloc[-2]
        price_lower_low = recent_df['low'].iloc[-1] < recent_df['low'].iloc[-2]
        rsi_higher_high = recent_df['rsi'].iloc[-1] > recent_df['rsi'].iloc[-2]
        rsi_lower_low = recent_df['rsi'].iloc[-1] < recent_df['rsi'].iloc[-2]
        
        if price_higher_high and not rsi_higher_high:
            return 'BEARISH'
        elif price_lower_low and not rsi_lower_low:
            return 'BULLISH'
        return 'NONE'
        
    def is_valid_session_time(self) -> bool:
        """Check if current time is within valid trading session."""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Define major session overlaps (adjust according to your timezone)
        asian_european_session = 7 <= current_hour < 10
        european_us_session = 13 <= current_hour < 16
        
        return asian_european_session or european_us_session
        
    def generate_signals(self) -> Dict:
        """Generate trading signals based on VWAP deviation."""
        df = self.get_market_data(lookback=100)
        df = self.calculate_indicators(df)
        df = self.calculate_vwap(df)
        
        current_price = df['close'].iloc[-1]
        current_vwap = df['vwap'].iloc[-1]
        divergence = self.detect_rsi_divergence(df)
        
        signal = {
            'type': None,
            'entry_price': current_price,
            'stop_loss': None,
            'targets': [],
            'position_sizes': [],
            'entry_time': datetime.now()
        }
        
        # Check if price is outside standard deviation bands
        price_above_band = current_price > df['upper_band_1'].iloc[-1]
        price_below_band = current_price < df['lower_band_1'].iloc[-1]
        
        # Calculate potential position size
        std_dev_size = df['vwap_std'].iloc[-1]
        base_stop_distance = std_dev_size * 0.5  # Half std dev for stop loss
        
        if price_below_band and divergence == 'BULLISH' and self.is_valid_session_time():
            signal['type'] = 'BUY'
            signal['stop_loss'] = current_price - (base_stop_distance * 1.2)  # 120% of half std dev
            
            # Calculate targets based on VWAP and std dev
            signal['targets'] = [
                current_vwap - (std_dev_size * target) for target in self.targets
            ]
            
            # Calculate position sizes for scaling
            base_size = self.calculate_position_size(
                (current_price - signal['stop_loss']) / self.get_point_value()
            )
            signal['position_sizes'] = [base_size * self.position_scale] * 4  # 4 entries of 25% each
            
        elif price_above_band and divergence == 'BEARISH' and self.is_valid_session_time():
            signal['type'] = 'SELL'
            signal['stop_loss'] = current_price + (base_stop_distance * 1.2)
            
            # Calculate targets based on VWAP and std dev
            signal['targets'] = [
                current_vwap + (std_dev_size * target) for target in self.targets
            ]
            
            # Calculate position sizes for scaling
            base_size = self.calculate_position_size(
                (signal['stop_loss'] - current_price) / self.get_point_value()
            )
            signal['position_sizes'] = [base_size * self.position_scale] * 4
            
        return signal
        
    def execute_trade(self, signal: Dict) -> bool:
        """Execute trade based on the generated signal."""
        if not signal['type'] or not self.check_risk_limits():
            return False
            
        try:
            # Prepare order parameters
            order_type = 'market'
            side = signal['type'].lower()
            amount = signal['position_sizes'][0]
            
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
                    
                # Place scaled entry orders
                for i in range(1, 4):
                    scale_price = (
                        signal['entry_price'] * (1 + (0.001 * i))  # 0.1% price increments for scaling
                        if signal['type'] == 'BUY'
                        else signal['entry_price'] * (1 - (0.001 * i))
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
                            },
                            'timeInForce': 'GTD',
                            'expireTime': int((datetime.now() + timedelta(minutes=self.max_hold_time)).timestamp() * 1000)
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