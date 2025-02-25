from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from .exness_client import ExnessClient

class ExnessAdapter:
    """Adapter for Exness broker integration."""
    
    def __init__(self, api_key: str, api_secret: str, account_type: str = 'standard'):
        """
        Initialize Exness adapter.
        
        Args:
            api_key: Exness API key
            api_secret: Exness API secret
            account_type: 'standard' or 'cent'
        """
        self.client = ExnessClient(api_key, api_secret, account_type)
        self.account_type = account_type
        
    def get_market_data(self, symbol: str, timeframe: str, lookback: int = 100) -> pd.DataFrame:
        """
        Get historical market data formatted for strategy use.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            timeframe: Time frame (e.g., '1m', '5m', '1h')
            lookback: Number of candles to retrieve
        """
        raw_data = self.client.get_market_data(symbol, timeframe, lookback)
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    def execute_market_order(self, symbol: str, side: str, volume: float,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None) -> Dict:
        """Execute a market order with optional SL/TP."""
        # Convert volume to lots based on account type
        if self.account_type == 'cent':
            volume = volume * 100  # Convert standard lots to cent lots
            
        return self.client.create_market_order(
            symbol=symbol,
            side=side,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
    def execute_limit_order(self, symbol: str, side: str, volume: float,
                          price: float, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          expiry: Optional[datetime] = None) -> Dict:
        """Execute a limit order with optional SL/TP and expiry."""
        # Convert volume to lots based on account type
        if self.account_type == 'cent':
            volume = volume * 100
            
        return self.client.create_limit_order(
            symbol=symbol,
            side=side,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expiry=expiry
        )
        
    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        return self.client.get_positions()
        
    def close_position(self, position_id: str) -> Dict:
        """Close a specific position."""
        return self.client.close_position(position_id)
        
    def modify_position(self, position_id: str, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Dict:
        """Modify an existing position's SL/TP."""
        return self.client.modify_position(
            position_id=position_id,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
    def get_account_info(self) -> Dict:
        """Get account information including balance and equity."""
        return self.client.get_account_info()
        
    def calculate_position_size(self, symbol: str, risk_amount: float,
                              stop_loss_pips: float) -> float:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Trading pair
            risk_amount: Amount to risk in account currency
            stop_loss_pips: Stop loss distance in pips
        """
        # Get current ticker info
        ticker = self.client.get_ticker(symbol)
        current_price = float(ticker['ask'])
        
        # Calculate pip value
        pip_value = 0.0001 if not symbol.endswith('JPY') else 0.01
        
        # Calculate position size in lots
        position_size = risk_amount / (stop_loss_pips * pip_value * current_price)
        
        # Round to nearest 0.01 lots for standard or 0.01 lots for cent
        position_size = round(position_size, 2)
        
        # Check margin requirements
        margin_info = self.client.calculate_margin(symbol, position_size)
        
        # Adjust position size if required margin is too high
        if margin_info['required_margin'] > risk_amount * 2:  # Use 50% margin
            position_size = position_size * (risk_amount * 2 / margin_info['required_margin'])
            position_size = round(position_size, 2)
            
        return position_size
        
    def convert_timeframe(self, timeframe: str) -> str:
        """Convert generic timeframe format to Exness format."""
        # Exness timeframe format mapping
        timeframe_map = {
            '1m': 'M1',
            '5m': 'M5',
            '15m': 'M15',
            '30m': 'M30',
            '1h': 'H1',
            '4h': 'H4',
            'D1': 'D1',
            'W1': 'W1',
            'MN1': 'MN1'
        }
        return timeframe_map.get(timeframe, timeframe) 