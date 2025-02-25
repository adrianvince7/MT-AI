from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from .mt5_client import MT5Client

class MT5Adapter:
    """Adapter for MetaTrader 5 integration."""
    
    def __init__(self, login: int, password: str, server: str = "Exness-MT5Trial9"):
        """
        Initialize MT5 adapter.
        
        Args:
            login: MT5 account login number
            password: MT5 account password
            server: MT5 server name (e.g., "Exness-MT5Trial9", "Exness-MT5Real2")
        """
        self.client = MT5Client(login, password, server)
        if not self.client.connected:
            raise ConnectionError("Failed to connect to MT5")
            
    def get_market_data(self, symbol: str, timeframe: str, lookback: int = 100) -> pd.DataFrame:
        """
        Get historical market data formatted for strategy use.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            timeframe: Time frame (e.g., '1m', '5m', '1h')
            lookback: Number of candles to retrieve
        """
        # Convert generic timeframe format to MT5 format
        tf_map = {
            '1m': 'M1',
            '5m': 'M5',
            '15m': 'M15',
            '30m': 'M30',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D1',
            'D1': 'D1',
            'W1': 'W1',
            'MN1': 'MN1'
        }
        mt5_timeframe = tf_map.get(timeframe, timeframe)
        
        return self.client.get_market_data(symbol, mt5_timeframe, lookback)
        
    def execute_market_order(self, symbol: str, side: str, volume: float,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None) -> Dict:
        """Execute a market order with optional SL/TP."""
        return self.client.create_market_order(
            symbol=symbol,
            side=side,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
    def execute_limit_order(self, symbol: str, side: str, volume: float,
                          price: float, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Dict:
        """Execute a limit order with optional SL/TP."""
        return self.client.create_limit_order(
            symbol=symbol,
            side=side,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        return self.client.get_positions()
        
    def close_position(self, ticket: int) -> bool:
        """Close a specific position."""
        return self.client.close_position(ticket)
        
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """Modify an existing position's SL/TP."""
        return self.client.modify_position(
            ticket=ticket,
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
        # Get symbol info
        symbol_info = self.client.get_symbol_info(symbol)
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        current_price = tick.ask  # Use ask price for calculations
        
        # Calculate pip value
        pip_value = symbol_info['point'] * symbol_info['trade_contract_size']
        
        # Calculate position size in lots
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to nearest valid lot size
        volume_step = symbol_info['volume_step']
        position_size = round(position_size / volume_step) * volume_step
        
        # Ensure position size is within allowed limits
        position_size = max(symbol_info['volume_min'],
                          min(position_size, symbol_info['volume_max']))
        
        return position_size 