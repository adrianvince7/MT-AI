import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

class MT5Client:
    """Client for interacting with MetaTrader 5 terminal."""
    
    def __init__(self, login: int, password: str, server: str = "Exness-MT5Trial9"):
        """
        Initialize MT5 client.
        
        Args:
            login: MT5 account login number
            password: MT5 account password
            server: MT5 server name (e.g., "Exness-MT5Trial9", "Exness-MT5Real2")
        """
        self.login = login
        self.password = password
        self.server = server
        self.logger = logging.getLogger(__name__)
        self.connected = self._initialize_mt5()
        
    def _initialize_mt5(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal."""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
                
            # Login to the account
            if not mt5.login(login=self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                self.logger.error(f"MT5 login failed: {error}")
                return False
                
            self.logger.info(f"Successfully connected to MT5 account {self.login}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False
            
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to get account info")
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage,
            'currency': account_info.currency
        }
        
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'swap': pos.swap,
                'time': datetime.fromtimestamp(pos.time)
            }
            for pos in positions
        ]
        
    def create_market_order(self, symbol: str, side: str, volume: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Dict:
        """
        Create a market order.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            side: 'buy' or 'sell'
            volume: Trade volume in lots
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).ask if side.lower() == 'buy' else mt5.symbol_info_tick(symbol).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "market order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit
            
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise ValueError(f"Order failed: {result.comment}")
            
        return {
            'ticket': result.order,
            'volume': volume,
            'price': result.price,
            'side': side
        }
        
    def create_limit_order(self, symbol: str, side: str, volume: float,
                         price: float, stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            side: 'buy' or 'sell'
            volume: Trade volume in lots
            price: Limit price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY_LIMIT if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL_LIMIT,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "limit order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit
            
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise ValueError(f"Order failed: {result.comment}")
            
        return {
            'ticket': result.order,
            'volume': volume,
            'price': price,
            'side': side
        }
        
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """Modify an existing position's stop loss or take profit."""
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise ValueError(f"Position {ticket} not found")
            
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": stop_loss if stop_loss is not None else position.sl,
            "tp": take_profit if take_profit is not None else position.tp
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
        
    def close_position(self, ticket: int) -> bool:
        """Close a specific position."""
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise ValueError(f"Position {ticket} not found")
            
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
        
    def get_market_data(self, symbol: str, timeframe: str,
                       count: int = 100) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            timeframe: Time frame (e.g., 'M1', 'M5', 'H1')
            count: Number of candles to retrieve
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        # Convert timeframe string to MT5 timeframe
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        tf = timeframe_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
            
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            raise ValueError(f"Failed to get market data for {symbol}")
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
        
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information."""
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol} not found")
            
        return {
            'symbol': info.name,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'trade_tick_size': info.trade_tick_size,
            'trade_tick_value': info.trade_tick_value,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        }
        
    def __del__(self):
        """Cleanup MT5 connection on object destruction."""
        mt5.shutdown() 