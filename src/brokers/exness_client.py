import requests
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
from urllib.parse import urlencode

class ExnessClient:
    """Client for interacting with Exness API."""
    
    def __init__(self, api_key: str, api_secret: str, account_type: str = 'standard'):
        """
        Initialize Exness API client.
        
        Args:
            api_key: Exness API key
            api_secret: Exness API secret
            account_type: 'standard' or 'cent'
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_type = account_type.lower()
        self.base_url = "https://api-demo.exness.com/v1"  # Change to live URL for production
        self.logger = logging.getLogger(__name__)
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate signature for API request."""
        sorted_params = dict(sorted(params.items()))
        message = urlencode(sorted_params)
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Exness API."""
        if params is None:
            params = {}
            
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Generate signature
        signature = self._generate_signature(params)
        headers = {
            'X-API-KEY': self.api_key,
            'X-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise
            
    def get_account_info(self) -> Dict:
        """Get account information."""
        return self._make_request('GET', '/account')
        
    def get_balance(self) -> float:
        """Get account balance."""
        account_info = self.get_account_info()
        return float(account_info['balance'])
        
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        return self._make_request('GET', '/positions')
        
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
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'volume': volume
        }
        
        if stop_loss:
            params['stopLoss'] = stop_loss
        if take_profit:
            params['takeProfit'] = take_profit
            
        return self._make_request('POST', '/order/create', params)
        
    def create_limit_order(self, symbol: str, side: str, volume: float,
                          price: float, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          expiry: Optional[datetime] = None) -> Dict:
        """
        Create a limit order.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            side: 'buy' or 'sell'
            volume: Trade volume in lots
            price: Limit price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            expiry: Optional expiry time
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'volume': volume,
            'price': price
        }
        
        if stop_loss:
            params['stopLoss'] = stop_loss
        if take_profit:
            params['takeProfit'] = take_profit
        if expiry:
            params['expiry'] = int(expiry.timestamp() * 1000)
            
        return self._make_request('POST', '/order/create', params)
        
    def modify_position(self, position_id: str, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Dict:
        """Modify an existing position's stop loss or take profit."""
        params = {
            'positionId': position_id
        }
        
        if stop_loss:
            params['stopLoss'] = stop_loss
        if take_profit:
            params['takeProfit'] = take_profit
            
        return self._make_request('POST', '/position/modify', params)
        
    def close_position(self, position_id: str) -> Dict:
        """Close a specific position."""
        params = {'positionId': position_id}
        return self._make_request('POST', '/position/close', params)
        
    def get_market_data(self, symbol: str, timeframe: str,
                       count: int = 100) -> List[Dict]:
        """
        Get historical market data.
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            timeframe: Time frame (e.g., '1m', '5m', '1h')
            count: Number of candles to retrieve
        """
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'count': count
        }
        return self._make_request('GET', '/market/candles', params)
        
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker information for a symbol."""
        params = {'symbol': symbol}
        return self._make_request('GET', '/market/ticker', params)
        
    def calculate_margin(self, symbol: str, volume: float) -> Dict:
        """Calculate required margin for a trade."""
        params = {
            'symbol': symbol,
            'volume': volume
        }
        return self._make_request('GET', '/calculator/margin', params)
        
    def get_trading_history(self, start_time: datetime,
                          end_time: Optional[datetime] = None,
                          limit: int = 100) -> List[Dict]:
        """Get trading history for the specified period."""
        params = {
            'startTime': int(start_time.timestamp() * 1000),
            'limit': limit
        }
        
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
            
        return self._make_request('GET', '/history/trades', params) 