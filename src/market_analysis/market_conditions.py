import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from typing import Tuple

class MarketConditionAnalyzer:
    def __init__(self):
        self.min_atr_pips = 12
        self.max_atr_pips = 40
        self.opening_range_delay = 15  # minutes
        self.news_buffer = 30  # minutes
        self.session_ranges = {
            'asia': {'start': 0, 'end': 9},    # UTC hours
            'europe': {'start': 7, 'end': 16},
            'us': {'start': 12, 'end': 21}
        }
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift()
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        atr = tr.rolling(period).mean().iloc[-1]
        return atr
        
    def is_volatility_suitable(self, symbol: str, timeframe: mt5.TIMEFRAME = mt5.TIMEFRAME_M5) -> bool:
        """Check if current volatility is within acceptable range."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)
        if rates is None:
            return False
            
        df = pd.DataFrame(rates)
        atr = self.calculate_atr(df)
        
        # Convert ATR to pips
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False
            
        atr_pips = atr / symbol_info.point
        return self.min_atr_pips <= atr_pips <= self.max_atr_pips
        
    def get_session_overlap(self) -> List[str]:
        """Identify current session overlap."""
        current_hour = datetime.utcnow().hour
        active_sessions = []
        
        for session, hours in self.session_ranges.items():
            if hours['start'] <= current_hour < hours['end']:
                active_sessions.append(session)
                
        return active_sessions
        
    def is_valid_trading_time(self) -> Tuple[bool, str]:
        """Check if current time is suitable for trading."""
        active_sessions = self.get_session_overlap()
        
        if len(active_sessions) < 2:
            return False, "No session overlap"
            
        current_time = datetime.utcnow()
        current_minute = current_time.minute
        
        # Check for opening range delay
        for session in active_sessions:
            session_start = self.session_ranges[session]['start']
            if current_time.hour == session_start and current_minute < self.opening_range_delay:
                return False, f"Within opening range delay of {session} session"
                
        return True, "Valid trading time"
        
    def calculate_opening_range(self, symbol: str, session: str) -> Dict:
        """Calculate session opening range."""
        session_start = self.session_ranges[session]['start']
        
        # Get data from session start
        start_time = datetime.utcnow().replace(hour=session_start, minute=0, second=0, microsecond=0)
        if datetime.utcnow().hour < session_start:
            start_time -= timedelta(days=1)
            
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, start_time, 
                                  self.opening_range_delay // 5 + 1)
        
        if rates is None or len(rates) == 0:
            return {'high': None, 'low': None}
            
        df = pd.DataFrame(rates)
        return {
            'high': df['high'].max(),
            'low': df['low'].min()
        }
        
    async def check_economic_calendar(self) -> bool:
        """Check for high-impact news events."""
        try:
            # Note: You would need to implement this with your preferred news data provider
            # This is a placeholder showing the structure
            current_time = datetime.utcnow()
            buffer_start = current_time - timedelta(minutes=self.news_buffer)
            buffer_end = current_time + timedelta(minutes=self.news_buffer)
            
            # Example API call to news provider
            # response = await self.news_api.get_events(
            #     start_time=buffer_start,
            #     end_time=buffer_end,
            #     impact='high'
            # )
            
            # Return True if no high-impact news events
            return True
            
        except Exception as e:
            print(f"Error checking economic calendar: {str(e)}")
            return True  # Default to allowing trading if calendar check fails
            
    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze overall market conditions."""
        conditions = {
            'suitable_for_trading': False,
            'volatility_suitable': False,
            'session_overlap': [],
            'valid_time': False,
            'time_message': "",
            'opening_ranges': {},
            'warnings': []
        }
        
        # Check volatility
        conditions['volatility_suitable'] = self.is_volatility_suitable(symbol)
        if not conditions['volatility_suitable']:
            conditions['warnings'].append("Volatility outside acceptable range")
            
        # Check sessions
        conditions['session_overlap'] = self.get_session_overlap()
        conditions['valid_time'], conditions['time_message'] = self.is_valid_trading_time()
        
        if len(conditions['session_overlap']) >= 2:
            # Calculate opening ranges for active sessions
            for session in conditions['session_overlap']:
                conditions['opening_ranges'][session] = self.calculate_opening_range(symbol, session)
                
        # Determine overall suitability
        conditions['suitable_for_trading'] = (
            conditions['volatility_suitable'] and
            conditions['valid_time'] and
            len(conditions['session_overlap']) >= 2
        )
        
        return conditions 