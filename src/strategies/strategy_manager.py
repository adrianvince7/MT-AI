import ccxt
import yfinance as yf
from typing import Dict, List
from datetime import datetime
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

from .momentum_breakout import MomentumBreakoutStrategy
from .support_resistance_reversal import SupportResistanceStrategy
from .vwap_deviation import VWAPDeviationStrategy

class StrategyManager:
    def __init__(self, symbols: List[str], exchange: str = 'binance', timeframe: str = '5m'):
        """
        Initialize the strategy manager.
        
        Args:
            symbols: List of trading symbols (e.g. ['BTC/USDT', 'ETH/USDT'] for crypto, ['EURUSD=X', 'GBPUSD=X'] for forex)
            exchange: Exchange name for crypto trading (ignored for forex)
            timeframe: Trading timeframe (e.g. '1m', '5m', '1h', '1d')
        """
        self.symbols = symbols
        self.exchange = exchange
        self.timeframe = timeframe
        self.strategies = {}
        self.logger = self._setup_logger()
        self.active = False
        
        # Initialize exchange connection for crypto
        if not any(s.endswith('=X') for s in symbols):  # All crypto symbols
            self.exchange_api = getattr(ccxt, exchange)()
        else:
            self.exchange_api = None
            
        self._initialize_strategies()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('StrategyManager')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('trading.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
        
    def _initialize_strategies(self):
        """Initialize trading strategies for each symbol."""
        for symbol in self.symbols:
            self.strategies[symbol] = {
                'momentum': MomentumBreakoutStrategy(symbol, self.timeframe, self.exchange),
                'sr_reversal': SupportResistanceStrategy(symbol, self.timeframe, self.exchange),
                'vwap': VWAPDeviationStrategy(symbol, self.timeframe, self.exchange)
            }
            self.logger.info(f"Initialized strategies for {symbol}")
            
    def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for trading."""
        try:
            if symbol.endswith('=X'):  # Forex
                # Get current market data from yfinance
                ticker = yf.Ticker(symbol)
                current_data = ticker.history(period='1d')
                if current_data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    return False
                    
                # Basic check for forex market hours
                current_hour = datetime.now().hour
                if current_hour < 2 or current_hour > 21:  # Example trading hours
                    self.logger.info(f"Outside trading hours for {symbol}")
                    return False
                    
            else:  # Crypto
                # Get market data from exchange
                ticker = self.exchange_api.fetch_ticker(symbol)
                if not ticker:
                    self.logger.warning(f"Failed to get ticker for {symbol}")
                    return False
                    
                # Check if trading is active
                market = self.exchange_api.market(symbol)
                if not market['active']:
                    self.logger.info(f"Market inactive for {symbol}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions for {symbol}: {str(e)}")
            return False
            
    def _process_symbol(self, symbol: str):
        """Process all strategies for a single symbol."""
        if not self._check_market_conditions(symbol):
            return
            
        for strategy_name, strategy in self.strategies[symbol].items():
            try:
                # Generate signals
                signal = strategy.generate_signals()
                
                if signal['type']:
                    self.logger.info(f"Signal generated for {symbol} using {strategy_name}: {signal['type']}")
                    
                    # Execute trade
                    if strategy.execute_trade(signal):
                        self.logger.info(f"Trade executed for {symbol} using {strategy_name}")
                    else:
                        self.logger.info(f"Trade execution failed for {symbol} using {strategy_name}")
                        
            except Exception as e:
                self.logger.error(f"Error processing {strategy_name} for {symbol}: {str(e)}")
                
    def start(self):
        """Start the strategy manager."""
        self.active = True
        self.logger.info("Strategy Manager started")
        
        with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
            while self.active:
                try:
                    # Process all symbols concurrently
                    executor.map(self._process_symbol, self.symbols)
                    
                    # Wait for next iteration
                    time.sleep(5)  # 5-second delay between iterations
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    
    def stop(self):
        """Stop the strategy manager."""
        self.active = False
        self.logger.info("Strategy Manager stopped")
        
    def get_active_positions(self) -> pd.DataFrame:
        """Get all active positions."""
        positions = []
        
        for symbol in self.symbols:
            try:
                if symbol.endswith('=X'):  # Forex
                    # Implementation depends on your forex broker's API
                    pass
                else:  # Crypto
                    positions.extend(self.exchange_api.fetch_positions(symbol))
            except Exception as e:
                self.logger.error(f"Error fetching positions for {symbol}: {str(e)}")
                
        return pd.DataFrame(positions)
        
    def get_strategy_performance(self) -> Dict:
        """Get performance metrics for all strategies."""
        performance = {}
        
        for symbol in self.symbols:
            performance[symbol] = {}
            
            try:
                if symbol.endswith('=X'):  # Forex
                    # Implementation depends on your forex broker's API
                    pass
                else:  # Crypto
                    # Get closed orders from exchange
                    trades = self.exchange_api.fetch_my_trades(symbol)
                    trades_df = pd.DataFrame(trades)
                    
                    for strategy_name in self.strategies[symbol].keys():
                        strategy_trades = trades_df[trades_df['info'].str.contains(strategy_name, case=False)]
                        
                        if len(strategy_trades) > 0:
                            performance[symbol][strategy_name] = {
                                'total_trades': len(strategy_trades),
                                'profitable_trades': len(strategy_trades[strategy_trades['profit'] > 0]),
                                'total_profit': strategy_trades['profit'].sum(),
                                'win_rate': len(strategy_trades[strategy_trades['profit'] > 0]) / len(strategy_trades)
                            }
                            
            except Exception as e:
                self.logger.error(f"Error getting performance for {symbol}: {str(e)}")
                
        return performance 