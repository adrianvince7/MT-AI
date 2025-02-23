import MetaTrader5 as mt5
from typing import Dict, List
import pandas as pd
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from .strategies.momentum_breakout import MomentumBreakoutStrategy
from .strategies.support_resistance_reversal import SupportResistanceStrategy
from .strategies.vwap_deviation import VWAPDeviationStrategy

class StrategyManager:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.strategies = {}
        self.logger = self._setup_logger()
        self.active = False
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
            
        self._initialize_strategies()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
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
                'momentum': MomentumBreakoutStrategy(symbol),
                'sr_reversal': SupportResistanceStrategy(symbol),
                'vwap': VWAPDeviationStrategy(symbol)
            }
            self.logger.info(f"Initialized strategies for {symbol}")
            
    def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for trading."""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.warning(f"Failed to get symbol info for {symbol}")
            return False
            
        # Check spread
        if symbol_info.spread > 35:  # Max 3.5 pips spread
            self.logger.info(f"Spread too high for {symbol}: {symbol_info.spread}")
            return False
            
        # Check if market is open
        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            self.logger.info(f"Market is closed for {symbol}")
            return False
            
        return True
        
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
        positions = mt5.positions_get()
        if positions is None:
            return pd.DataFrame()
            
        return pd.DataFrame(list(positions), columns=mt5.positions_get()[0]._asdict().keys())
        
    def get_strategy_performance(self) -> Dict:
        """Get performance metrics for all strategies."""
        performance = {}
        for symbol in self.symbols:
            performance[symbol] = {}
            
            # Get closed positions for the symbol
            history = mt5.history_deals_get(
                datetime.now().replace(hour=0, minute=0, second=0),
                datetime.now(),
                symbol=symbol
            )
            
            if history is None:
                continue
                
            deals_df = pd.DataFrame(list(history), columns=history[0]._asdict().keys())
            
            for strategy_name in self.strategies[symbol].keys():
                strategy_deals = deals_df[deals_df['comment'].str.contains(strategy_name, case=False)]
                
                if len(strategy_deals) > 0:
                    performance[symbol][strategy_name] = {
                        'total_trades': len(strategy_deals),
                        'profitable_trades': len(strategy_deals[strategy_deals['profit'] > 0]),
                        'total_profit': strategy_deals['profit'].sum(),
                        'win_rate': len(strategy_deals[strategy_deals['profit'] > 0]) / len(strategy_deals)
                    }
                    
        return performance 