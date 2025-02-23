import MetaTrader5 as mt5
from typing import Dict, List
from datetime import datetime
import logging
from .momentum_breakout import MomentumBreakoutStrategy
from .support_resistance_reversal import SupportResistanceStrategy
from .vwap_deviation import VWAPDeviationStrategy

class StrategyManager:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.strategies = {}
        self.logger = self._setup_logger()
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
                'momentum': MomentumBreakoutStrategy(symbol),
                'sr_reversal': SupportResistanceStrategy(symbol),
                'vwap': VWAPDeviationStrategy(symbol)
            }
            
    def check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for trading."""
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            return False
            
        # Get symbol information
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return False
            
        # Check spread
        current_spread = symbol_info.spread
        if current_spread > 35:  # Max 3.5 pips spread
            self.logger.warning(f"Spread too high for {symbol}: {current_spread/10} pips")
            return False
            
        # Check if market is open
        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            self.logger.warning(f"Market is closed for {symbol}")
            return False
            
        return True
        
    def execute_strategies(self):
        """Execute all trading strategies."""
        for symbol in self.symbols:
            if not self.check_market_conditions(symbol):
                continue
                
            self.logger.info(f"Executing strategies for {symbol}")
            
            try:
                # Generate and execute signals for each strategy
                for strategy_name, strategy in self.strategies[symbol].items():
                    signal = strategy.generate_signals()
                    
                    if signal['type']:
                        self.logger.info(f"{strategy_name} generated {signal['type']} signal for {symbol}")
                        
                        if strategy.execute_trade(signal):
                            self.logger.info(f"Successfully executed {signal['type']} trade for {symbol} using {strategy_name}")
                        else:
                            self.logger.warning(f"Failed to execute {signal['type']} trade for {symbol} using {strategy_name}")
                            
            except Exception as e:
                self.logger.error(f"Error executing strategies for {symbol}: {str(e)}")
                continue
                
    def monitor_positions(self):
        """Monitor and manage open positions."""
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            return
            
        positions = mt5.positions_get()
        if positions is None:
            return
            
        for position in positions:
            # Check if position has been open too long
            open_time = datetime.fromtimestamp(position.time)
            time_open = datetime.now() - open_time
            
            # Close positions open more than 2 hours (VWAP strategy requirement)
            if time_open.total_seconds() > 7200:  # 2 hours in seconds
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).bid,
                    "deviation": 20,
                    "magic": position.magic,
                    "comment": "Time-based exit",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(close_request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"Failed to close position {position.ticket}: {result.retcode}")
                else:
                    self.logger.info(f"Closed position {position.ticket} due to time limit")
                    
    def run(self, interval_seconds: int = 300):
        """Run the strategy manager with specified interval."""
        import time
        
        self.logger.info("Starting Strategy Manager")
        
        while True:
            try:
                self.execute_strategies()
                self.monitor_positions()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying 