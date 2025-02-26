"""Main entry point for the MT-AI trading system."""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import MetaTrader5 as mt5
from dotenv import load_dotenv
import time

from strategies.strategy_manager import StrategyManager
from market_analysis.market_conditions import MarketConditionAnalyzer
from risk_management.performance_metrics import PerformanceManager
from config.trading_config import (
    FOREX_SYMBOLS, MT5_SETTINGS, 
    TIMEFRAMES, SYSTEM_SETTINGS, LOGS_DIR
)

# Setup logging
log_file = LOGS_DIR / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=SYSTEM_SETTINGS['log_level'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_mt5():
    """Initialize MT5 connection."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize MT5
        if not mt5.initialize(
            path=MT5_SETTINGS['terminal_path'],
            login=MT5_SETTINGS['login'],
            password=MT5_SETTINGS['password'],
            server=MT5_SETTINGS['server']
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        # Test connection by getting account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to connect to MT5 account")
            return False
            
        logger.info(f"Successfully connected to MT5 - Account: {account_info.login}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize MT5: {str(e)}")
        return False
        
def main():
    """Main function to run the trading system."""
    if not initialize_mt5():
        sys.exit(1)
        
    try:
        # Initialize strategy manager with forex symbols
        strategy_manager = StrategyManager(
            symbols=FOREX_SYMBOLS,
            broker="MT5",
            timeframe=TIMEFRAMES[0]  # Use first timeframe as default
        )
        
        # Initialize market analyzer
        market_analyzer = MarketConditionAnalyzer()
        
        # Initialize performance manager
        performance_manager = PerformanceManager()
        
        # Start the strategy manager
        strategy_manager.start()
        
        # Keep the main thread running
        try:
            while True:
                # Monitor system performance
                performance = strategy_manager.get_strategy_performance()
                performance_manager.update_metrics(performance)
                
                # Log performance metrics
                logger.info("Current Performance Metrics:")
                for symbol, metrics in performance.items():
                    for strategy, stats in metrics.items():
                        logger.info(f"{symbol} - {strategy}: Win Rate: {stats['win_rate']:.2%}, "
                                  f"Total Profit: {stats['total_profit']:.2f}")
                                  
                # Sleep for a while before next update
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Shutting down trading system...")
            strategy_manager.stop()
            mt5.shutdown()
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        mt5.shutdown()
        sys.exit(1)
        
if __name__ == "__main__":
    main() 