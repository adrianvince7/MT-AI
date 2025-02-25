"""Main entry point for the MT-AI trading system."""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import ccxt
from dotenv import load_dotenv
import time

from strategies.strategy_manager import StrategyManager
from market_analysis.market_conditions import MarketConditionAnalyzer
from risk_management.performance_metrics import PerformanceManager
from config.trading_config import (
    CRYPTO_SYMBOLS, FOREX_SYMBOLS, EXCHANGE_SETTINGS,
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

def initialize_exchange():
    """Initialize exchange connection."""
    try:
        # Load environment variables
        load_dotenv()
        
        if EXCHANGE_SETTINGS['test_mode']:
            exchange_class = getattr(ccxt, f"{EXCHANGE_SETTINGS['name']}{'Test' if EXCHANGE_SETTINGS['test_mode'] else ''}")
        else:
            exchange_class = getattr(ccxt, EXCHANGE_SETTINGS['name'])
            
        exchange = exchange_class({
            'apiKey': EXCHANGE_SETTINGS['api_key'],
            'secret': EXCHANGE_SETTINGS['secret_key'],
            'enableRateLimit': True
        })
        
        if EXCHANGE_SETTINGS['test_mode']:
            exchange.set_sandbox_mode(True)
            
        # Test connection
        exchange.fetch_balance()
        logger.info(f"Successfully connected to {EXCHANGE_SETTINGS['name']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize exchange: {str(e)}")
        return False
        
def main():
    """Main function to run the trading system."""
    if not initialize_exchange():
        sys.exit(1)
        
    try:
        # Initialize strategy manager with both crypto and forex symbols
        strategy_manager = StrategyManager(
            symbols=CRYPTO_SYMBOLS + FOREX_SYMBOLS,
            exchange=EXCHANGE_SETTINGS['name'],
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
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 