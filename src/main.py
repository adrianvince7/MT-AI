"""Main entry point for the MT-AI trading system."""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import MetaTrader5 as mt5
from dotenv import load_dotenv

from strategies.strategy_manager import StrategyManager
from market_analysis.market_conditions import MarketConditionAnalyzer
from risk_management.performance_metrics import PerformanceManager
from config.trading_config import (
    SYMBOLS, SYSTEM_SETTINGS, LOGS_DIR
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
    """Initialize MetaTrader 5 connection."""
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
        
    # Load environment variables
    load_dotenv()
    
    # Login to MT5
    authorized = mt5.login(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER')
    )
    
    if not authorized:
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        return False
        
    logger.info("MT5 initialized successfully")
    return True

def main():
    """Main execution function."""
    try:
        # Initialize MT5
        if not initialize_mt5():
            sys.exit(1)
            
        # Initialize components
        strategy_manager = StrategyManager(SYMBOLS)
        market_analyzer = MarketConditionAnalyzer()
        performance_manager = PerformanceManager()
        
        logger.info("Starting trading system...")
        
        while True:
            try:
                # Check if we should continue trading
                can_trade, reason = performance_manager.should_continue_trading()
                if not can_trade:
                    logger.warning(f"Trading paused: {reason}")
                    continue
                    
                # Execute trading strategies
                strategy_manager.execute_strategies()
                
                # Monitor positions
                strategy_manager.monitor_positions()
                
                # Sleep for update interval
                mt5.sleep(SYSTEM_SETTINGS['update_interval'] * 1000)  # Convert to milliseconds
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                mt5.sleep(SYSTEM_SETTINGS['retry_delay'] * 1000)
                
    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
    finally:
        mt5.shutdown()
        logger.info("Trading system stopped")

if __name__ == "__main__":
    main() 