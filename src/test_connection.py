import MetaTrader5 as mt5
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mt5_connection(login: int, password: str, server: str) -> bool:
    """Test connection to MT5 terminal and display account information."""
    try:
        # Initialize MT5
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
            
        # Login to the account
        if not mt5.login(login=login, password=password, server=server):
            error = mt5.last_error()
            logger.error(f"MT5 login failed: {error}")
            return False
            
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return False
            
        # Display account information
        logger.info("\n=== Account Information ===")
        logger.info(f"Login: {account_info.login}")
        logger.info(f"Server: {account_info.server}")
        logger.info(f"Balance: {account_info.balance:.2f} {account_info.currency}")
        logger.info(f"Equity: {account_info.equity:.2f} {account_info.currency}")
        logger.info(f"Free Margin: {account_info.margin_free:.2f} {account_info.currency}")
        logger.info(f"Leverage: 1:{account_info.leverage}")
        
        # Test market data access
        symbol = "EURUSD"
        logger.info(f"\n=== Testing Market Data ({symbol}) ===")
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get {symbol} info")
        else:
            logger.info(f"Spread: {symbol_info.spread} points")
            logger.info(f"Tick Size: {symbol_info.trade_tick_size}")
            logger.info(f"Min Lot: {symbol_info.volume_min:.2f}")
            logger.info(f"Max Lot: {symbol_info.volume_max:.2f}")
            
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get {symbol} price")
        else:
            logger.info(f"Current Bid: {tick.bid:.5f}")
            logger.info(f"Current Ask: {tick.ask:.5f}")
            
        logger.info("\nConnection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during connection test: {str(e)}")
        return False
        
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    # Your credentials
    LOGIN = 208619926
    PASSWORD = "TestSubject1!"
    SERVER = "Exness-MT5Trial9"
    
    test_mt5_connection(LOGIN, PASSWORD, SERVER) 