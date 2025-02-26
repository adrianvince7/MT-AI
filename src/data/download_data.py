import os
import sys
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import time

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables
load_dotenv()

from config.trading_config import FOREX_SYMBOLS, TIMEFRAMES

def check_mt5_terminal():
    """Check if MT5 terminal is running and properly connected."""
    # Check if MT5 is installed
    if not mt5.version():
        print("MetaTrader 5 package is not installed")
        return False
        
    print(f"MetaTrader5 package version: {mt5.version()}")
    
    # Initialize MT5
    if not mt5.initialize():
        print("MetaTrader 5 terminal not found. Please start the terminal first.")
        return False
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("Failed to get terminal info")
        return False
    
    print("\nTerminal Info:")
    print(f"Connected: {terminal_info.connected}")
    print(f"Trade Allowed: {terminal_info.trade_allowed}")
    print(f"Path: {terminal_info.path}")
    print(f"Data Path: {terminal_info.data_path}")
    
    return terminal_info.connected

def download_historical_data():
    """Download historical data for all symbols and timeframes."""
    # Check terminal first
    if not check_mt5_terminal():
        print("Please start MetaTrader 5 terminal and try again")
        return False
    
    # Print MT5 credentials for debugging
    print(f"\nMT5 Login: {os.getenv('MT5_LOGIN')}")
    print(f"MT5 Server: {os.getenv('MT5_SERVER')}")
    print(f"MT5 Password: {'*' * len(os.getenv('MT5_PASSWORD', ''))}")
    
    # Initialize MT5 with account credentials
    if not mt5.initialize(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER')
    ):
        error = mt5.last_error()
        print(f"Failed to initialize MT5. Error code: {error[0]}, Description: {error[1]}")
        return False
    
    # Wait for connection to stabilize
    print("\nWaiting for connection to stabilize...")
    time.sleep(5)
    
    # Check account info
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return False
    
    print("\nAccount Info:")
    print(f"Login: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Balance: {account_info.balance}")
    print(f"Leverage: {account_info.leverage}")
    
    # Create data directories if they don't exist
    base_dir = Path(__file__).parent.parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days of data
    
    timeframe_map = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1
    }
    
    for symbol in FOREX_SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\nDownloading {symbol} {timeframe} data...")
            
            try:
                # Get historical data
                rates = mt5.copy_rates_range(
                    symbol,
                    timeframe_map[timeframe],
                    start_date,
                    end_date
                )
                
                if rates is None:
                    error = mt5.last_error()
                    print(f"Failed to download {symbol} {timeframe} data. Error: {error[1]}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Save to file
                filename = f"{symbol}_{timeframe}.csv"
                df.to_csv(raw_data_dir / filename, index=False)
                print(f"Saved {filename}")
                
            except Exception as e:
                print(f"Error downloading {symbol} {timeframe} data: {str(e)}")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    download_historical_data() 