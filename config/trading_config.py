"""Trading configuration parameters."""

import os
from pathlib import Path

# System paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Trading parameters
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # Crypto trading pairs
FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]   # Forex symbols
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]           # Available timeframes
STRATEGIES = ["momentum", "sr_reversal", "vwap"]

# Exchange settings
EXCHANGE_SETTINGS = {
    'name': 'binance',
    'test_mode': True,  # Use testnet for development
    'api_key': os.getenv('EXCHANGE_API_KEY', ''),
    'secret_key': os.getenv('EXCHANGE_SECRET_KEY', ''),
}

# Strategy configurations
MOMENTUM_CONFIG = {
    'lookback_period': 20,
    'volatility_period': 14,
    'breakout_threshold': 1.5,
    'initial_position': 0.3,
    'scale_in_levels': [(0.2, 5), (0.5, 8)],
    'profit_targets': [15, 30],
    'stop_loss': 10
}

SR_CONFIG = {
    'sr_period': 50,
    'zone_threshold': 0.001,
    'reversal_threshold': 2.0,
    'initial_position': 0.4,
    'scale_in_levels': [(0.3, 12)],
    'max_hold_time': 30
}

VWAP_CONFIG = {
    'vwap_period': 24,
    'deviation_threshold': 2.0,
    'volume_factor': 1.5,
    'std_dev_threshold': 1.5,
    'position_scale': 0.25,
    'max_hold_time': 120,
    'targets': [0.5, 1.0, 1.5]
}

# Risk management settings
RISK_SETTINGS = {
    'max_position_size': 0.02,    # 2% per trade
    'max_total_exposure': 0.06,   # 6% total
    'daily_loss_limit': 0.05,     # 5% daily max loss
    'trailing_stop': 5,           # 5 pip trailing stop
    'profit_retracement': 0.4,    # 40% profit retracement
    'consecutive_losses': 3        # Max consecutive losses
}

# Market condition parameters
MARKET_CONDITIONS = {
    'min_volatility': 0.002,      # Minimum price volatility (0.2%)
    'max_volatility': 0.02,       # Maximum price volatility (2%)
    'min_volume': 1000000,        # Minimum 24h volume in quote currency
    'max_spread': 0.002,          # Maximum spread (0.2%)
    'min_liquidity': 500000,      # Minimum order book liquidity
    'news_buffer': 30,            # Minutes before/after news events
}

# Data collection settings
DATA_SETTINGS = {
    'historical_days': 30,        # Days of historical data to maintain
    'update_interval': 60,        # Seconds between data updates
    'max_retries': 3,            # Maximum API retry attempts
    'timeout': 30,               # API timeout in seconds
}

# ML model parameters
ML_CONFIG = {
    'confidence_threshold': 0.7,
    'ensemble_agreement': 0.8,
    'sequence_length': 10,
    'feature_window': 100,
    'min_training_size': 1000,
    'validation_size': 0.2,
    'max_models': 5
}

# Backtesting parameters
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.0001,         # 0.01%
    'slippage': 0.0001,          # 0.01%
    'use_ml': True,
    'in_sample_size': 1000,
    'out_sample_size': 500,
    'step_size': 250
}

# Trading sessions (UTC)
TRADING_SESSIONS = {
    'asia': {'start': 0, 'end': 9},
    'europe': {'start': 7, 'end': 16},
    'us': {'start': 12, 'end': 21}
}

# System settings
SYSTEM_SETTINGS = {
    'log_level': 'INFO',
    'update_interval': 300,       # 5 minutes
    'max_retries': 3,
    'retry_delay': 60,           # 1 minute
    'heartbeat_interval': 30,    # 30 seconds
    'cleanup_interval': 86400    # 24 hours
} 