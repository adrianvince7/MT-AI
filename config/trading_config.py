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
FOREX_SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm"]   # Updated with 'm' suffix
TIMEFRAMES = ["M5", "M15", "H1"]                     # Available timeframes
STRATEGIES = ["momentum", "sr_reversal", "vwap"]

# Strategy configurations
MOMENTUM_CONFIG = {
    'lookback_period': 10,          # Reduced from 20
    'volatility_period': 7,         # Reduced from 14
    'breakout_threshold': 1.2,      # Reduced from 1.5
    'initial_position': 0.1,        # Reduced from 0.3
    'scale_in_levels': [(0.1, 3)],  # Reduced scale-in
    'profit_targets': [5, 10],      # Reduced from [15, 30]
    'stop_loss': 5                  # Reduced from 10
}

SR_CONFIG = {
    'sr_period': 25,                # Reduced from 50
    'zone_threshold': 0.0005,       # Reduced from 0.001
    'reversal_threshold': 1.5,      # Reduced from 2.0
    'initial_position': 0.1,        # Reduced from 0.4
    'scale_in_levels': [(0.1, 5)],  # Reduced scale-in
    'max_hold_time': 15             # Reduced from 30
}

VWAP_CONFIG = {
    'vwap_period': 12,              # Reduced from 24
    'deviation_threshold': 1.5,      # Reduced from 2.0
    'volume_factor': 1.2,           # Reduced from 1.5
    'std_dev_threshold': 1.2,       # Reduced from 1.5
    'position_scale': 0.1,          # Reduced from 0.25
    'max_hold_time': 60,            # Reduced from 120
    'targets': [0.3, 0.5, 0.8]      # Reduced from [0.5, 1.0, 1.5]
}

# Risk management settings
RISK_SETTINGS = {
    'max_position_size': 0.01,      # Reduced to 1% per trade
    'max_total_exposure': 0.02,     # Reduced to 2% total
    'daily_loss_limit': 0.03,       # Reduced to 3% daily max loss
    'trailing_stop': 3,             # Reduced from 5 pips
    'profit_retracement': 0.3,      # Reduced from 0.4
    'consecutive_losses': 2          # Reduced from 3
}

# Market condition parameters
MARKET_CONDITIONS = {
    'min_volatility': 0.001,        # Reduced from 0.002
    'max_volatility': 0.015,        # Reduced from 0.02
    'min_volume': 500000,           # Reduced from 1000000
    'max_spread': 0.0015,           # Reduced from 0.002
    'min_liquidity': 250000,        # Reduced from 500000
    'news_buffer': 45               # Increased from 30 for more safety
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
    'confidence_threshold': 0.90,    # Increased to 0.90 for maximum precision on small account
    'ensemble_agreement': 0.95,      # Increased to 0.95 for near-unanimous consensus
    'sequence_length': 3,            # Reduced to 3 for faster reactions (considering M5 timeframe)
    'feature_window': 30,            # Reduced to 30 for more responsive adaptation
    'min_training_size': 300,        # Reduced to 300 (about 25 hours of M5 data)
    'validation_size': 0.15,         # Reduced to save more data for training
    'max_models': 2,                 # Reduced to 2 models for faster processing
    'prediction_threshold': 0.85,    # Increased for stricter entry criteria
    'feature_importance_threshold': 0.15,  # Increased for stronger feature selection
    'retraining_interval': 6,        # Reduced to 6 hours for more frequent updates
    'early_stopping_patience': 3,     # Reduced for faster training cycles
    'learning_rate': 0.0005,         # Reduced for more stable learning
    'batch_size': 32,                # Added small batch size for better generalization
    'dropout_rate': 0.3,             # Added dropout for regularization
    'l2_regularization': 0.01,       # Added L2 regularization
    'momentum': 0.9,                 # Added momentum for optimization
    'use_early_stopping': True,      # Added explicit early stopping flag
    'cross_validation_folds': 3      # Added k-fold validation
}

# Backtesting parameters
BACKTEST_CONFIG = {
    'initial_capital': 10,           # Maintained at $10
    'commission': 0.0001,            # 0.01%
    'slippage': 0.0001,             # 0.01%
    'use_ml': True,                 # Enabled ML features
    'in_sample_size': 200,          # Reduced for faster validation cycles
    'out_sample_size': 100,         # Reduced for more frequent updates
    'step_size': 50,                # Reduced for more granular testing
    'ml_validation_period': 12,     # Reduced to 12 hours for more frequent validation
    'feature_engineering': True,     # Maintained
    'ensemble_weighting': True,      # Maintained
    'use_walk_forward': True,       # Added walk-forward analysis
    'min_trades_per_fold': 5,       # Added minimum trades requirement
    'profit_factor_threshold': 1.5,  # Added minimum profit factor requirement
    'max_drawdown_threshold': 0.05   # Added maximum drawdown threshold (5%)
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
    'update_interval': 300,        # 5 minutes
    'max_retries': 3,
    'retry_delay': 60,            # 1 minute
    'heartbeat_interval': 30,     # 30 seconds
    'cleanup_interval': 86400     # 24 hours
} 