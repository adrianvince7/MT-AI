# MT-AI: Advanced Forex Trading System

An AI-enhanced forex trading system integrating with MetaTrader 5, featuring machine learning-driven decision making, sophisticated risk management, and comprehensive backtesting capabilities.

## Features

- **Advanced Trading Strategies**
  - Momentum Breakout
  - Support/Resistance Reversal
  - VWAP Deviation
  - ML-Enhanced Signal Generation

- **Risk Management**
  - Dynamic Position Sizing
  - Quick Profit Capture
  - Multi-level Risk Controls
  - Performance Analytics

- **Machine Learning Integration**
  - Real-time Feature Engineering
  - Multiple Model Architectures (LSTM, CNN-LSTM, Attention)
  - Walk-Forward Optimization
  - Ensemble Predictions

- **Market Analysis**
  - Session-based Trading
  - Volatility Management
  - Real-time Market Condition Analysis
  - Economic Calendar Integration

## System Requirements

- Python 3.8 or later
- MetaTrader 5 Terminal
- 8GB RAM minimum (16GB recommended)
- 50GB free disk space
- Internet connection for live trading

### Supported Platforms
- macOS 12.0 (Monterey) or later
- Windows 10/11 64-bit

## Installation

### 1. MetaTrader 5 Setup

#### Windows
1. Download and install MetaTrader 5 from your broker's website
2. Launch MT5 and log in to your trading account
3. Enable "AutoTrading" and "Allow Automated Trading"

#### macOS
1. Install MetaTrader 5 using CrossOver or Parallels
2. Configure the Windows environment
3. Follow Windows setup steps above

### 2. Python Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MT-AI.git
cd MT-AI

# Create and activate virtual environment
## Windows
python -m venv venv
venv\Scripts\activate

## macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
INITIAL_DEPOSIT=100000
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
```

## Project Structure

```
MT-AI/
├── src/
│   ├── strategies/          # Trading strategies
│   ├── ml/                  # Machine learning components
│   ├── risk_management/     # Risk management modules
│   ├── market_analysis/     # Market analysis tools
│   ├── backtesting/        # Backtesting engine
│   └── data/               # Data management
├── models/                  # Trained ML models
├── data/                   # Historical data
│   ├── raw/
│   └── processed/
├── results/                # Backtesting results
├── logs/                   # Trading and system logs
└── tests/                  # Test suite
```

## Quick Start

1. **Configure Trading Parameters**

Edit `config/trading_config.py`:
```python
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAMES = ["H1", "H4", "D1"]
STRATEGIES = ["momentum", "sr_reversal", "vwap"]
```

2. **Run Backtesting**

```bash
# Run walk-forward optimization
python -m src.backtesting.run_optimization

# View results in results/optimization/
```

3. **Start Live Trading**

```bash
# Start the trading system
python -m src.main

# Monitor logs in logs/trading.log
```

## Trading Strategy Configuration

### Momentum Breakout
```python
MOMENTUM_CONFIG = {
    'lookback_period': 20,
    'volatility_period': 14,
    'breakout_threshold': 1.5
}
```

### Support/Resistance
```python
SR_CONFIG = {
    'sr_period': 50,
    'zone_threshold': 0.001,
    'reversal_threshold': 2.0
}
```

### VWAP Deviation
```python
VWAP_CONFIG = {
    'vwap_period': 24,
    'deviation_threshold': 2.0,
    'volume_factor': 1.5
}
```

## Risk Management Settings

Edit `config/risk_config.py`:
```python
RISK_SETTINGS = {
    'max_position_size': 0.02,  # 2% per trade
    'max_total_exposure': 0.06, # 6% total
    'daily_loss_limit': 0.05,   # 5% daily max loss
    'trailing_stop': 5          # 5 pip trailing stop
}
```

## Monitoring and Maintenance

### Log Monitoring
- Trading logs: `logs/trading.log`
- System logs: `logs/system.log`
- ML predictions: `logs/predictions.log`

### Performance Analysis
```bash
# Generate performance report
python -m src.analysis.generate_report

# View in results/reports/
```

### Database Maintenance
```bash
# Backup trading database
python -m src.utils.backup_db

# Clean old logs
python -m src.utils.clean_logs
```

## Troubleshooting

### Common Issues

1. **MT5 Connection Issues**
   - Check internet connection
   - Verify MT5 is running and AutoTrading is enabled
   - Confirm credentials in `.env` file

2. **Performance Issues**
   - Close unnecessary applications
   - Check system resource usage
   - Reduce number of monitored symbols

3. **ML Model Errors**
   - Verify model files in models/ directory
   - Check feature engineering pipeline
   - Ensure sufficient historical data

### Error Codes

- `E001`: MT5 connection failed
- `E002`: Insufficient data
- `E003`: Model prediction error
- `E004`: Risk limit exceeded

## Support

For issues and feature requests, please:
1. Check the troubleshooting guide
2. Review existing issues
3. Create a new issue with detailed information

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Disclaimer

Trading forex carries significant risk. This software is for educational purposes only. Always test thoroughly before live trading.
