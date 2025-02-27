# MT-AI: AI-Enhanced Forex Trading System

A sophisticated forex trading system that leverages machine learning for market analysis and decision making.

## Features

- Machine Learning driven market analysis
- Multiple trading strategies implementation
- Backtesting capabilities
- Real-time trading with MT5 integration
- Risk management system
- Market data analysis tools

## Installation

### Prerequisites

- Python 3.8 or higher
- MetaTrader 5 platform installed
- Valid MT5 account credentials
- pip (Python package installer)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/MT-AI.git
cd MT-AI

# Create and activate virtual environment
# On Windows PowerShell:
py -m venv venv --clear
.\venv\Scripts\activate  # You should see (venv) in your prompt after this

# On Windows Command Prompt:
py -m venv venv --clear
venv\Scripts\activate.bat

# On Unix/MacOS:
python3 -m venv venv
source venv/bin/activate

# Install the package with development dependencies
py -m pip install -e ".[dev,test]"

# Verify installation
py -m pytest  # Should run without errors
```

If you encounter any errors:

1. PowerShell Execution Policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

2. Python Version Compatibility:
   - Ensure you're using Python 3.8 or higher
   - Check with: `py --version`

3. Package Installation Issues:
   - Update pip: `py -m pip install --upgrade pip`
   - Clear pip cache: `py -m pip cache purge`
   - Try installing without optional dependencies: `py -m pip install -e .`

## Usage

### Initial Setup

1. Create and configure your environment:
```bash
# Copy environment template
copy .env.example .env  # On Windows
# or
cp .env.example .env    # On Unix/MacOS

# Edit .env with your settings
notepad .env  # On Windows
# or
nano .env     # On Unix/MacOS
```

2. Configure MT5 Credentials:
   - Open your `.env` file
   - Update the MT5 credentials section:
   ```
   MT5_LOGIN=your_login_number
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server
   ```
   - These credentials will be used by default unless overridden by command line arguments

3. Test MT5 Connection:
```bash
# Run the connection test
py -m src.test_connection
```

### Quick Start Backtesting

For optimal backtesting results, follow these recommended periods:

1. Initial Strategy Validation (Quick):
```bash
# Test over 1 month of recent data
py -m src.backtest_runner --strategy ScalpingStrategy --period "2024-02-01 2024-02-29" --verbose
```

2. Strategy Robustness Testing (Recommended):
```bash
# Test over 3 months with different market conditions
py -m src.backtest_runner --strategy ScalpingStrategy --period "2023-12-01 2024-02-29" --verbose
```

3. Full Historical Validation (Comprehensive):
```bash
# Test over 1 year for thorough validation
py -m src.backtest_runner --strategy ScalpingStrategy --period "2023-03-01 2024-02-29" --verbose
```

Recommended Testing Sequence:
1. Start with 1-month test to quickly validate strategy logic
2. If results are promising, expand to 3-month test
3. Finally, run 1-year test for comprehensive validation

Tips for Optimal Backtesting:
- Recent data (last 3 months) is most relevant for current market conditions
- Include both trending and ranging market periods
- Test during major market events for robustness
- Use multiple timeframes to validate consistency

Example Quick Start (Recommended):
```bash
# 1. First quick test (1 week)
py -m src.backtest_runner --strategy ScalpingStrategy --period "2024-02-20 2024-02-27" --initial_capital 10000 --verbose

# 2. If promising, test recent month
py -m src.backtest_runner --strategy ScalpingStrategy --period "2024-02-01 2024-02-29" --initial_capital 10000 --verbose

# 3. Finally, validate with 3 months
py -m src.backtest_runner --strategy ScalpingStrategy --period "2023-12-01 2024-02-29" --initial_capital 10000 --verbose
```

Performance Benchmarks:
- Minimum Win Rate: > 55%
- Risk-Reward Ratio: > 1.5
- Maximum Drawdown: < 20%
- Profit Factor: > 1.3

The system will generate detailed performance metrics and visualizations for each test period.

### Running the System

The MT-AI system can be run in different modes:

#### 1. Live Trading Mode
```bash
# Start the main trading system
py -m src.main

# Or use the installed command
mt-ai
```

Key Features:
- Real-time market data processing
- Live trading execution
- Risk management monitoring
- Performance tracking

#### 2. Backtesting Mode

There are several ways to run backtesting:

a. Using credentials from `.env` file:
```bash
# Run backtesting with default parameters
py -m src.backtest_runner --strategy ScalpingStrategy --period "2024-01-01 2024-02-26"
```

b. Overriding specific credentials:
```bash
# Override MT5 login while using other credentials from .env
py -m src.backtest_runner --strategy ScalpingStrategy --period "2024-01-01 2024-02-26" --mt5_login 12345
```

c. Running without MT5 credentials (backtest mode only):
```bash
# Run in pure backtest mode
py -m src.backtest_runner --strategy ScalpingStrategy --period "2024-01-01 2024-02-26" --verbose
```

Available Arguments:
- `--strategy`: Name of the strategy to test (default: ScalpingStrategy)
- `--period`: Testing period in format "YYYY-MM-DD YYYY-MM-DD"
- `--initial_capital`: Starting capital for backtesting (default: 10000.0)
- `--mt5_login`: Override MT5 login from .env
- `--mt5_password`: Override MT5 password from .env
- `--mt5_server`: Override MT5 server from .env
- `--verbose`: Enable detailed output

#### 3. Strategy Development Mode
```bash
# Run strategy optimization
py -m src.backtesting.run_optimization --strategy YourStrategy

# Test a specific strategy
py -m src.strategies.your_strategy --debug
```

### Monitoring and Management

1. View Trading Logs:
```bash
# Real-time log monitoring
py -m src.main --log-level DEBUG

# View historical logs
type logs\trading.log  # On Windows
# or
cat logs/trading.log   # On Unix/MacOS
```

2. Performance Analysis:
```bash
# Generate performance report
py -m src.market_analysis.performance_report

# View results
start results\performance_report.html  # On Windows
# or
open results/performance_report.html   # On Unix/MacOS
```

### System Components

1. Risk Management Settings (in .env):
```
RISK_PERCENTAGE=1.0      # Risk per trade
MAX_POSITIONS=5          # Maximum open positions
STOP_LOSS_PIPS=50       # Stop loss in pips
TAKE_PROFIT_PIPS=100    # Take profit in pips
```

2. Data Collection Configuration:
```
SYMBOLS=["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAMES=["M15", "H1", "H4"]
HISTORY_DAYS=30
```

3. Machine Learning Parameters:
```
MODEL_VERSION=v1.0
BATCH_SIZE=32
EPOCHS=100
VALIDATION_SPLIT=0.2
```

### Safety Measures

1. Emergency Stop:
   - Press Ctrl+C to safely stop the trading system
   - All open positions will be properly closed
   - State will be saved for recovery

2. Recovery Process:
```bash
# Recover from last known good state
py -m src.main --recover

# Force close all positions
py -m src.main --emergency-close
```

### Maintenance

1. Database Maintenance:
```bash
# Backup trading database
py -m src.data.backup_db

# Clean old data
py -m src.data.cleanup --older-than 90
```

2. Model Management:
```bash
# Update ML models
py -m src.ml.update_models

# Clean old model versions
py -m src.ml.cleanup_models --keep-last 3
```

## Project Structure

```
MT-AI/
├── src/                    # Source code
│   ├── main.py            # Main entry point
│   ├── backtesting/       # Backtesting modules
│   ├── brokers/           # Broker integration
│   ├── config/            # Configuration
│   ├── data/              # Data handling
│   ├── market_analysis/   # Market analysis tools
│   ├── ml/                # Machine learning models
│   ├── risk_management/   # Risk management
│   └── strategies/        # Trading strategies
├── tests/                 # Test suite
├── models/               # Saved ML models
├── data/                 # Market data
├── logs/                 # Application logs
├── results/              # Trading results
└── config/              # Configuration files
```

## Development Environment

The project uses several development tools to ensure code quality:

### Code Formatting and Linting

1. Black (Code Formatter):
```bash
# Format a specific file
py -m black src/your_file.py

# Format the entire project
py -m black .
```

2. isort (Import Sorter):
```bash
# Sort imports in a file
py -m isort src/your_file.py

# Sort all project imports
py -m isort .
```

3. Flake8 (Linter):
```bash
# Lint a specific file
py -m flake8 src/your_file.py

# Lint the entire project
py -m flake8
```

4. MyPy (Type Checker):
```bash
# Type check a file
py -m mypy src/your_file.py

# Type check the project
py -m mypy .
```

### Testing

The project uses pytest for testing:

```bash
# Run all tests
py -m pytest

# Run tests with coverage
py -m pytest --cov=src

# Run specific test file
py -m pytest tests/test_specific.py

# Run tests with verbose output
py -m pytest -v
```

## Development

### Running Tests

```bash
pytest
```

### Adding New Strategies

1. Create a new strategy file in `src/strategies/`
2. Implement the strategy interface
3. Register the strategy in `src/strategy_manager.py`

## Troubleshooting

### Common Issues

1. Virtual Environment Activation Issues
   - On Windows PowerShell, if activation fails:
     ```powershell
     # First, allow script execution
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
     
     # Then activate
     .\venv\Scripts\Activate.ps1
     ```
   - If scripts are missing, recreate the venv:
     ```powershell
     Remove-Item -Recurse -Force venv  # Remove old venv
     cmd /c "py -3.8 -m venv venv --clear"  # Create new one
     ```

2. `ModuleNotFoundError: No module named 'MetaTrader5'`
   - Run: `pip install MetaTrader5`
   - Verify installation: `python -c "import MetaTrader5"`

3. MT5 Connection Issues
   - Ensure MetaTrader 5 terminal is running
   - Verify your credentials in `.env`
   - Check your internet connection

4. Package Import Errors
   - Ensure you're in your virtual environment
   - Run: `pip install -r requirements.txt`
   - Verify installation: `pip list`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 