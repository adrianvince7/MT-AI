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
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify MetaTrader5 installation
python -c "import MetaTrader5 as mt5; print(mt5.__version__)"
```

If you encounter any errors:
1. Ensure MetaTrader5 package is installed:
```bash
pip install MetaTrader5
```
2. Make sure all dependencies are installed:
```bash
pip install numpy pandas scikit-learn tensorflow python-dotenv pytest
```

## Usage

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env  # On Windows use: copy .env.example .env
```

2. Edit `.env` with your MT5 credentials:
```
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
```

### Running the Trading System

1. Ensure MetaTrader 5 terminal is running and you're logged in

2. Start the main trading system:
```bash
python -m src.main
# or use the installed command
mt-ai
```

3. Run backtesting:
```bash
python -m src.backtesting.run_optimization
# or use the installed command
mt-ai-backtest
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

1. `ModuleNotFoundError: No module named 'MetaTrader5'`
   - Run: `pip install MetaTrader5`
   - Verify installation: `python -c "import MetaTrader5"`

2. MT5 Connection Issues
   - Ensure MetaTrader 5 terminal is running
   - Verify your credentials in `.env`
   - Check your internet connection

3. Package Import Errors
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