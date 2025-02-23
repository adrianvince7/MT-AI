import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from ..strategies.momentum_breakout import MomentumBreakoutStrategy
from ..strategies.support_resistance import SupportResistanceStrategy
from ..strategies.vwap_deviation import VWAPDeviationStrategy
from .walk_forward import WalkForwardOptimizer
from ..data.data_loader import DataLoader
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class OptimizationRunner:
    """Runs walk-forward optimization for all trading strategies."""
    
    def __init__(self, data_path: str, output_dir: str,
                 symbols: List[str], timeframes: List[str]):
        """Initialize the optimization runner.
        
        Args:
            data_path: Path to historical data directory
            output_dir: Path to save optimization results
            symbols: List of currency pairs to optimize
            timeframes: List of timeframes to test
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_loader = DataLoader()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_strategies(self) -> List[Dict]:
        """Return list of strategies with their configurations."""
        return [
            {
                'name': 'MomentumBreakout',
                'class': MomentumBreakoutStrategy,
                'params': {
                    'lookback_period': 20,
                    'volatility_period': 14,
                    'breakout_threshold': 1.5
                }
            },
            {
                'name': 'SupportResistance',
                'class': SupportResistanceStrategy,
                'params': {
                    'sr_period': 50,
                    'zone_threshold': 0.001,
                    'reversal_threshold': 2.0
                }
            },
            {
                'name': 'VWAPDeviation',
                'class': VWAPDeviationStrategy,
                'params': {
                    'vwap_period': 24,
                    'deviation_threshold': 2.0,
                    'volume_factor': 1.5
                }
            }
        ]
        
    def optimize_strategy(self, data: pd.DataFrame, strategy_config: Dict,
                         output_dir: Path) -> None:
        """Optimize a single strategy.
        
        Args:
            data: Historical price data
            strategy_config: Strategy configuration dictionary
            output_dir: Directory to save results
        """
        strategy_name = strategy_config['name']
        strategy_class = strategy_config['class']
        strategy_params = strategy_config['params']
        
        try:
            # Initialize strategy and optimizer
            strategy = strategy_class(**strategy_params)
            optimizer = WalkForwardOptimizer(
                in_sample_size=1000,
                out_sample_size=500,
                step_size=250
            )
            
            # Run optimization
            results = optimizer.run_optimization(data, strategy)
            
            # Save results
            report_path = output_dir / f"{strategy_name}_report.txt"
            plot_path = output_dir / f"{strategy_name}_performance.png"
            
            optimizer.generate_optimization_report(results, str(report_path))
            optimizer.plot_window_results(results['results'], str(plot_path))
            
            logger.info(f"Optimization complete for {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error optimizing {strategy_name}: {str(e)}")
        
    def run_optimization(self):
        """Run walk-forward optimization for all strategies and instruments."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                logger.info(f"Processing {symbol} on {timeframe} timeframe")
                
                # Load and validate data
                data = self.data_loader.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    data_path=self.data_path
                )
                
                if data is None or len(data) < 2000:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue
                
                # Setup output directory
                output_subdir = self.output_dir / timestamp / symbol / timeframe
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Optimize each strategy
                for strategy_config in self.get_strategies():
                    self.optimize_strategy(data, strategy_config, output_subdir)
                    
        logger.info("Walk-forward optimization completed for all strategies")

def main():
    """Run the optimization process."""
    # Configuration
    DATA_PATH = "data/historical"
    OUTPUT_DIR = "results/optimization"
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
    TIMEFRAMES = ["H1", "H4", "D1"]
    
    # Run optimization
    runner = OptimizationRunner(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        symbols=SYMBOLS,
        timeframes=TIMEFRAMES
    )
    runner.run_optimization()

if __name__ == "__main__":
    main() 