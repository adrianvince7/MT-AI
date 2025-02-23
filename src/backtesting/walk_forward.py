import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .backtest_engine import BacktestEngine
from ..strategies.base_strategy import BaseStrategy
from ..ml.strategy_enhancement import MLStrategyEnhancer
from ..ml.models.model_training import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns

class WalkForwardOptimizer:
    def __init__(self,
                 in_sample_size: int = 1000,    # Number of bars for training
                 out_sample_size: int = 500,    # Number of bars for testing
                 step_size: int = 250,          # Number of bars to step forward
                 min_training_size: int = 500): # Minimum bars needed for training
        self.in_sample_size = in_sample_size
        self.out_sample_size = out_sample_size
        self.step_size = step_size
        self.min_training_size = min_training_size
        self.results = []
        
    def generate_windows(self, data: pd.DataFrame) -> List[Dict]:
        """Generate walk-forward windows."""
        windows = []
        start_idx = 0
        
        while start_idx + self.in_sample_size + self.out_sample_size <= len(data):
            window = {
                'train_start': start_idx,
                'train_end': start_idx + self.in_sample_size,
                'test_start': start_idx + self.in_sample_size,
                'test_end': start_idx + self.in_sample_size + self.out_sample_size
            }
            windows.append(window)
            start_idx += self.step_size
            
        return windows
        
    def optimize_window(self, data: pd.DataFrame, window: Dict,
                       strategy: BaseStrategy) -> Tuple[MLStrategyEnhancer, Dict]:
        """Optimize strategy and ML models for a single window."""
        # Split data
        train_data = data.iloc[window['train_start']:window['train_end']]
        test_data = data.iloc[window['test_start']:window['test_end']]
        
        # Generate strategy signals on training data
        strategy_signals = []
        for i in range(len(train_data)):
            signal = strategy.generate_signals(
                train_data.iloc[max(0, i-100):i+1]
            )
            if signal['type']:
                signal['entry_time'] = train_data.index[i]
                strategy_signals.append(signal)
                
        # Train ML models
        ml_enhancer = MLStrategyEnhancer(strategy, [])
        model_path = ml_enhancer.train_strategy_specific_model(
            train_data,
            strategy_signals
        )
        ml_enhancer.model_paths = [model_path]
        
        # Run backtest on test data
        backtest_engine = BacktestEngine(use_ml=True)
        results = backtest_engine.run_backtest(
            strategy,
            test_data,
            ml_enhancer
        )
        
        return ml_enhancer, results
        
    def run_optimization(self, data: pd.DataFrame,
                        strategy: BaseStrategy) -> Dict:
        """Run walk-forward optimization."""
        windows = self.generate_windows(data)
        all_results = []
        ml_enhancers = []
        
        for window in windows:
            ml_enhancer, results = self.optimize_window(
                data,
                window,
                strategy
            )
            
            results['window'] = window
            all_results.append(results)
            ml_enhancers.append(ml_enhancer)
            
        # Aggregate results
        aggregated_results = self.aggregate_results(all_results)
        
        # Select best ML models
        best_models = self.select_best_models(ml_enhancers, all_results)
        
        return {
            'windows': windows,
            'results': all_results,
            'aggregated_results': aggregated_results,
            'best_models': best_models
        }
        
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all windows."""
        # Collect metrics
        win_rates = []
        profit_factors = []
        sharpe_ratios = []
        returns = []
        ml_improvements = []
        
        for result in results:
            metrics = result['performance_metrics']
            win_rates.append(metrics['win_rate'])
            profit_factors.append(metrics['profit_factor'])
            sharpe_ratios.append(metrics['sharpe_ratio'])
            returns.append(
                (result['final_capital'] - result['initial_capital']) /
                result['initial_capital']
            )
            
            # Calculate ML improvement
            ml_metrics = metrics['ml_enhanced_performance']
            if ml_metrics:
                improvement = (ml_metrics['ml_win_rate'] -
                             ml_metrics['non_ml_win_rate'])
                ml_improvements.append(improvement)
                
        return {
            'avg_win_rate': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'avg_profit_factor': np.mean(profit_factors),
            'profit_factor_std': np.std(profit_factors),
            'avg_sharpe': np.mean(sharpe_ratios),
            'sharpe_std': np.std(sharpe_ratios),
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'avg_ml_improvement': np.mean(ml_improvements) if ml_improvements else 0,
            'ml_improvement_std': np.std(ml_improvements) if ml_improvements else 0
        }
        
    def select_best_models(self, ml_enhancers: List[MLStrategyEnhancer],
                          results: List[Dict]) -> List[str]:
        """Select best performing ML models across all windows."""
        model_performance = {}
        
        # Track performance of each model
        for enhancer, result in zip(ml_enhancers, results):
            for model_path in enhancer.model_paths:
                if model_path not in model_performance:
                    model_performance[model_path] = []
                    
                ml_metrics = result['performance_metrics']['ml_enhanced_performance']
                if ml_metrics:
                    score = (ml_metrics['ml_win_rate'] * 
                            result['performance_metrics']['profit_factor'])
                    model_performance[model_path].append(score)
                    
        # Calculate average performance for each model
        avg_performance = {
            path: np.mean(scores)
            for path, scores in model_performance.items()
        }
        
        # Select top 5 models
        best_models = sorted(
            avg_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return [model[0] for model in best_models]
        
    def plot_window_results(self, results: List[Dict],
                          save_path: Optional[str] = None):
        """Plot results across all windows."""
        plt.style.use('seaborn')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics
        win_rates = [r['performance_metrics']['win_rate'] for r in results]
        profit_factors = [r['performance_metrics']['profit_factor'] for r in results]
        returns = [(r['final_capital'] - r['initial_capital']) / 
                  r['initial_capital'] for r in results]
        ml_improvements = [
            r['performance_metrics']['ml_enhanced_performance'].get('ml_win_rate', 0) -
            r['performance_metrics']['ml_enhanced_performance'].get('non_ml_win_rate', 0)
            for r in results
        ]
        
        # Plot metrics over time
        window_numbers = range(1, len(results) + 1)
        
        ax1.plot(window_numbers, win_rates, marker='o')
        ax1.set_title('Win Rate by Window')
        ax1.set_xlabel('Window Number')
        ax1.set_ylabel('Win Rate')
        
        ax2.plot(window_numbers, profit_factors, marker='o')
        ax2.set_title('Profit Factor by Window')
        ax2.set_xlabel('Window Number')
        ax2.set_ylabel('Profit Factor')
        
        ax3.plot(window_numbers, returns, marker='o')
        ax3.set_title('Return by Window')
        ax3.set_xlabel('Window Number')
        ax3.set_ylabel('Return')
        
        ax4.plot(window_numbers, ml_improvements, marker='o')
        ax4.set_title('ML Improvement by Window')
        ax4.set_xlabel('Window Number')
        ax4.set_ylabel('Win Rate Improvement')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_optimization_report(self, results: Dict,
                                   save_path: Optional[str] = None) -> str:
        """Generate detailed optimization report."""
        agg = results['aggregated_results']
        
        report = f"""
Walk-Forward Optimization Report
=============================
Number of Windows: {len(results['windows'])}
In-Sample Size: {self.in_sample_size} bars
Out-of-Sample Size: {self.out_sample_size} bars
Step Size: {self.step_size} bars

Aggregated Performance
-------------------
Average Win Rate: {agg['avg_win_rate']*100:.2f}% (±{agg['win_rate_std']*100:.2f}%)
Average Profit Factor: {agg['avg_profit_factor']:.2f} (±{agg['profit_factor_std']:.2f})
Average Sharpe Ratio: {agg['avg_sharpe']:.2f} (±{agg['sharpe_std']:.2f})
Average Return: {agg['avg_return']*100:.2f}% (±{agg['return_std']*100:.2f}%)

ML Enhancement Performance
----------------------
Average Win Rate Improvement: {agg['avg_ml_improvement']*100:.2f}% (±{agg['ml_improvement_std']*100:.2f}%)

Best Models Selected
-----------------
{chr(10).join(results['best_models'])}

Window Analysis
------------
Best Window:
  Win Rate: {max(r['performance_metrics']['win_rate'] for r in results['results'])*100:.2f}%
  Profit Factor: {max(r['performance_metrics']['profit_factor'] for r in results['results']):.2f}
  Return: {max((r['final_capital'] - r['initial_capital'])/r['initial_capital'] for r in results['results'])*100:.2f}%

Worst Window:
  Win Rate: {min(r['performance_metrics']['win_rate'] for r in results['results'])*100:.2f}%
  Profit Factor: {min(r['performance_metrics']['profit_factor'] for r in results['results']):.2f}
  Return: {min((r['final_capital'] - r['initial_capital'])/r['initial_capital'] for r in results['results'])*100:.2f}%
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report 