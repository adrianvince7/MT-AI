import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from ..strategies.base_strategy import BaseStrategy
from ..ml.strategy_enhancement import MLStrategyEnhancer
from ..risk_management.performance_metrics import PerformanceManager
from ..market_analysis.market_conditions import MarketConditionAnalyzer

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.0001,  # 0.01%
                 slippage: float = 0.0001,    # 0.01%
                 use_ml: bool = True):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_ml = use_ml
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
    def calculate_position_value(self, price: float, volume: float) -> float:
        """Calculate position value including commission and slippage."""
        transaction_cost = price * volume * (self.commission + self.slippage)
        return price * volume + transaction_cost
        
    def execute_trade(self, signal: Dict, current_price: float, 
                     available_capital: float) -> Tuple[float, Dict]:
        """Execute trade based on signal and return cost and trade details."""
        if not signal['type']:
            return 0.0, None
            
        volume = signal['position_sizes'][0]  # Initial position size
        position_value = self.calculate_position_value(current_price, volume)
        
        if position_value > available_capital:
            # Adjust position size based on available capital
            volume = available_capital / (current_price * (1 + self.commission + self.slippage))
            position_value = self.calculate_position_value(current_price, volume)
            
        trade = {
            'entry_time': signal['entry_time'],
            'type': signal['type'],
            'entry_price': current_price,
            'volume': volume,
            'stop_loss': signal['stop_loss'],
            'targets': signal['targets'],
            'position_value': position_value,
            'remaining_targets': signal['targets'].copy(),
            'remaining_volume': volume,
            'max_profit': 0.0,
            'current_profit': 0.0,
            'status': 'OPEN',
            'ml_enhanced': signal.get('ml_enhanced', False)
        }
        
        return position_value, trade
        
    def update_trade(self, trade: Dict, current_price: float, 
                    current_time: datetime) -> Tuple[float, bool]:
        """Update trade status and return realized profit and close status."""
        if trade['status'] == 'CLOSED':
            return 0.0, True
            
        # Calculate current profit
        if trade['type'] == 'BUY':
            profit_pips = current_price - trade['entry_price']
        else:  # SELL
            profit_pips = trade['entry_price'] - current_price
            
        trade['current_profit'] = profit_pips * trade['remaining_volume']
        trade['max_profit'] = max(trade['max_profit'], trade['current_profit'])
        
        # Check stop loss
        if (trade['type'] == 'BUY' and current_price <= trade['stop_loss']) or \
           (trade['type'] == 'SELL' and current_price >= trade['stop_loss']):
            realized_profit = trade['current_profit']
            trade['status'] = 'CLOSED'
            trade['exit_time'] = current_time
            trade['exit_price'] = current_price
            trade['exit_reason'] = 'STOP_LOSS'
            return realized_profit, True
            
        # Check take profit levels
        if trade['remaining_targets']:
            target_hit = False
            if trade['type'] == 'BUY' and current_price >= trade['remaining_targets'][0]:
                target_hit = True
            elif trade['type'] == 'SELL' and current_price <= trade['remaining_targets'][0]:
                target_hit = True
                
            if target_hit:
                # Calculate partial profit
                exit_volume = trade['volume'] * 0.5  # Exit half position at each target
                trade['remaining_volume'] -= exit_volume
                realized_profit = (current_price - trade['entry_price']) * exit_volume \
                                if trade['type'] == 'BUY' else \
                                (trade['entry_price'] - current_price) * exit_volume
                trade['remaining_targets'].pop(0)
                
                if not trade['remaining_targets']:
                    trade['status'] = 'CLOSED'
                    trade['exit_time'] = current_time
                    trade['exit_price'] = current_price
                    trade['exit_reason'] = 'TARGET_REACHED'
                    
                return realized_profit, len(trade['remaining_targets']) == 0
                
        return 0.0, False
        
    def run_backtest(self, strategy: BaseStrategy, historical_data: pd.DataFrame,
                    ml_enhancer: Optional[MLStrategyEnhancer] = None) -> Dict:
        """Run backtest simulation."""
        capital = self.initial_capital
        current_trades = []
        market_analyzer = MarketConditionAnalyzer()
        
        for i in range(len(historical_data)):
            current_time = historical_data.index[i]
            current_price = historical_data['close'].iloc[i]
            
            # Update existing trades
            for trade in current_trades:
                if trade['status'] == 'OPEN':
                    profit, closed = self.update_trade(trade, current_price, current_time)
                    capital += profit
                    if closed:
                        self.trades.append(trade)
                        
            # Remove closed trades
            current_trades = [t for t in current_trades if t['status'] == 'OPEN']
            
            # Check market conditions
            market_conditions = market_analyzer.analyze_market_conditions(
                historical_data.iloc[max(0, i-100):i+1]
            )
            
            # Generate new signals
            signal = strategy.generate_signals(
                historical_data.iloc[max(0, i-100):i+1]
            )
            
            if signal['type'] and self.use_ml and ml_enhancer:
                signal = ml_enhancer.enhance_signal(
                    signal,
                    historical_data.iloc[max(0, i-100):i+1],
                    market_conditions
                )
                
            # Execute new trades if conditions are met
            if signal['type'] and market_conditions['suitable_for_trading']:
                position_value, trade = self.execute_trade(
                    signal,
                    current_price,
                    capital
                )
                if trade:
                    current_trades.append(trade)
                    capital -= position_value
                    
            # Record equity
            total_position_value = sum(
                t['remaining_volume'] * current_price 
                for t in current_trades
            )
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': capital + total_position_value
            })
            
        # Close any remaining trades at the end
        for trade in current_trades:
            if trade['status'] == 'OPEN':
                trade['status'] = 'CLOSED'
                trade['exit_time'] = historical_data.index[-1]
                trade['exit_price'] = historical_data['close'].iloc[-1]
                trade['exit_reason'] = 'BACKTEST_END'
                self.trades.append(trade)
                
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve[-1]['equity'],
            'total_trades': len(self.trades),
            'performance_metrics': self.performance_metrics,
            'equity_curve': pd.DataFrame(self.equity_curve)
        }
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return
            
        # Basic metrics
        profitable_trades = [t for t in self.trades if t['current_profit'] > 0]
        losing_trades = [t for t in self.trades if t['current_profit'] <= 0]
        
        self.performance_metrics = {
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(profitable_trades) / len(self.trades),
            'average_profit': np.mean([t['current_profit'] for t in profitable_trades]) if profitable_trades else 0,
            'average_loss': np.mean([t['current_profit'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t['current_profit'] for t in profitable_trades) / 
                               sum(t['current_profit'] for t in losing_trades)) if losing_trades else float('inf'),
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'ml_enhanced_performance': self.calculate_ml_performance()
        }
        
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        equity = pd.DataFrame(self.equity_curve)['equity']
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        return float(drawdowns.min())
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve."""
        equity = pd.DataFrame(self.equity_curve)['equity']
        returns = equity.pct_change().dropna()
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())
        
    def calculate_ml_performance(self) -> Dict:
        """Calculate performance metrics for ML-enhanced trades."""
        if not any(t.get('ml_enhanced', False) for t in self.trades):
            return {}
            
        ml_trades = [t for t in self.trades if t.get('ml_enhanced', False)]
        non_ml_trades = [t for t in self.trades if not t.get('ml_enhanced', False)]
        
        return {
            'ml_trades_count': len(ml_trades),
            'ml_win_rate': len([t for t in ml_trades if t['current_profit'] > 0]) / len(ml_trades) if ml_trades else 0,
            'non_ml_win_rate': len([t for t in non_ml_trades if t['current_profit'] > 0]) / len(non_ml_trades) if non_ml_trades else 0,
            'ml_avg_profit': np.mean([t['current_profit'] for t in ml_trades]) if ml_trades else 0,
            'non_ml_avg_profit': np.mean([t['current_profit'] for t in non_ml_trades]) if non_ml_trades else 0
        }
        
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        plt.style.use('seaborn')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        ax1.plot(equity_df['timestamp'], equity_df['equity'])
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity')
        
        # Plot drawdown
        equity = equity_df['equity']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        ax2.fill_between(equity_df['timestamp'], drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        
        # Plot trade distribution
        profits = [t['current_profit'] for t in self.trades]
        sns.histplot(profits, ax=ax3, bins=50)
        ax3.set_title('Trade Profit Distribution')
        ax3.set_xlabel('Profit')
        ax3.set_ylabel('Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate detailed backtest report."""
        report = f"""
Backtest Report
==============
Period: {self.equity_curve[0]['timestamp']} to {self.equity_curve[-1]['timestamp']}

Performance Summary
-----------------
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.equity_curve[-1]['equity']:,.2f}
Total Return: {((self.equity_curve[-1]['equity'] / self.initial_capital) - 1) * 100:.2f}%
Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}
Max Drawdown: {self.performance_metrics['max_drawdown']*100:.2f}%

Trading Statistics
----------------
Total Trades: {self.performance_metrics['total_trades']}
Win Rate: {self.performance_metrics['win_rate']*100:.2f}%
Profit Factor: {self.performance_metrics['profit_factor']:.2f}
Average Profit: ${self.performance_metrics['average_profit']:,.2f}
Average Loss: ${self.performance_metrics['average_loss']:,.2f}

ML Enhancement Analysis
--------------------
ML-Enhanced Trades: {self.performance_metrics['ml_enhanced_performance'].get('ml_trades_count', 0)}
ML Win Rate: {self.performance_metrics['ml_enhanced_performance'].get('ml_win_rate', 0)*100:.2f}%
Non-ML Win Rate: {self.performance_metrics['ml_enhanced_performance'].get('non_ml_win_rate', 0)*100:.2f}%
ML Average Profit: ${self.performance_metrics['ml_enhanced_performance'].get('ml_avg_profit', 0):,.2f}
Non-ML Average Profit: ${self.performance_metrics['ml_enhanced_performance'].get('non_ml_avg_profit', 0):,.2f}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report 