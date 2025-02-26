from datetime import datetime
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

class BBandMomentumStrategy(Strategy):
    bb_period = 10  # Shorter period for more signals
    bb_std = 1.5    # Tighter bands for more opportunities
    momentum_period = 5  # Shorter momentum period
    
    def init(self):
        # Calculate Bollinger Bands
        close = pd.Series(self.data.Close)
        
        # Calculate and store indicators
        self.sma = self.I(lambda x: x.rolling(self.bb_period).mean(), close)
        self.std = self.I(lambda x: x.rolling(self.bb_period).std(), close)
        self.bb_upper = self.I(lambda x, y: x + self.bb_std * y, self.sma, self.std)
        self.bb_lower = self.I(lambda x, y: x - self.bb_std * y, self.sma, self.std)
        self.momentum = self.I(lambda x: x.pct_change(self.momentum_period), close)
        self.volatility = self.I(lambda x: x.rolling(20).std(), close)
    
    def next(self):
        # Wait for indicators to be ready
        if pd.isna(self.sma[-1]) or pd.isna(self.momentum[-1]) or pd.isna(self.volatility[-1]):
            return
            
        # Calculate position size based on volatility (with minimum size)
        volatility_pct = max(self.volatility[-1] / self.data.Close[-1], 0.001)
        risk_pct = 0.02  # 2% risk per trade
        position_size = min(
            (self.equity * risk_pct) / (volatility_pct * self.data.Close[-1]),
            self.equity * 0.5  # Maximum 50% of equity per trade
        )
            
        # Long entry
        if (self.data.Close[-1] < self.bb_lower[-1] and 
            self.momentum[-1] > 0 and
            not self.position):
            sl_price = self.data.Close[-1] * 0.99  # 1% stop loss
            tp_price = self.data.Close[-1] * 1.02  # 2% take profit
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            
        # Short entry    
        elif (self.data.Close[-1] > self.bb_upper[-1] and 
              self.momentum[-1] < 0 and
              not self.position):
            sl_price = self.data.Close[-1] * 1.01  # 1% stop loss
            tp_price = self.data.Close[-1] * 0.98  # 2% take profit
            self.sell(size=position_size, sl=sl_price, tp=tp_price)
            
        # Update trailing stops
        elif self.position:
            if self.position.is_long and self.data.Close[-1] < self.sma[-1]:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] > self.sma[-1]:
                self.position.close()

def generate_sample_data(periods=1000):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate more volatile random walk with trends
    t = np.linspace(0, 4*np.pi, periods)
    trend = np.sin(t) * 10  # Add a sine wave trend
    
    returns = np.random.standard_normal(periods) * 0.02  # 2% daily volatility
    returns = returns + trend * 0.001  # Add trend component
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate other OHLCV data with more realistic relationships
    daily_range = close * 0.015  # 1.5% average daily range
    high = close + np.random.uniform(0, daily_range, periods)
    low = close - np.random.uniform(0, daily_range, periods)
    open = low + np.random.uniform(0, high - low, periods)
    
    # Volume increases with volatility
    volume = np.random.uniform(1000, 5000, periods) * (1 + np.abs(returns))
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='5T')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df

def main():
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    
    print("\nData Sample:")
    print("============")
    print(df.head())
    print("\nPeriod:", df.index[0], "to", df.index[-1])
    print("Number of candles:", len(df))
    
    # Run backtest
    bt = Backtest(df, BBandMomentumStrategy,
                 cash=10000,
                 commission=.0002,
                 exclusive_orders=True,
                 trade_on_close=True)
                 
    stats = bt.run()
    print("\nBacktest Results:")
    print("================")
    print(f"Total Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Number of Trades: {stats['# Trades']}")
    
    # Additional performance metrics
    print("\nDetailed Statistics:")
    print("===================")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Average Trade: {stats['Avg. Trade [%]']:.2f}%")
    print(f"Max Trade Duration: {stats['Max. Trade Duration']}")
    print(f"Avg Trade Duration: {stats['Avg. Trade Duration']}")
    
    # Plot the results
    bt.plot()

if __name__ == "__main__":
    main() 