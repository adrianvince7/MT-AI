import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import talib
import tensorflow as tf

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        self.lookback_periods = [5, 10, 20, 50]  # Multiple timeframes for analysis
        
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical analysis features using TA-Lib."""
        # Price action features
        df['returns'] = df['close'].pct_change()
        
        # Trend indicators
        for period in self.lookback_periods:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Stochastic
        df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'],
                                              fastk_period=5, slowk_period=3, slowk_matype=0,
                                              slowd_period=3, slowd_matype=0)
        
        # Volatility indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], 
                                                                       timeperiod=20,
                                                                       nbdevup=2,
                                                                       nbdevdn=2,
                                                                       matype=0)
        
        # Volume indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Remove NaN values
        df = df.dropna()
        
        return df
        
    def calculate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom trading strategy specific features."""
        # Momentum breakout features
        df['ema_alignment'] = ((df['ema_5'] > df['ema_8']) & 
                             (df['ema_8'] > df['ema_13'])).astype(int)
        
        # Support/Resistance features
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # VWAP features
        for period in self.lookback_periods:
            df[f'vwap_deviation_{period}'] = (df['close'] - df[f'vwap_{period}']) / df[f'vwap_{period}']
            
        return df
        
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Session indicators
        df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['europe_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
        df['session_overlap'] = (df['asia_session'] + df['europe_session'] + 
                               df['us_session']).clip(0, 1)
        
        return df
        
    def create_sequence_features(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequential features for time series models."""
        features = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:(i + sequence_length)]
            target = df.iloc[i + sequence_length]
            
            # Calculate target variables (future returns)
            future_return = (target['close'] - sequence['close'].iloc[-1]) / sequence['close'].iloc[-1]
            target_direction = np.sign(future_return)
            
            features.append(sequence.values)
            targets.append(target_direction)
            
        return np.array(features), np.array(targets)
        
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare and normalize features."""
        # Calculate all features
        df = self.calculate_technical_features(df)
        df = self.calculate_custom_features(df)
        df = self.add_temporal_features(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Select features for modeling
        feature_columns = [col for col in df.columns if col not in 
                         ['open', 'high', 'low', 'close', 'volume', 'returns']]
        
        if is_training:
            # Initialize scalers for each feature
            for column in feature_columns:
                self.scalers[column] = StandardScaler()
                df[column] = self.scalers[column].fit_transform(df[[column]])
        else:
            # Use existing scalers for transformation
            for column in feature_columns:
                if column in self.scalers:
                    df[column] = self.scalers[column].transform(df[[column]])
                    
        self.feature_names = feature_columns
        return df[feature_columns], feature_columns
        
    def create_train_test_sequences(self, df: pd.DataFrame, sequence_length: int = 10, 
                                  test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create training and testing sequences."""
        # Prepare features
        df_features, _ = self.prepare_features(df, is_training=True)
        
        # Create sequences
        X, y = self.create_sequence_features(df_features, sequence_length)
        
        # Split into train and test
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test 