from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from .models.feature_engineering import FeatureEngineer
from .models.model_training import ModelTrainer
from .models.inference import ModelInference
from ..strategies.base_strategy import BaseStrategy

class MLStrategyEnhancer:
    def __init__(self, base_strategy: BaseStrategy, model_paths: List[str]):
        self.base_strategy = base_strategy
        self.feature_engineer = FeatureEngineer()
        self.model_inference = ModelInference(
            model_paths[0],  # Primary model
            self.feature_engineer
        )
        self.model_paths = model_paths
        self.min_confidence = 0.7
        self.min_agreement = 0.8
        
    def enhance_signal(self, signal: Dict, market_data: pd.DataFrame,
                      market_conditions: Dict) -> Dict:
        """Enhance trading signal with ML predictions."""
        if not signal['type']:  # No base signal
            return signal
            
        # Get ensemble prediction
        ml_prediction = self.model_inference.get_ensemble_prediction(
            market_data,
            self.model_paths
        )
        
        # Check if ML prediction agrees with base signal
        ml_agrees = (
            (signal['type'] == 'BUY' and ml_prediction['direction'] == 'BUY') or
            (signal['type'] == 'SELL' and ml_prediction['direction'] == 'SELL')
        )
        
        # Adjust signal based on ML prediction
        if ml_agrees and ml_prediction['confidence'] >= self.min_confidence:
            # Strengthen the signal
            signal['ml_enhanced'] = True
            signal['confidence'] = ml_prediction['confidence']
            
            # Adjust position sizes based on confidence
            confidence_factor = min(ml_prediction['confidence'], 1.2)  # Cap at 120%
            signal['position_sizes'] = [
                size * confidence_factor for size in signal['position_sizes']
            ]
            
            # Adjust targets based on trend strength
            trend_analysis = self.model_inference.analyze_prediction_history()
            if trend_analysis['trend_strength'] > 0.5:
                # Extend targets for strong trends
                signal['targets'] = [
                    target * 1.2 for target in signal['targets']
                ]
        else:
            # Weaken or invalidate the signal
            signal['ml_enhanced'] = False
            signal['type'] = None  # Cancel the signal
            
        # Add ML metadata to signal
        signal['ml_metadata'] = {
            'prediction': ml_prediction,
            'agreement': ml_agrees,
            'trend_analysis': trend_analysis if 'trend_analysis' in locals() else None
        }
        
        return signal
        
    def train_strategy_specific_model(self, historical_data: pd.DataFrame,
                                    strategy_signals: List[Dict]) -> str:
        """Train ML model specific to the strategy's characteristics."""
        # Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_strategy_training_data(
            historical_data,
            strategy_signals
        )
        
        # Initialize model trainer
        trainer = ModelTrainer(save_dir='models/strategy_specific')
        
        # Train models with cross-validation
        models = trainer.train_with_cross_validation(
            X_train, y_train, n_splits=5
        )
        
        # Select best model based on validation performance
        best_model_idx = np.argmax([
            np.mean(history['val_accuracy']) for _, history in models
        ])
        
        # Save best model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/strategy_specific/model_{timestamp}.h5'
        models[best_model_idx][0].save(model_path)
        
        return model_path
        
    def prepare_strategy_training_data(self, historical_data: pd.DataFrame,
                                     strategy_signals: List[Dict]) -> Tuple:
        """Prepare training data using strategy signals as labels."""
        # Create labels from strategy signals
        labels = pd.Series(0, index=historical_data.index)
        for signal in strategy_signals:
            if signal['type'] == 'BUY':
                labels[signal['entry_time']] = 1
            elif signal['type'] == 'SELL':
                labels[signal['entry_time']] = -1
                
        # Prepare features
        features_df, _ = self.feature_engineer.prepare_features(
            historical_data,
            is_training=True
        )
        
        # Create sequences
        return self.feature_engineer.create_train_test_sequences(
            features_df,
            sequence_length=10,
            test_size=0.2
        )
        
    def analyze_model_performance(self, historical_data: pd.DataFrame,
                                strategy_signals: List[Dict]) -> Dict:
        """Analyze ML model performance against strategy signals."""
        # Get ML predictions for historical data
        predictions = []
        for i in range(len(historical_data) - 100):  # 100 bar lookback
            window = historical_data.iloc[i:i+100]
            pred = self.model_inference.predict_direction(window)
            predictions.append(pred)
            
        # Compare with strategy signals
        true_positives = 0
        false_positives = 0
        missed_signals = 0
        
        for signal in strategy_signals:
            # Find corresponding ML prediction
            pred_idx = historical_data.index.get_loc(signal['entry_time'])
            if pred_idx >= len(predictions):
                continue
                
            ml_pred = predictions[pred_idx]
            
            if signal['type'] == ml_pred['direction']:
                true_positives += 1
            else:
                missed_signals += 1
                
        # Calculate metrics
        total_predictions = len([p for p in predictions if p['should_trade']])
        false_positives = total_predictions - true_positives
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'missed_signals': missed_signals,
            'precision': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
            'recall': true_positives / (true_positives + missed_signals) if (true_positives + missed_signals) > 0 else 0,
            'total_predictions': total_predictions,
            'total_signals': len(strategy_signals)
        }
        
    def update_models(self, new_data: pd.DataFrame, strategy_signals: List[Dict]) -> None:
        """Update ML models with new data."""
        # Train new model
        new_model_path = self.train_strategy_specific_model(
            new_data,
            strategy_signals
        )
        
        # Add to model ensemble
        self.model_paths.append(new_model_path)
        
        # Keep only best 5 models
        if len(self.model_paths) > 5:
            # Evaluate models and remove worst performing
            performances = []
            for path in self.model_paths:
                model = ModelInference(path, self.feature_engineer)
                perf = self.analyze_model_performance(new_data, strategy_signals)
                performances.append((path, perf['precision'] * perf['recall']))
                
            # Sort by performance and keep top 5
            performances.sort(key=lambda x: x[1], reverse=True)
            self.model_paths = [p[0] for p in performances[:5]] 