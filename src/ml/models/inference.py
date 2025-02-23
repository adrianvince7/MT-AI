import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
import os
from .feature_engineering import FeatureEngineer

class ModelInference:
    def __init__(self, model_path: str, feature_engineer: FeatureEngineer,
                 threshold: float = 0.5):
        self.model = load_model(model_path)
        self.feature_engineer = feature_engineer
        self.threshold = threshold
        self.prediction_history = []
        
    def prepare_real_time_data(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare real-time market data for prediction."""
        # Calculate features
        features_df, _ = self.feature_engineer.prepare_features(
            market_data.copy(),
            is_training=False
        )
        
        # Create sequence
        sequence_length = 10  # Must match training sequence length
        if len(features_df) < sequence_length:
            return None
            
        # Get latest sequence
        latest_sequence = features_df.iloc[-sequence_length:].values
        return np.expand_dims(latest_sequence, axis=0)
        
    def predict_direction(self, market_data: pd.DataFrame) -> Dict:
        """Predict market direction and confidence."""
        # Prepare data
        sequence = self.prepare_real_time_data(market_data)
        if sequence is None:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'should_trade': False
            }
            
        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        confidence = abs(prediction)
        
        # Determine direction and trading signal
        if prediction > self.threshold:
            direction = 'BUY'
            should_trade = confidence > 0.7  # Higher confidence threshold for trading
        elif prediction < -self.threshold:
            direction = 'SELL'
            should_trade = confidence > 0.7
        else:
            direction = 'NEUTRAL'
            should_trade = False
            
        # Store prediction for tracking
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'direction': direction,
            'confidence': confidence,
            'prediction': prediction
        })
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
            
        return {
            'direction': direction,
            'confidence': float(confidence),
            'should_trade': should_trade,
            'raw_prediction': float(prediction)
        }
        
    def get_ensemble_prediction(self, market_data: pd.DataFrame,
                              model_paths: List[str]) -> Dict:
        """Get ensemble prediction from multiple models."""
        predictions = []
        confidences = []
        
        # Prepare data once
        sequence = self.prepare_real_time_data(market_data)
        if sequence is None:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'should_trade': False
            }
            
        # Get predictions from all models
        for model_path in model_paths:
            model = load_model(model_path)
            pred = model.predict(sequence, verbose=0)[0][0]
            predictions.append(pred)
            confidences.append(abs(pred))
            
        # Ensemble decision
        avg_prediction = np.mean(predictions)
        avg_confidence = np.mean(confidences)
        
        # Determine direction with stricter criteria for ensemble
        if avg_prediction > self.threshold and min(predictions) > 0:
            direction = 'BUY'
            should_trade = avg_confidence > 0.8  # Stricter threshold for ensemble
        elif avg_prediction < -self.threshold and max(predictions) < 0:
            direction = 'SELL'
            should_trade = avg_confidence > 0.8
        else:
            direction = 'NEUTRAL'
            should_trade = False
            
        return {
            'direction': direction,
            'confidence': float(avg_confidence),
            'should_trade': should_trade,
            'raw_prediction': float(avg_prediction),
            'model_agreement': np.std(predictions)  # Lower means higher agreement
        }
        
    def analyze_prediction_history(self, window: int = 100) -> Dict:
        """Analyze recent prediction history."""
        if not self.prediction_history:
            return {
                'trend_strength': 0.0,
                'direction_changes': 0,
                'avg_confidence': 0.0
            }
            
        recent_predictions = self.prediction_history[-window:]
        
        # Calculate trend strength
        predictions = [p['raw_prediction'] for p in recent_predictions]
        trend_strength = np.abs(np.mean(predictions))
        
        # Count direction changes
        directions = [p['direction'] for p in recent_predictions]
        direction_changes = sum(1 for i in range(1, len(directions))
                              if directions[i] != directions[i-1])
        
        # Average confidence
        confidences = [p['confidence'] for p in recent_predictions]
        avg_confidence = np.mean(confidences)
        
        return {
            'trend_strength': float(trend_strength),
            'direction_changes': direction_changes,
            'avg_confidence': float(avg_confidence)
        }
        
    def adjust_prediction_threshold(self, market_volatility: float):
        """Dynamically adjust prediction threshold based on market conditions."""
        # Increase threshold during high volatility
        base_threshold = 0.5
        volatility_factor = min(market_volatility / 0.01, 2.0)  # Cap at 2x
        self.threshold = base_threshold * volatility_factor
        
    def get_trading_decision(self, market_data: pd.DataFrame,
                           market_conditions: Dict) -> Dict:
        """Get final trading decision considering all factors."""
        # Get base prediction
        prediction = self.predict_direction(market_data)
        
        # Adjust threshold based on volatility
        self.adjust_prediction_threshold(market_conditions.get('volatility', 0.01))
        
        # Analyze prediction history
        history_analysis = self.analyze_prediction_history()
        
        # Combine all factors for final decision
        should_trade = (
            prediction['should_trade'] and
            market_conditions.get('suitable_for_trading', False) and
            history_analysis['trend_strength'] > 0.3 and
            history_analysis['direction_changes'] < 5  # Not too choppy
        )
        
        return {
            **prediction,
            'should_trade': should_trade,
            'threshold_used': self.threshold,
            'trend_analysis': history_analysis,
            'market_conditions': market_conditions
        } 