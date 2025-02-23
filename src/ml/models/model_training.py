import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                   Input, Conv1D, MaxPooling1D, Flatten,
                                   Bidirectional, Attention, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, Optional, List
import optuna
from datetime import datetime
import os

class ModelTrainer:
    def __init__(self, save_dir: str = 'models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def build_lstm_model(self, input_shape: Tuple[int, int],
                        units: List[int] = [64, 32],
                        dropout_rate: float = 0.2) -> Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(units[0], input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout_rate),
            LSTM(units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')  # Output layer for direction prediction
        ])
        return model
        
    def build_cnn_lstm_model(self, input_shape: Tuple[int, int],
                            filters: List[int] = [64, 32],
                            kernel_size: int = 3) -> Model:
        """Build CNN-LSTM hybrid model architecture."""
        model = Sequential([
            Conv1D(filters[0], kernel_size, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(filters[1], kernel_size, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')
        ])
        return model
        
    def build_attention_model(self, input_shape: Tuple[int, int],
                            num_heads: int = 4) -> Model:
        """Build Transformer-based model with attention mechanism."""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention layer
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[-1]
        )(inputs, inputs)
        
        # Combine attention with original input
        x = tf.keras.layers.Add()([inputs, attention_output])
        x = BatchNormalization()(x)
        
        # Process with bidirectional LSTM
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(16))(x)
        
        # Final dense layers
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, activation='tanh')(x)
        
        return Model(inputs=inputs, outputs=outputs)
        
    def train_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   batch_size: int = 32, epochs: int = 100) -> Tuple[Model, Dict]:
        """Train the model with early stopping and checkpointing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.save_dir, f'model_{timestamp}.h5')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history.history
        
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for hyperparameter optimization."""
        # Hyperparameters to optimize
        model_type = trial.suggest_categorical('model_type', ['lstm', 'cnn_lstm', 'attention'])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        
        if model_type == 'lstm':
            units_1 = trial.suggest_categorical('units_1', [32, 64, 128])
            units_2 = trial.suggest_categorical('units_2', [16, 32, 64])
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            model = self.build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                units=[units_1, units_2],
                dropout_rate=dropout_rate
            )
        elif model_type == 'cnn_lstm':
            filters_1 = trial.suggest_categorical('filters_1', [32, 64, 128])
            filters_2 = trial.suggest_categorical('filters_2', [16, 32, 64])
            kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
            model = self.build_cnn_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                filters=[filters_1, filters_2],
                kernel_size=kernel_size
            )
        else:  # attention
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
            model = self.build_attention_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_heads=num_heads
            )
            
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with early stopping
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=50,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        return history.history['val_accuracy'][-1]
        
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               n_trials: int = 100) -> Dict:
        """Optimize hyperparameters using Optuna."""
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials
        )
        
        return study.best_params
        
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                  n_splits: int = 5) -> List[Tuple[Model, Dict]]:
        """Train model using time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Get best hyperparameters
            best_params = self.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials=50
            )
            
            # Build and train model with best parameters
            if best_params['model_type'] == 'lstm':
                model = self.build_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    units=[best_params['units_1'], best_params['units_2']],
                    dropout_rate=best_params['dropout_rate']
                )
            elif best_params['model_type'] == 'cnn_lstm':
                model = self.build_cnn_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    filters=[best_params['filters_1'], best_params['filters_2']],
                    kernel_size=best_params['kernel_size']
                )
            else:
                model = self.build_attention_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    num_heads=best_params['num_heads']
                )
                
            model, history = self.train_model(
                model, X_train, y_train, X_val, y_val,
                batch_size=best_params['batch_size']
            )
            
            models.append((model, history))
            
        return models 