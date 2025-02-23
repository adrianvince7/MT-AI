import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from enum import Enum

class TradeResult(Enum):
    WIN = 'win'
    LOSS = 'loss'
    BREAKEVEN = 'breakeven'

@dataclass
class TradeMetrics:
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    trade_type: str
    entry_price: float
    exit_price: float
    volume: float
    profit_loss: float
    pips: float
    result: TradeResult

class PerformanceManager:
    def __init__(self, db_path: str = 'trading_metrics.db'):
        self.db_path = db_path
        self.daily_loss_limit = 0.05  # 5% account maximum
        self.target_win_rate = 0.60  # 60%
        self.min_profit_factor = 1.8
        self.max_consecutive_losses = 3
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database for storing trade metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP NOT NULL,
                trade_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                volume REAL NOT NULL,
                profit_loss REAL NOT NULL,
                pips REAL NOT NULL,
                result TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_trade(self, metrics: TradeMetrics):
        """Record trade metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                symbol, strategy, entry_time, exit_time, trade_type,
                entry_price, exit_price, volume, profit_loss, pips, result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.symbol, metrics.strategy, metrics.entry_time, metrics.exit_time,
            metrics.trade_type, metrics.entry_price, metrics.exit_price, metrics.volume,
            metrics.profit_loss, metrics.pips, metrics.result.value
        ))
        
        conn.commit()
        conn.close()
        
    def calculate_win_rate(self, period_days: int = 30) -> float:
        """Calculate win rate for specified period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE exit_time >= ?
        ''', (start_date,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0] == 0:  # No trades
            return 0.0
            
        return result[1] / result[0]
        
    def calculate_profit_factor(self, period_days: int = 30) -> float:
        """Calculate profit factor for specified period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN profit_loss < 0 THEN profit_loss ELSE 0 END)) as gross_loss
            FROM trades
            WHERE exit_time >= ?
        ''', (start_date,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result[1]:  # No losses
            return float('inf')
        return result[0] / result[1] if result[1] else 0.0
        
    def get_average_metrics(self, period_days: int = 30) -> Dict:
        """Calculate average trade metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        cursor.execute('''
            SELECT 
                AVG(CASE WHEN profit_loss > 0 THEN pips ELSE 0 END) as avg_win_pips,
                AVG(CASE WHEN profit_loss < 0 THEN pips ELSE 0 END) as avg_loss_pips,
                AVG(profit_loss) as avg_profit_loss
            FROM trades
            WHERE exit_time >= ?
        ''', (start_date,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'average_winner_pips': result[0] or 0.0,
            'average_loser_pips': abs(result[1] or 0.0),
            'average_profit_loss': result[2] or 0.0
        }
        
    def check_consecutive_losses(self) -> int:
        """Check number of consecutive losses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT result
            FROM trades
            ORDER BY exit_time DESC
            LIMIT 10
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        consecutive_losses = 0
        for result in results:
            if result[0] == TradeResult.LOSS.value:
                consecutive_losses += 1
            else:
                break
                
        return consecutive_losses
        
    def calculate_daily_loss(self) -> float:
        """Calculate total loss for current day."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        cursor.execute('''
            SELECT SUM(profit_loss)
            FROM trades
            WHERE exit_time >= ? AND profit_loss < 0
        ''', (today_start,))
        
        result = cursor.fetchone()
        conn.close()
        
        return abs(result[0] or 0.0)
        
    def should_continue_trading(self) -> Tuple[bool, str]:
        """Check if trading should continue based on performance metrics."""
        # Check daily loss limit
        daily_loss = self.calculate_daily_loss()
        if daily_loss >= self.daily_loss_limit:
            return False, "Daily loss limit reached"
            
        # Check consecutive losses
        if self.check_consecutive_losses() >= self.max_consecutive_losses:
            return False, "Maximum consecutive losses reached"
            
        # Check win rate
        win_rate = self.calculate_win_rate()
        if win_rate < self.target_win_rate and win_rate != 0:  # Ignore if no trades yet
            return False, "Win rate below target"
            
        # Check profit factor
        profit_factor = self.calculate_profit_factor()
        if profit_factor < self.min_profit_factor and profit_factor != 0:  # Ignore if no trades yet
            return False, "Profit factor below minimum"
            
        return True, "All metrics within acceptable ranges"
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        return {
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'average_metrics': self.get_average_metrics(),
            'consecutive_losses': self.check_consecutive_losses(),
            'daily_loss': self.calculate_daily_loss(),
            'can_trade': self.should_continue_trading()
        } 