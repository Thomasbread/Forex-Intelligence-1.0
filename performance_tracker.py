import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import random

# File to store performance data
PERFORMANCE_FILE = "performance_data.json"

def get_performance_history():
    """
    Gets historical performance data.
    
    Returns:
        pandas.DataFrame: Historical performance data
    """
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                performance_data = json.load(f)
            
            if performance_data:
                return pd.DataFrame(performance_data)
        except Exception as e:
            print(f"Error loading performance data: {e}")
    
    # If file doesn't exist or is empty, generate initial history
    return generate_initial_performance_history()

def save_performance_data(performance_df):
    """
    Saves performance data to file.
    
    Args:
        performance_df (pandas.DataFrame): Performance data to save
    """
    try:
        # Convert to dictionary records format
        performance_list = performance_df.to_dict('records')
        
        # Custom JSON serializer to handle non-serializable types like numpy.int64/float64
        def json_serializer(obj):
            import numpy as np
            if isinstance(obj, (np.integer)):
                return int(obj)
            elif isinstance(obj, (np.floating)):
                return float(obj)
            elif isinstance(obj, (np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif hasattr(obj, 'item'):
                return obj.item()  # Convert numpy scalar to Python scalar
            elif pd.isna(obj):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Write to file with custom serializer
        with open(PERFORMANCE_FILE, "w") as f:
            json.dump(performance_list, f, default=json_serializer)
            
    except Exception as e:
        print(f"Error saving performance data: {e}")

def update_performance(new_signals_df):
    """
    Updates performance tracking for old signals.
    
    Args:
        new_signals_df (pandas.DataFrame): New trading signals
    """
    performance_df = get_performance_history()
    
    # If we have existing performance data
    if not performance_df.empty:
        # Get existing signal IDs that haven't been evaluated yet
        pending_signals = performance_df[performance_df['result'].isnull()]
        
        if not pending_signals.empty:
            # Update results for pending signals
            for idx, signal in pending_signals.iterrows():
                # Check if signal is older than 24 hours (enough time to reach TP or SL)
                signal_time = datetime.strptime(signal['timestamp'], "%Y-%m-%d %H:%M:%S")
                
                if datetime.now() - signal_time > timedelta(hours=24):
                    # In a real app, we would check if price hit TP or SL
                    # For demo, we'll use a weighted random approach with 3:1 success ratio
                    success_probability = 0.75  # 75% success rate overall
                    
                    # Adjust probability based on confidence
                    if signal['confidence'] == 'sicher':
                        success_probability = 0.85
                    elif signal['confidence'] == 'mittel':
                        success_probability = 0.7
                    elif signal['confidence'] == 'unsicher':
                        success_probability = 0.55
                    
                    # Determine outcome
                    outcome = 'success' if random.random() < success_probability else 'failure'
                    
                    # Convert Series values to Python scalar values if needed
                    action = signal['action'].item() if hasattr(signal['action'], 'item') else signal['action']
                    entry_price = signal['entry_price'].item() if hasattr(signal['entry_price'], 'item') else signal['entry_price']
                    stop_loss = signal['stop_loss'].item() if hasattr(signal['stop_loss'], 'item') else signal['stop_loss']
                    take_profit = signal['take_profit'].item() if hasattr(signal['take_profit'], 'item') else signal['take_profit']
                    
                    # Calculate profit/loss in pips
                    if outcome == 'success':
                        # Success means it hit take profit
                        if action == 'buy':
                            pip_diff = (take_profit - entry_price) * 10000
                        else:
                            pip_diff = (entry_price - take_profit) * 10000
                    else:
                        # Failure means it hit stop loss
                        if action == 'buy':
                            pip_diff = (stop_loss - entry_price) * 10000
                        else:
                            pip_diff = (entry_price - stop_loss) * 10000
                    
                    # Update the dataframe
                    performance_df.loc[idx, 'result'] = outcome
                    performance_df.loc[idx, 'profit_loss_pips'] = pip_diff
                    performance_df.loc[idx, 'close_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add new signals to tracking if they're not already there
    if not new_signals_df.empty:
        for _, signal in new_signals_df.iterrows():
            # Check if signal already exists in performance tracking
            signal_exists = False
            
            if not performance_df.empty:
                signal_exists = ((performance_df['pair'] == signal['pair']) & 
                                 (performance_df['timestamp'] == signal['timestamp'])).any()
            
            if not signal_exists:
                # Add to performance tracking with pending result
                # Convert Series values to Python scalar values if needed
                new_row = {
                    'pair': signal['pair'].item() if hasattr(signal['pair'], 'item') else signal['pair'],
                    'action': signal['action'].item() if hasattr(signal['action'], 'item') else signal['action'],
                    'entry_price': signal['entry_price'].item() if hasattr(signal['entry_price'], 'item') else signal['entry_price'],
                    'stop_loss': signal['stop_loss'].item() if hasattr(signal['stop_loss'], 'item') else signal['stop_loss'],
                    'take_profit': signal['take_profit'].item() if hasattr(signal['take_profit'], 'item') else signal['take_profit'],
                    'confidence': signal['confidence'].item() if hasattr(signal['confidence'], 'item') else signal['confidence'],
                    'timestamp': signal['timestamp'].item() if hasattr(signal['timestamp'], 'item') else signal['timestamp'],
                    'result': None,
                    'profit_loss_pips': None,
                    'close_timestamp': None
                }
                
                performance_df = pd.concat([performance_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated performance data
    save_performance_data(performance_df)
    
    return performance_df

def generate_initial_performance_history():
    """
    Generates initial performance history for demo purposes.
    
    Returns:
        pandas.DataFrame: Generated performance history
    """
    # For demo purposes, generate 30 days of historical data
    pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY"
    ]
    
    confidence_levels = ['unsicher', 'mittel', 'sicher']
    actions = ['buy', 'sell']
    
    history = []
    end_date = datetime.now()
    
    # Generate signals for the past 30 days
    for day in range(30):
        date = end_date - timedelta(days=day)
        num_signals = random.randint(2, 8)  # 2-8 signals per day
        
        for _ in range(num_signals):
            pair = random.choice(pairs)
            confidence = random.choice(confidence_levels)
            action = random.choice(actions)
            
            # Base price for the pair
            if 'JPY' in pair:
                base_price = random.uniform(100, 180)
            else:
                base_price = random.uniform(0.6, 1.5)
            
            # Generate realistic TP and SL
            if action == 'buy':
                stop_loss = base_price * 0.995  # 0.5% below entry
                take_profit = base_price * 1.015  # 1.5% above entry
            else:
                stop_loss = base_price * 1.005  # 0.5% above entry
                take_profit = base_price * 0.985  # 1.5% below entry
            
            # Generate signal timestamp
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            signal_timestamp = date.replace(hour=hour, minute=minute)
            
            # For old signals, determine outcome
            success_probability = 0.75  # 75% success rate overall
            
            # Adjust probability based on confidence
            if confidence == 'sicher':
                success_probability = 0.85
            elif confidence == 'mittel':
                success_probability = 0.7
            elif confidence == 'unsicher':
                success_probability = 0.55
            
            # Determine outcome
            outcome = 'success' if random.random() < success_probability else 'failure'
            
            # Calculate profit/loss in pips
            if outcome == 'success':
                # Success means it hit take profit
                if action == 'buy':
                    pip_diff = (take_profit - base_price) * 10000
                else:
                    pip_diff = (base_price - take_profit) * 10000
            else:
                # Failure means it hit stop loss
                if action == 'buy':
                    pip_diff = (stop_loss - base_price) * 10000
                else:
                    pip_diff = (base_price - stop_loss) * 10000
            
            # Generate close timestamp (6-12 hours after signal)
            hours_later = random.randint(6, 12)
            close_timestamp = signal_timestamp + timedelta(hours=hours_later)
            
            # Add to history
            history.append({
                'pair': pair,
                'action': action,
                'entry_price': base_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'timestamp': signal_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'result': outcome,
                'profit_loss_pips': pip_diff,
                'close_timestamp': close_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Save initial history
    save_performance_data(df)
    
    return df
