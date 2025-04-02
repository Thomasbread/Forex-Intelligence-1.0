import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from data_handler import get_forex_data, calculate_indicators, get_market_sentiment
from performance_tracker import update_performance

def generate_signals(available_pairs, max_signals=1):
    """
    Generate trading signals based on market analysis.
    
    Args:
        available_pairs (list): List of available currency pairs
        max_signals (int): Maximum number of signals to generate (default: 1)
        
    Returns:
        pandas.DataFrame: Generated trading signals
    """
    signals = []
    
    # Check when the last signal was generated (at least 5 minutes between signals)
    last_signal_time = datetime.now() - timedelta(minutes=5)
    if 'last_signal_time' in globals():
        time_since_last = datetime.now() - globals()['last_signal_time']
        if time_since_last.total_seconds() < 300:  # 300 seconds = 5 minutes
            # Not enough time has passed, return empty DataFrame
            return pd.DataFrame()
    
    # We'll only generate ONE highly confident signal (if conditions are met)
    # Shuffle pairs to analyze them in random order
    random.shuffle(available_pairs)
    
    # Find the best possible signal from all pairs
    best_signal = None
    best_confidence_score = 0
    
    for pair in available_pairs:
        # Get forex data for analysis (more data for better analysis)
        data = get_forex_data(pair, '1h', 200)
        
        if data.empty:
            continue
        
        # Calculate technical indicators
        data_with_indicators = calculate_indicators(data)
        
        # Get market sentiment
        sentiment = get_market_sentiment(pair)
        
        # Generate signal based on analysis
        signal = analyze_market(pair, data_with_indicators, sentiment, require_high_confidence=True)
        
        if signal:
            # Calculate confidence score
            confidence_score = 0
            if signal['confidence'] == 'sicher':
                confidence_score = 3
            elif signal['confidence'] == 'mittel':
                confidence_score = 2
            else:
                confidence_score = 1
                
            # Additional score from sentiment strength
            confidence_score += sentiment['trend_strength'] * 2
            
            # Keep the best signal
            if confidence_score > best_confidence_score:
                best_signal = signal
                best_confidence_score = confidence_score
    
    # Only add the signal if it's confident enough
    if best_signal and best_confidence_score >= 3:
        signals.append(best_signal)
        # Update the last signal time globally
        globals()['last_signal_time'] = datetime.now()
    
    # Convert to DataFrame
    if signals:
        signals_df = pd.DataFrame(signals)
        
        # Update performance for any old signals that have completed
        update_performance(signals_df)
        
        return signals_df
    else:
        return pd.DataFrame()

def analyze_market(pair, data, sentiment, require_high_confidence=False):
    """
    Analyze market data and generate a trading signal if conditions are met.
    
    Args:
        pair (str): Currency pair
        data (pandas.DataFrame): OHLC data with indicators
        sentiment (dict): Market sentiment data
        require_high_confidence (bool): If True, only return signals with high confidence
        
    Returns:
        dict: Trading signal if generated, None otherwise
    """
    # Check if we have enough data
    if len(data) < 50:
        return None
    
    # Get the most recent data
    recent_data = data.iloc[-10:].copy()
    latest = recent_data.iloc[-1]
    prev = recent_data.iloc[-2]
    
    # Current price
    current_price = latest['Close']
    
    # Determine if conditions are favorable for a signal
    signal_probability = random.uniform(0, 1)
    
    # Only generate a signal 70% of the time to simulate selective signals
    if signal_probability > 0.3:
        # Determine signal direction (buy/sell)
        # Use a combination of indicators and sentiment
        
        # MACD Signal
        macd_signal = 0
        if 'MACD' in latest.index and 'MACD_Signal' in latest.index:
            try:
                # Convert Series to scalar values using .item()
                latest_macd = latest.loc['MACD'].item() if hasattr(latest.loc['MACD'], 'item') else latest.loc['MACD']
                latest_macd_signal = latest.loc['MACD_Signal'].item() if hasattr(latest.loc['MACD_Signal'], 'item') else latest.loc['MACD_Signal']
                prev_macd = prev.loc['MACD'].item() if hasattr(prev.loc['MACD'], 'item') else prev.loc['MACD']
                prev_macd_signal = prev.loc['MACD_Signal'].item() if hasattr(prev.loc['MACD_Signal'], 'item') else prev.loc['MACD_Signal']
                
                if latest_macd > latest_macd_signal and prev_macd <= prev_macd_signal:
                    macd_signal = 1  # Bullish crossover
                elif latest_macd < latest_macd_signal and prev_macd >= prev_macd_signal:
                    macd_signal = -1  # Bearish crossover
            except Exception as e:
                print(f"Error calculating MACD signal: {e}")
                # Safely continue without setting the signal
        
        # RSI Signal
        rsi_signal = 0
        if 'RSI' in latest.index:
            try:
                rsi_value = latest.loc['RSI'].item() if hasattr(latest.loc['RSI'], 'item') else latest.loc['RSI']
                if rsi_value < 30:
                    rsi_signal = 1  # Oversold
                elif rsi_value > 70:
                    rsi_signal = -1  # Overbought
            except Exception as e:
                print(f"Error calculating RSI signal: {e}")
        
        # Moving Average Signal
        ma_signal = 0
        if 'SMA_20' in latest.index and 'SMA_50' in latest.index:
            try:
                latest_sma20 = latest.loc['SMA_20'].item() if hasattr(latest.loc['SMA_20'], 'item') else latest.loc['SMA_20']
                latest_sma50 = latest.loc['SMA_50'].item() if hasattr(latest.loc['SMA_50'], 'item') else latest.loc['SMA_50']
                prev_sma20 = prev.loc['SMA_20'].item() if hasattr(prev.loc['SMA_20'], 'item') else prev.loc['SMA_20']
                prev_sma50 = prev.loc['SMA_50'].item() if hasattr(prev.loc['SMA_50'], 'item') else prev.loc['SMA_50']
                
                if latest_sma20 > latest_sma50 and prev_sma20 <= prev_sma50:
                    ma_signal = 1  # Bullish crossover
                elif latest_sma20 < latest_sma50 and prev_sma20 >= prev_sma50:
                    ma_signal = -1  # Bearish crossover
            except Exception as e:
                print(f"Error calculating MA signal: {e}")
        
        # Combine technical signals
        technical_signal = macd_signal + rsi_signal + ma_signal
        
        # Combine with sentiment
        overall_signal = technical_signal + (1 if sentiment['bullish_probability'] > 0.6 else 0) - (1 if sentiment['bearish_probability'] > 0.6 else 0)
        
        # Determine action
        action = 'buy' if overall_signal > 0 else 'sell'
        
        # Stop loss and take profit calculation
        if action == 'buy':
            # For buy: SL below recent low, TP at 3x the risk
            recent_low = recent_data['Low'].min()
            stop_loss = recent_low * 0.998  # Slightly below the low
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 3)  # 3:1 reward-to-risk ratio
        else:
            # For sell: SL above recent high, TP at 3x the risk
            recent_high = recent_data['High'].max()
            stop_loss = recent_high * 1.002  # Slightly above the high
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 3)  # 3:1 reward-to-risk ratio
        
        # Calculate confidence level based on signal strength and market conditions
        signal_strength = abs(overall_signal)
        trend_strength = sentiment['trend_strength']
        
        combined_confidence = (signal_strength / 3) * 0.5 + trend_strength * 0.5
        
        if combined_confidence > 0.7:
            confidence = 'sicher'
        elif combined_confidence > 0.4:
            confidence = 'mittel'
        else:
            confidence = 'unsicher'
        
        # Generate analysis text
        analysis = generate_analysis_text(pair, action, data, sentiment, signal_strength)
        
        # Check if we need high confidence signals only
        if require_high_confidence and confidence != 'sicher':
            return None
            
        # Create signal
        return {
            'pair': pair,
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': 3,  # Fixed at 1:3 risk-reward
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis': analysis
        }
    
    return None

def generate_analysis_text(pair, action, data, sentiment, signal_strength):
    """
    Generate analysis text explaining the trading signal.
    
    Args:
        pair (str): Currency pair
        action (str): Trading action (buy/sell)
        data (pandas.DataFrame): OHLC data with indicators
        sentiment (dict): Market sentiment
        signal_strength (float): Strength of the signal
        
    Returns:
        str: Analysis text
    """
    latest = data.iloc[-1]
    
    # Technical indicator descriptions
    macd_text = ""
    if 'MACD' in latest.index and 'MACD_Signal' in latest.index:
        try:
            latest_macd = latest.loc['MACD'].item() if hasattr(latest.loc['MACD'], 'item') else latest.loc['MACD']
            latest_macd_signal = latest.loc['MACD_Signal'].item() if hasattr(latest.loc['MACD_Signal'], 'item') else latest.loc['MACD_Signal']
            
            if latest_macd > latest_macd_signal:
                macd_text = "MACD ist über der Signallinie, was bullische Momentum anzeigt. "
            else:
                macd_text = "MACD ist unter der Signallinie, was bärisches Momentum anzeigt. "
        except Exception as e:
            print(f"Error generating MACD text: {e}")
            macd_text = "MACD-Analyse nicht verfügbar. "
    
    rsi_text = ""
    if 'RSI' in latest.index:
        try:
            rsi_value = latest.loc['RSI'].item() if hasattr(latest.loc['RSI'], 'item') else latest.loc['RSI']
            if rsi_value < 30:
                rsi_text = f"RSI bei {rsi_value:.1f} zeigt überkaufte Bedingungen an. "
            elif rsi_value > 70:
                rsi_text = f"RSI bei {rsi_value:.1f} zeigt überverkaufte Bedingungen an. "
            else:
                rsi_text = f"RSI bei {rsi_value:.1f} zeigt neutrales Momentum an. "
        except Exception as e:
            print(f"Error generating RSI text: {e}")
            rsi_text = "RSI-Analyse nicht verfügbar. "
    
    ma_text = ""
    if 'SMA_20' in latest.index and 'SMA_50' in latest.index:
        try:
            latest_sma20 = latest.loc['SMA_20'].item() if hasattr(latest.loc['SMA_20'], 'item') else latest.loc['SMA_20']
            latest_sma50 = latest.loc['SMA_50'].item() if hasattr(latest.loc['SMA_50'], 'item') else latest.loc['SMA_50']
            
            if latest_sma20 > latest_sma50:
                ma_text = "Die 20-Perioden-MA liegt über der 50-Perioden-MA, was einen bullischen Trend anzeigt. "
            else:
                ma_text = "Die 20-Perioden-MA liegt unter der 50-Perioden-MA, was einen bärischen Trend anzeigt. "
        except Exception as e:
            print(f"Error generating MA text: {e}")
            ma_text = "MA-Analyse nicht verfügbar. "
    
    # Sentiment text
    sentiment_text = ""
    if sentiment['bullish_probability'] > 0.6:
        sentiment_text = "Marktsentiment ist überwiegend bullisch. "
    elif sentiment['bearish_probability'] > 0.6:
        sentiment_text = "Marktsentiment ist überwiegend bärisch. "
    else:
        sentiment_text = "Marktsentiment ist gemischt. "
    
    # Trend strength
    trend_text = f"Trendstärke ist {sentiment['trend_strength']*100:.1f}% "
    
    # Volatility
    volatility_text = ""
    if sentiment['volatility'] > 0.6:
        volatility_text = "mit hoher Volatilität. "
    elif sentiment['volatility'] > 0.3:
        volatility_text = "mit moderater Volatilität. "
    else:
        volatility_text = "mit niedriger Volatilität. "
    
    # Action reason
    action_text = ""
    if action == 'buy':
        action_text = (
            f"Basierend auf der technischen Analyse und dem Marktsentiment empfiehlt die KI "
            f"einen KAUF für {pair}. Die Kombination aus {macd_text.lower()}{rsi_text.lower()}"
            f"und {ma_text.lower()}unterstützt diese Entscheidung. "
            f"{sentiment_text}{trend_text}{volatility_text}"
        )
    else:
        action_text = (
            f"Basierend auf der technischen Analyse und dem Marktsentiment empfiehlt die KI "
            f"einen VERKAUF für {pair}. Die Kombination aus {macd_text.lower()}{rsi_text.lower()}"
            f"und {ma_text.lower()}unterstützt diese Entscheidung. "
            f"{sentiment_text}{trend_text}{volatility_text}"
        )
    
    return action_text
