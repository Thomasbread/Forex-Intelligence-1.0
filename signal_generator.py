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
        max_signals (int): Maximum number of signals to generate (Default: 1)
        
    Returns:
        pandas.DataFrame: Generated trading signals
    """
    signals = []
    best_signal = None
    best_confidence_score = 0
    
    # Analyze all pairs but only keep the best signal
    for pair in available_pairs:
        # Get forex data for analysis
        data = get_forex_data(pair, '1h', 100)
        
        if data.empty:
            continue
        
        # Calculate technical indicators
        data_with_indicators = calculate_indicators(data)
        
        # Get market sentiment
        sentiment = get_market_sentiment(pair)
        
        # Generate signal based on analysis
        signal = analyze_market(pair, data_with_indicators, sentiment)
        
        if signal:
            # Calculate a confidence score for comparison
            confidence_score = 0
            if signal['confidence'] == 'sicher':
                confidence_score = 3
            elif signal['confidence'] == 'mittel':
                confidence_score = 2
            else:
                confidence_score = 1
                
            # Add randomness to avoid always selecting the same pair
            confidence_score += random.uniform(0, 0.5)
            
            # If this is the best signal so far, keep it
            if confidence_score > best_confidence_score:
                best_signal = signal
                best_confidence_score = confidence_score
    
    # Only use the signal if it's "sicher" (secure)
    if best_signal and best_signal['confidence'] == 'sicher':
        signals.append(best_signal)
    
    # Convert to DataFrame
    if signals:
        signals_df = pd.DataFrame(signals)
        
        # Update performance for any old signals that have completed
        update_performance(signals_df)
        
        return signals_df
    else:
        return pd.DataFrame()

def analyze_market(pair, data, sentiment):
    """
    Analyze market data and generate a trading signal if conditions are met.
    Advanced version with higher accuracy requirements and more confirmations.
    
    Args:
        pair (str): Currency pair
        data (pandas.DataFrame): OHLC data with indicators
        sentiment (dict): Market sentiment data
        
    Returns:
        dict: Trading signal if generated, None otherwise
    """
    # Check if we have enough data
    if len(data) < 50:
        return None
    
    # Get the most recent data
    recent_data = data.iloc[-20:].copy()
    latest = recent_data.iloc[-1]
    prev = recent_data.iloc[-2]
    prev3 = recent_data.iloc[-3] if len(recent_data) >= 3 else prev
    
    # Current price
    current_price = latest['Close']
    
    # Determine if market conditions are suitable for a trade
    # Much more selective now - only 30% of potential signals will be considered
    signal_probability = random.uniform(0, 1)
    
    # Only proceed if the random filter passes (more selective)
    if signal_probability > 0.7:  
        # Initialize all signal components
        signal_components = {
            'macd': 0,
            'rsi': 0,
            'ma_cross': 0,
            'ma_trend': 0,
            'volume': 0,
            'bollinger': 0,
            'price_action': 0,
            'sentiment': 0
        }
        
        # 1. MACD Signal (stronger confirmation required)
        if 'MACD' in latest.index and 'MACD_Signal' in latest.index:
            try:
                # Convert Series to scalar values using .item()
                latest_macd = latest.loc['MACD'].item() if hasattr(latest.loc['MACD'], 'item') else latest.loc['MACD']
                latest_macd_signal = latest.loc['MACD_Signal'].item() if hasattr(latest.loc['MACD_Signal'], 'item') else latest.loc['MACD_Signal']
                prev_macd = prev.loc['MACD'].item() if hasattr(prev.loc['MACD'], 'item') else prev.loc['MACD']
                prev_macd_signal = prev.loc['MACD_Signal'].item() if hasattr(prev.loc['MACD_Signal'], 'item') else prev.loc['MACD_Signal']
                prev3_macd = prev3.loc['MACD'].item() if hasattr(prev3.loc['MACD'], 'item') else prev3.loc['MACD']
                
                # Require stronger confirmation - fresh crossover + increasing momentum
                if latest_macd > latest_macd_signal and prev_macd <= prev_macd_signal:
                    if latest_macd > prev_macd:  # Increasing momentum
                        signal_components['macd'] = 2  # Strong bullish signal
                    else:
                        signal_components['macd'] = 1  # Bullish but weak momentum
                elif latest_macd < latest_macd_signal and prev_macd >= prev_macd_signal:
                    if latest_macd < prev_macd:  # Increasing downward momentum
                        signal_components['macd'] = -2  # Strong bearish signal
                    else:
                        signal_components['macd'] = -1  # Bearish but weak momentum
                # Consider MACD histogram trend without crossover
                elif latest_macd - latest_macd_signal > prev_macd - prev_macd_signal:
                    signal_components['macd'] = 0.5  # Bullish potential developing
                elif latest_macd - latest_macd_signal < prev_macd - prev_macd_signal:
                    signal_components['macd'] = -0.5  # Bearish potential developing
            except Exception as e:
                print(f"Error calculating MACD signal: {e}")
        
        # 2. RSI Signal (more nuanced with extreme levels)
        if 'RSI' in latest.index:
            try:
                rsi_value = latest.loc['RSI'].item() if hasattr(latest.loc['RSI'], 'item') else latest.loc['RSI']
                prev_rsi = prev.loc['RSI'].item() if hasattr(prev.loc['RSI'], 'item') else prev.loc['RSI']
                
                # Extreme oversold/overbought conditions 
                if rsi_value < 20:
                    signal_components['rsi'] = 2  # Extremely oversold - strong buy
                elif rsi_value < 30:
                    signal_components['rsi'] = 1  # Oversold - buy
                elif rsi_value > 80:
                    signal_components['rsi'] = -2  # Extremely overbought - strong sell
                elif rsi_value > 70:
                    signal_components['rsi'] = -1  # Overbought - sell
                
                # RSI trend - reversals from extremes are powerful signals
                if rsi_value > prev_rsi and prev_rsi < 30:
                    signal_components['rsi'] += 1  # Bullish RSI reversal from oversold
                elif rsi_value < prev_rsi and prev_rsi > 70:
                    signal_components['rsi'] -= 1  # Bearish RSI reversal from overbought
                
                # Divergence (basic check - more complex in real systems)
                if rsi_value > prev_rsi and current_price < recent_data.iloc[-2]['Close']:
                    signal_components['rsi'] -= 0.5  # Bearish divergence (price down, RSI up)
                elif rsi_value < prev_rsi and current_price > recent_data.iloc[-2]['Close']:
                    signal_components['rsi'] += 0.5  # Bullish divergence (price up, RSI down)
            except Exception as e:
                print(f"Error calculating RSI signal: {e}")
        
        # 3. Moving Average Signals (multiple timeframes)
        if 'SMA_20' in latest.index and 'SMA_50' in latest.index:
            try:
                latest_sma20 = latest.loc['SMA_20'].item() if hasattr(latest.loc['SMA_20'], 'item') else latest.loc['SMA_20']
                latest_sma50 = latest.loc['SMA_50'].item() if hasattr(latest.loc['SMA_50'], 'item') else latest.loc['SMA_50']
                prev_sma20 = prev.loc['SMA_20'].item() if hasattr(prev.loc['SMA_20'], 'item') else prev.loc['SMA_20']
                prev_sma50 = prev.loc['SMA_50'].item() if hasattr(prev.loc['SMA_50'], 'item') else prev.loc['SMA_50']
                
                # MA Crossovers
                if latest_sma20 > latest_sma50 and prev_sma20 <= prev_sma50:
                    signal_components['ma_cross'] = 2  # Fresh bullish crossover
                elif latest_sma20 < latest_sma50 and prev_sma20 >= prev_sma50:
                    signal_components['ma_cross'] = -2  # Fresh bearish crossover
                
                # MA Trend
                ma_gap_percent = abs(latest_sma20 - latest_sma50) / latest_sma50 * 100
                if latest_sma20 > latest_sma50:
                    # Calculate slope of MA
                    ma20_slope = (latest_sma20 - prev_sma20) / prev_sma20 * 100
                    if ma20_slope > 0.1:  # Significant upward slope
                        signal_components['ma_trend'] = 1
                        if ma_gap_percent > 0.3:  # Widening gap
                            signal_components['ma_trend'] += 0.5
                elif latest_sma20 < latest_sma50:
                    ma20_slope = (latest_sma20 - prev_sma20) / prev_sma20 * 100
                    if ma20_slope < -0.1:  # Significant downward slope
                        signal_components['ma_trend'] = -1
                        if ma_gap_percent > 0.3:  # Widening gap
                            signal_components['ma_trend'] -= 0.5
            except Exception as e:
                print(f"Error calculating MA signals: {e}")
        
        # 4. Bollinger Bands signals
        if 'BB_Upper' in latest.index and 'BB_Lower' in latest.index:
            try:
                bb_upper = latest.loc['BB_Upper'].item() if hasattr(latest.loc['BB_Upper'], 'item') else latest.loc['BB_Upper']
                bb_lower = latest.loc['BB_Lower'].item() if hasattr(latest.loc['BB_Lower'], 'item') else latest.loc['BB_Lower']
                bb_middle = latest.loc['BB_Middle'].item() if hasattr(latest.loc['BB_Middle'], 'item') else latest.loc['BB_Middle']
                
                # Price relative to bands
                if current_price <= bb_lower:
                    signal_components['bollinger'] = 1.5  # Price at or below lower band - buy signal
                elif current_price >= bb_upper:
                    signal_components['bollinger'] = -1.5  # Price at or above upper band - sell signal
                
                # Band width - volatility indicator
                band_width = (bb_upper - bb_lower) / bb_middle * 100
                # Narrowing bands (low volatility) often precede breakouts
                if band_width < 1.0:  # Very narrow bands
                    # Neutral but important to note for timing
                    signal_components['bollinger'] += 0.5 if current_price > bb_middle else -0.5
            except Exception as e:
                print(f"Error calculating Bollinger Band signals: {e}")
                
        # 5. Volume analysis 
        if 'Volume' in recent_data.columns:
            try:
                # Calculate average volume for reference
                avg_volume = recent_data['Volume'].mean()
                latest_volume = recent_data.iloc[-1]['Volume']
                
                # Volume spikes often confirm price moves
                if latest_volume > avg_volume * 1.5:  # 50% above average
                    # High volume confirms the direction
                    price_change = current_price - recent_data.iloc[-2]['Close']
                    if price_change > 0:
                        signal_components['volume'] = 1  # Bullish volume confirmation
                    elif price_change < 0:
                        signal_components['volume'] = -1  # Bearish volume confirmation
            except Exception as e:
                print(f"Error calculating volume signals: {e}")
                
        # 6. Price action patterns
        try:
            # Check for bullish/bearish engulfing patterns
            curr_candle_size = abs(current_price - recent_data.iloc[-1]['Open'])
            prev_candle_size = abs(recent_data.iloc[-2]['Close'] - recent_data.iloc[-2]['Open'])
            curr_bullish = current_price > recent_data.iloc[-1]['Open']
            prev_bullish = recent_data.iloc[-2]['Close'] > recent_data.iloc[-2]['Open']
            
            # Engulfing pattern
            if curr_candle_size > prev_candle_size:
                if curr_bullish and not prev_bullish:
                    signal_components['price_action'] = 1.5  # Bullish engulfing
                elif not curr_bullish and prev_bullish:
                    signal_components['price_action'] = -1.5  # Bearish engulfing
            
            # Recent consecutive candles in one direction
            bullish_count = 0
            bearish_count = 0
            for i in range(1, min(5, len(recent_data))):
                if recent_data.iloc[-i]['Close'] > recent_data.iloc[-i]['Open']:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            if bullish_count >= 3:
                signal_components['price_action'] += 0.5  # Strong bullish momentum
            elif bearish_count >= 3:
                signal_components['price_action'] -= 0.5  # Strong bearish momentum
        except Exception as e:
            print(f"Error calculating price action signals: {e}")
            
        # 7. Market sentiment analysis (enhanced)
        bull_probability = sentiment['bullish_probability']
        bear_probability = sentiment['bearish_probability']
        trend_strength = sentiment['trend_strength']
        volatility = sentiment['volatility']
        
        # Stronger sentiment influence for higher confidence
        if bull_probability > 0.7:  # Very bullish sentiment
            signal_components['sentiment'] = 2
        elif bull_probability > 0.6:
            signal_components['sentiment'] = 1
        elif bear_probability > 0.7:  # Very bearish sentiment
            signal_components['sentiment'] = -2
        elif bear_probability > 0.6:
            signal_components['sentiment'] = -1
            
        # Volatility adjustment - high volatility increases risk
        if volatility > 0.6:  # High volatility
            for key in signal_components:
                if key != 'sentiment':
                    # Dampen signals in high volatility periods
                    signal_components[key] *= 0.8
                    
        # Calculate overall signal with weighted components
        # MACD and MA crossovers have higher weight as they're trend indicators
        component_weights = {
            'macd': 1.5,
            'rsi': 1.2,
            'ma_cross': 1.5,
            'ma_trend': 1.0,
            'volume': 0.8,
            'bollinger': 1.2,
            'price_action': 1.3,
            'sentiment': 1.0
        }
        
        weighted_signals = {key: signal_components[key] * component_weights[key] for key in signal_components}
        overall_signal = sum(weighted_signals.values())
        
        # Determine action - needs stronger conviction now
        if overall_signal > 3:  # Require stronger buy signal
            action = 'buy'
        elif overall_signal < -3:  # Require stronger sell signal
            action = 'sell'
        else:
            return None  # Not strong enough - no signal
            
        # Calculate entry timing
        entry_timing = "Sofort"  # Default is immediate entry
        
        # For timing, we consider:
        # 1. Time of day (market sessions)
        # 2. Pending price action confirmation
        current_hour = datetime.now().hour
        
        # Major forex session times (rough approximations)
        asian_session = 0 <= current_hour < 8
        london_session = 8 <= current_hour < 16
        us_session = 13 <= current_hour < 21
        
        # Adjust timing based on sessions and signal type
        if asian_session and not (london_session or us_session):
            # Asian session often has less volatility
            if abs(overall_signal) < 5:  # Not an extremely strong signal
                entry_timing = "Warten auf London-Session (ab 8:00 Uhr)"
                
        # Also consider price action for timing
        if signal_components['bollinger'] != 0 and abs(signal_components['bollinger']) > 1:
            # Bollinger band extremes might suggest waiting for reversion
            if (action == 'buy' and signal_components['bollinger'] < 0) or \
               (action == 'sell' and signal_components['bollinger'] > 0):
                entry_timing = "Einstieg bei Preisbestätigung (Kerzenschluss)"
                
        # If RSI is extreme, might need confirmation
        if abs(signal_components['rsi']) > 1.5:
            entry_timing = "Nach RSI-Bestätigung"
            
        # Stop loss and take profit calculation (with better placement)
        if action == 'buy':
            # For buy: SL based on recent swing lows plus buffer
            # Look for significant recent lows in data, not just most recent
            recent_lows = recent_data['Low'].nsmallest(3).values
            stop_loss = min(recent_lows) * 0.998  # Slightly below significant low
            
            # Ensure SL isn't too far from entry - limit risk
            max_risk_percent = 0.5  # Maximum 0.5% risk
            if (current_price - stop_loss) / current_price > max_risk_percent/100:
                stop_loss = current_price * (1 - max_risk_percent/100)
                
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 3)  # 3:1 reward-to-risk ratio
        else:
            # For sell: SL based on recent swing highs plus buffer
            recent_highs = recent_data['High'].nlargest(3).values
            stop_loss = max(recent_highs) * 1.002  # Slightly above significant high
            
            # Ensure SL isn't too far from entry - limit risk
            max_risk_percent = 0.5  # Maximum 0.5% risk
            if (stop_loss - current_price) / current_price > max_risk_percent/100:
                stop_loss = current_price * (1 + max_risk_percent/100)
                
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 3)  # 3:1 reward-to-risk ratio
        
        # Estimate trade duration
        # Based on volatility, pair, and price targets
        volatility_factor = sentiment['volatility']
        
        # Calculate approximate pips to target
        pips_to_target = abs(take_profit - current_price) * 10000  # Convert to pips
        
        # Base duration on volatility and distance
        if 'JPY' in pair:  # JPY pairs move differently
            pips_to_target = pips_to_target / 100  # Adjust for JPY denomination
            
        # Estimated hours based on volatility and distance
        est_hours = pips_to_target / (5 + volatility_factor * 20)
        
        # Convert to a human-readable format
        if est_hours < 24:
            trade_duration = f"~{int(est_hours)} Stunden"
        else:
            est_days = est_hours / 24
            trade_duration = f"~{int(est_days)} Tage"
            
        # Add variability based on market conditions
        if volatility_factor < 0.3:  # Low volatility
            trade_duration += " (bei niedriger Volatilität evtl. länger)"
        elif volatility_factor > 0.7:  # High volatility
            trade_duration += " (bei hoher Volatilität evtl. kürzer)"
        
        # Calculate confidence level based on signal strength and confirmations
        # Much stricter confidence calculation
        signal_strength = abs(overall_signal)
        
        # Count how many signal components agree with the overall direction
        agreeing_components = sum(1 for key, value in signal_components.items() 
                                if (action == 'buy' and value > 0) or 
                                   (action == 'sell' and value < 0))
        
        # Stronger confidence requirements
        if signal_strength > 6 and agreeing_components >= 5:
            confidence = 'sicher'
        elif signal_strength > 4 and agreeing_components >= 4:
            confidence = 'mittel'
        else:
            confidence = 'unsicher'
            
        # Only return 'sicher' signals
        if confidence != 'sicher':
            return None
        
        # Generate detailed analysis text
        # Use the original analysis function for backward compatibility
        # We'll add our additional information manually
        base_analysis = generate_analysis_text(pair, action, data, sentiment, signal_strength)
        
        # Define direction_text for our analysis template
        direction_text = "bullisch" if action == "buy" else "bearisch"
        
        # Create enhanced analysis with additional details
        analysis = f"""**{pair} {direction_text.upper()} SIGNAL - HOHE KONFIDENZ**

{base_analysis}

**Optimale Eintrittszeit:** {entry_timing}
**Geschätzte Tradedauer:** {trade_duration}

**Risikofaktoren zu beachten:**
• Unerwartete wirtschaftliche Ereignisse oder Nachrichten könnten die Preisentwicklung beeinflussen.
• Hohe Volatilität könnte zu Kursausbrüchen führen.
"""
        
        # Create enhanced signal with new fields
        return {
            'pair': pair,
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': 3,  # Fixed at 1:3 risk-reward
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'entry_timing': entry_timing,
            'estimated_duration': trade_duration,
            'analysis': analysis
        }
    
    return None

def generate_enhanced_analysis(pair, action, data, signal_components, weighted_signals, sentiment, signal_strength, entry_timing, trade_duration):
    """
    Generate detailed analysis text explaining the trading signal with multiple factor considerations.
    
    Args:
        pair (str): Currency pair
        action (str): Trading action (buy/sell)
        data (pandas.DataFrame): OHLC data with indicators
        signal_components (dict): Individual signal component values
        weighted_signals (dict): Signal components with weights applied
        sentiment (dict): Market sentiment data
        signal_strength (float): Overall strength of the signal
        entry_timing (str): Recommended entry timing
        trade_duration (str): Estimated trade duration
        
    Returns:
        str: Detailed analysis text
    """
    # Recent price action
    recent_data = data.iloc[-5:].copy()
    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0] * 100
    
    # Generate analysis based on indicators and sentiment
    action_text = "Kaufen" if action == "buy" else "Verkaufen"
    direction_text = "bullisch" if action == "buy" else "bearisch"
    
    # Format price change
    price_change_text = f"+{price_change:.2f}%" if price_change >= 0 else f"{price_change:.2f}%"
    
    # Check technical indicators
    rsi_value = 0
    macd_value = 0
    bb_position = ""
    
    # Get RSI value if available
    if 'RSI' in recent_data.columns or 'RSI' in recent_data.index:
        try:
            # Handle both DataFrame and Series representations
            if isinstance(recent_data, pd.DataFrame) and 'RSI' in recent_data.columns:
                rsi_value = recent_data['RSI'].iloc[-1]
            else:
                rsi_value = recent_data.iloc[-1].loc['RSI'].item() if hasattr(recent_data.iloc[-1].loc['RSI'], 'item') else recent_data.iloc[-1].loc['RSI']
        except Exception as e:
            print(f"Error accessing RSI value: {e}")
    
    # Check MACD status
    if 'MACD' in recent_data.index or ('MACD' in recent_data.columns and 'MACD_Signal' in recent_data.columns):
        try:
            if 'MACD' in recent_data.index:
                macd_value = recent_data.iloc[-1].loc['MACD'].item() if hasattr(recent_data.iloc[-1].loc['MACD'], 'item') else recent_data.iloc[-1].loc['MACD']
                macd_signal = recent_data.iloc[-1].loc['MACD_Signal'].item() if hasattr(recent_data.iloc[-1].loc['MACD_Signal'], 'item') else recent_data.iloc[-1].loc['MACD_Signal']
            else:
                macd_value = recent_data['MACD'].iloc[-1]
                macd_signal = recent_data['MACD_Signal'].iloc[-1]
                
            macd_status = "bullisch über Signallinie" if macd_value > macd_signal else "bearisch unter Signallinie"
        except Exception as e:
            print(f"Error accessing MACD values: {e}")
            macd_status = "nicht verfügbar"
    else:
        macd_status = "nicht verfügbar"
    
    # Check Bollinger Band position
    if all(band in recent_data.index for band in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        try:
            bb_upper = recent_data.iloc[-1].loc['BB_Upper'].item() if hasattr(recent_data.iloc[-1].loc['BB_Upper'], 'item') else recent_data.iloc[-1].loc['BB_Upper']
            bb_lower = recent_data.iloc[-1].loc['BB_Lower'].item() if hasattr(recent_data.iloc[-1].loc['BB_Lower'], 'item') else recent_data.iloc[-1].loc['BB_Lower']
            bb_middle = recent_data.iloc[-1].loc['BB_Middle'].item() if hasattr(recent_data.iloc[-1].loc['BB_Middle'], 'item') else recent_data.iloc[-1].loc['BB_Middle']
            
            current_price = recent_data['Close'].iloc[-1]
            
            if current_price > bb_upper:
                bb_position = "über dem oberen Band (überkauft)"
            elif current_price < bb_lower:
                bb_position = "unter dem unteren Band (überverkauft)"
            elif current_price > bb_middle:
                bb_position = "im oberen Bereich der Bänder"
            else:
                bb_position = "im unteren Bereich der Bänder"
        except Exception as e:
            print(f"Error determining Bollinger Band position: {e}")
    
    # Start building the analysis text
    # Introduction with overall assessment
    analysis_text = f"**{pair} {direction_text.upper()} SIGNAL - HOHE KONFIDENZ**\n\n"
    analysis_text += f"Der {pair} zeigt ein starkes {direction_text} Signal mit einer Preisänderung von {price_change_text} in den letzten Perioden.\n\n"
    
    # List the most significant factors
    analysis_text += "**Signalstärke nach Faktoren:**\n"
    
    # Sort signal components by absolute weighted strength
    sorted_components = sorted(
        [(k, v, weighted_signals[k]) for k, v in signal_components.items() if v != 0],
        key=lambda x: abs(x[2]),
        reverse=True
    )
    
    # Add key signal components (max 4 most important)
    for i, (component, value, weighted) in enumerate(sorted_components[:4]):
        # Translate component names
        component_names = {
            'macd': 'MACD',
            'rsi': 'RSI',
            'ma_cross': 'MA Kreuzung',
            'ma_trend': 'MA Trend',
            'volume': 'Volumen',
            'bollinger': 'Bollinger Bänder',
            'price_action': 'Preisaktions-Muster',
            'sentiment': 'Marktsentiment'
        }
        
        direction = "Bullish" if value > 0 else "Bearish"
        strength = "stark" if abs(value) > 1 else "moderat"
        component_name = component_names.get(component, component)
        
        analysis_text += f"• {component_name}: {direction} ({strength})\n"
    
    analysis_text += "\n**Detaillierte Analyse:**\n"
    
    # Add RSI analysis
    if rsi_value > 0:
        if rsi_value > 70:
            rsi_text = f"Der RSI liegt bei {rsi_value:.1f} und zeigt überkaufte Bedingungen"
        elif rsi_value < 30:
            rsi_text = f"Der RSI liegt bei {rsi_value:.1f} und zeigt überverkaufte Bedingungen"
        else:
            rsi_text = f"Der RSI liegt bei {rsi_value:.1f} im neutralen Bereich"
            
        if (action == 'buy' and rsi_value < 40) or (action == 'sell' and rsi_value > 60):
            rsi_text += f", was den {action_text}-Signal unterstützt"
            
        analysis_text += f"• {rsi_text}.\n"
    
    # Add MACD analysis
    if macd_status != "nicht verfügbar":
        analysis_text += f"• MACD ist {macd_status}, "
        if (action == 'buy' and "bullisch" in macd_status) or (action == 'sell' and "bearisch" in macd_status):
            analysis_text += f"was den {action_text}-Signal bestätigt.\n"
        else:
            analysis_text += "was auf möglichen Momentum-Wechsel hinweist.\n"
    
    # Add Bollinger Bands analysis if available
    if bb_position:
        analysis_text += f"• Der Preis befindet sich {bb_position} der Bollinger Bänder"
        
        # Interpret based on band position and action
        if (action == 'buy' and ("unteren" in bb_position or "überverkauft" in bb_position)) or \
           (action == 'sell' and ("oberen" in bb_position or "überkauft" in bb_position)):
            analysis_text += f", was den {action_text}-Signal unterstützt.\n"
        else:
            analysis_text += ".\n"
    
    # Add sentiment analysis
    bull_prob = sentiment['bullish_probability'] * 100
    bear_prob = sentiment['bearish_probability'] * 100
    trend_strength = sentiment['trend_strength'] * 100
    
    analysis_text += f"• Marktsentiment: {bull_prob:.1f}% bullisch, {bear_prob:.1f}% bearisch, mit einer Trendstärke von {trend_strength:.1f}%.\n"
    
    # Add volatility assessment
    volatility = sentiment['volatility'] * 100
    volatility_text = "hoch" if volatility > 50 else "moderat" if volatility > 25 else "niedrig"
    analysis_text += f"• Aktuelle Marktvolatilität: {volatility_text} ({volatility:.1f}%).\n\n"
    
    # Add entry and timing guidance
    analysis_text += f"**Handelsempfehlung:**\n"
    analysis_text += f"• Aktion: {action_text} bei aktuellen Preisen.\n"
    analysis_text += f"• Optimale Eintrittszeit: {entry_timing}\n"
    analysis_text += f"• Geschätzte Tradedauer: {trade_duration}\n\n"
    
    # Potential risk factors
    analysis_text += "**Risikofaktoren zu beachten:**\n"
    
    # Add specific risk factors based on the signal and market conditions
    if action == 'buy':
        if volatility > 50:
            analysis_text += "• Hohe Volatilität könnte zu Fehlsignalen führen.\n"
        if "bearisch" in macd_status and action == 'buy':
            analysis_text += "• MACD zeigt weiterhin negative Dynamik, was gegen den Kauf sprechen könnte.\n"
        if bear_prob > 40:
            analysis_text += f"• Beträchtliche bearische Sentiment ({bear_prob:.1f}%) bleibt bestehen.\n"
    else:  # sell
        if volatility > 50:
            analysis_text += "• Hohe Volatilität könnte zu Fehlsignalen führen.\n"
        if "bullisch" in macd_status and action == 'sell':
            analysis_text += "• MACD zeigt weiterhin positive Dynamik, was gegen den Verkauf sprechen könnte.\n"
        if bull_prob > 40:
            analysis_text += f"• Beträchtliche bullische Sentiment ({bull_prob:.1f}%) bleibt bestehen.\n"
    
    # Always add this generic risk warning
    analysis_text += "• Unerwartete wirtschaftliche Ereignisse oder Nachrichten könnten die Preisentwicklung beeinflussen.\n"
    
    return analysis_text


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
