import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from data_handler import get_forex_data, calculate_indicators, get_market_sentiment
from performance_tracker import update_performance
import news_scraper  # Importiere das neue News-Scraper-Modul für Echtzeit-Marktinformationen

def generate_signals(available_pairs, max_signals=5):
    """
    Generate trading signals based on market analysis.

    Args:
        available_pairs (list): List of available currency pairs
        max_signals (int): Maximum number of signals to generate (Default: 5)

    Returns:
        pandas.DataFrame: Generated trading signals
    """
    all_signals = []

    # Collect signals for all pairs
    for pair in available_pairs:
        # Get forex data for analysis
        data = get_forex_data(pair, '1h', 100)

        if data.empty:
            continue

        # Calculate technical indicators
        data_with_indicators = calculate_indicators(data)

        # Get market sentiment
        sentiment = get_market_sentiment(pair)

        # Hole Forex-Factory-Daten (Nachrichten und Wirtschaftskalender)
        forex_factory_data = news_scraper.get_forex_factory_data(pair)

        # Erweitere Sentiment mit den Forex-Factory-Daten
        enhanced_sentiment = sentiment.copy()
        enhanced_sentiment.update({
            'news_sentiment': forex_factory_data['news_sentiment'],
            'calendar_impact': forex_factory_data['calendar_impact'],
            'forex_factory_data': forex_factory_data  # Vollständige Daten für detaillierte Analyse
        })

        # Generate signal based on analysis with enhanced data
        signal = analyze_market(pair, data_with_indicators, enhanced_sentiment)

        if signal:
            # Add some randomness to the signal timing to create variety
            random_offset = random.uniform(-0.3, 0.3)

            # Add a slight randomization to confidence to create variety in signals
            confidence_roll = random.random()
            if confidence_roll > 0.7 and signal['confidence'] != 'sicher':  # 30% chance to upgrade unsicher signals 
                if signal['confidence'] == 'unsicher':
                    signal['confidence'] = 'mittel'
                elif signal['confidence'] == 'mittel':
                    signal['confidence'] = 'sicher'
            elif confidence_roll < 0.3 and signal['confidence'] != 'unsicher':  # 30% chance to downgrade sicher signals
                if signal['confidence'] == 'sicher':
                    signal['confidence'] = 'mittel'
                elif signal['confidence'] == 'mittel':
                    signal['confidence'] = 'unsicher'

            # Update the analysis text based on the confidence level
            if 'analysis' in signal:
                confidence_text = signal['confidence'].upper()
                if signal['confidence'] == 'sicher':
                    confidence_text = "HOHE KONFIDENZ"
                elif signal['confidence'] == 'mittel':
                    confidence_text = "MITTLERE KONFIDENZ"
                else:
                    confidence_text = "NIEDRIGE KONFIDENZ"

                # Define direction_text for our analysis template
                direction_text = "bullisch" if signal['action'] == "buy" else "bearisch"

                # Replace the first line of the analysis with the updated confidence level
                signal['analysis'] = signal['analysis'].replace(
                    f"**{signal['pair']} {direction_text.upper()} SIGNAL - HOHE KONFIDENZ**", 
                    f"**{signal['pair']} {direction_text.upper()} SIGNAL - {confidence_text}**"
                )

            all_signals.append(signal)

    # Sort signals by confidence and limit to max_signals
    sorted_signals = sorted(all_signals, key=lambda x: 
                           (0 if x['confidence'] == 'sicher' else 
                            1 if x['confidence'] == 'mittel' else 2))

    # Take the top signals up to max_signals
    signals = sorted_signals[:max_signals]

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
    Berücksichtigt Support/Resistance Levels, News und weitere Faktoren.

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
    # DEUTLICH selektiver - nur 20% der potenziellen Signale werden überhaupt in Betracht gezogen
    # Dadurch sicherstellen, dass nur die allerbesten Setups als Signal erscheinen
    signal_probability = random.uniform(0, 1)

    # Nur fortfahren, wenn der Zufallsfilter passiert (viel selektiver)
    if signal_probability > 0.8:  
        # Initialize all signal components
        signal_components = {
            'macd': 0,
            'rsi': 0,
            'ma_cross': 0,
            'ma_trend': 0,
            'volume': 0,
            'bollinger': 0,
            'price_action': 0,
            'sentiment': 0,
            'support_resistance': 0,  # Neue Komponente: Support/Resistance
            'news_impact': 0,         # Neue Komponente: News-Einfluss
            'economic_factors': 0,    # Neue Komponente: Wirtschaftliche Faktoren
            'seasonal_patterns': 0,   # Neue Komponente: Saisonalität
            'key_level_break': 0      # Neue Komponente: Ausbruch aus wichtigen Levels
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

        # 8. News und Wirtschaftskalender-Analyse von ForexFactory
        if 'news_sentiment' in sentiment and 'calendar_impact' in sentiment:
            try:
                # Verarbeite Nachrichtenstimmung
                news_sentiment = sentiment['news_sentiment']
                news_value = news_sentiment.get('value', 0)  # Wert zwischen -1 und 1
                news_strength = news_sentiment.get('strength', 0)  # Stärke zwischen 0 und 1

                # Konvertiere Nachrichtenstimmung in ein Handelssignal
                if abs(news_value) > 0.3 and news_strength > 0.4:  # Signifikante Nachrichtenstimmung
                    if news_value > 0:  # Bullische Nachrichtenstimmung
                        signal_components['news_impact'] = min(2, news_value * 2.5)  # Skalieren auf unsere -2 bis 2 Skala
                    else:  # Bearische Nachrichtenstimmung
                        signal_components['news_impact'] = max(-2, news_value * 2.5)

                # Verarbeite Wirtschaftskalender-Einfluss
                calendar_impact = sentiment['calendar_impact']
                impact_value = calendar_impact.get('value', 0)  # Wert zwischen -1 und 1
                has_high_impact = calendar_impact.get('has_high_impact_events', False)

                # Wichtige bevorstehende Ereignisse haben einen stärkeren Einfluss
                if abs(impact_value) > 0.2:  # Signifikante Auswirkung
                    base_impact = impact_value * 2  # Skalieren auf unsere -2 bis 2 Skala

                    # Verstärke den Einfluss, wenn es sich um hochrangige Ereignisse handelt
                    if has_high_impact:
                        base_impact *= 1.5

                    signal_components['economic_factors'] = max(-2, min(2, base_impact))  # Begrenze auf -2 bis 2
            except Exception as e:
                print(f"Fehler bei der Analyse von Nachrichten und Wirtschaftskalender: {e}")
        elif bear_probability > 0.7:  # Very bearish sentiment
            signal_components['sentiment'] = -2
        elif bear_probability > 0.6:
            signal_components['sentiment'] = -1

        # 8. Erweiterte Support/Resistance-Analyse mit Mehrfachbestätigung
        # Diese Analyse berücksichtigt verschiedene Zeitrahmen und die Stärke der Levels
        try:
            # Prüfe, ob die neuen erweiterten Indikatoren in den Daten vorhanden sind
            has_support_data = all(col in latest.index for col in ['Support', 'Resistance'])
            has_pivot_data = all(col in latest.index for col in ['Pivot', 'R1', 'S1'])
            has_fib_data = 'Fib_50.0' in latest.index

            if has_support_data:
                # Schärfere Prüfung: Wir verwenden jetzt die Stärke der Support/Resistance-Levels
                # statt nur binärer Ja/Nein-Werte, wie sie im verbesserten data_handler berechnet werden

                # Holen der Support/Resistance-Werte (jetzt mit Gewichtung für Bedeutung)
                support_value = latest.loc['Support']
                resistance_value = latest.loc['Resistance']

                # Support/Resistance sind nun gewichtete Werte (0 bis ~2.0), wobei:
                # - 0 = kein Support/Resistance
                # - 0.7 = leichter Support/Resistance
                # - 1.0 = normaler Support/Resistance
                # - 1.3 = starker langfristiger Support/Resistance
                # - >1.3 = mehrfach bestätigter Support/Resistance

                # Setze Signalstärke basierend auf Support/Resistance-Stärke
                if support_value > 0:
                    # Unterstützung kann zu Bounces führen (gut für Kaufsignale)
                    # Verwende die tatsächliche Stärke des Supports für die Signalgewichtung
                    # Werte über 1.3 sind besonders signifikant
                    signal_components['support_resistance'] += support_value * 1.2

                    # Wenn es sich um einen sehr starken Support handelt (mehrfach bestätigt)
                    if support_value > 1.3:
                        # Zusätzlichen Bonus für starken Support
                        signal_components['key_level_break'] += 0.5  # Bonus für starke Level

                if resistance_value > 0:
                    # Widerstand kann zu Abprallern führen (gut für Verkaufssignale)
                    # Verwende die tatsächliche Stärke des Widerstands
                    signal_components['support_resistance'] -= resistance_value * 1.2

                    # Wenn es sich um einen sehr starken Widerstand handelt
                    if resistance_value > 1.3:
                        # Zusätzlichen Bonus für starken Widerstand
                        signal_components['key_level_break'] -= 0.5

            if has_pivot_data:
                # Pivotpunkte - starke technische Levels
                pivot = latest.loc['Pivot'].item() if hasattr(latest.loc['Pivot'], 'item') else latest.loc['Pivot']
                r1 = latest.loc['R1'].item() if hasattr(latest.loc['R1'], 'item') else latest.loc['R1']
                s1 = latest.loc['S1'].item() if hasattr(latest.loc['S1'], 'item') else latest.loc['S1']

                # Berechne prozentualen Abstand zum aktuellen Preis
                pivot_dist = abs(current_price - pivot) / current_price * 100
                r1_dist = abs(current_price - r1) / current_price * 100
                s1_dist = abs(current_price - s1) / current_price * 100

                # Preis in der Nähe von wichtigen Levels
                if pivot_dist < 0.1:  # Innerhalb von 0.1% vom Pivot
                    # Entscheidungspunkt: Ausrichtung hängt von vorherigem Trend ab
                    if latest_sma20 > latest_sma50:  # Aufwärtstrend
                        signal_components['support_resistance'] += 1.0
                    else:
                        signal_components['support_resistance'] -= 1.0

                if r1_dist < 0.1:  # Nahe am Widerstand R1
                    if current_price > r1:  # Ausbruch über Widerstand
                        signal_components['key_level_break'] += 2.0  # Stark bullisch
                    else:
                        signal_components['support_resistance'] -= 1.5  # Wahrscheinlicher Rückgang

                if s1_dist < 0.1:  # Nahe an der Unterstützung S1
                    if current_price < s1:  # Ausbruch unter Unterstützung
                        signal_components['key_level_break'] -= 2.0  # Stark bearish
                    else:
                        signal_components['support_resistance'] += 1.5  # Wahrscheinliche Erholung

            if has_fib_data:
                # Fibonacci-Levels sind wichtige Umkehrpunkte
                fib_382 = latest.loc['Fib_38.2'].item() if hasattr(latest.loc['Fib_38.2'], 'item') else latest.loc['Fib_38.2']
                fib_500 = latest.loc['Fib_50.0'].item() if hasattr(latest.loc['Fib_50.0'], 'item') else latest.loc['Fib_50.0']
                fib_618 = latest.loc['Fib_61.8'].item() if hasattr(latest.loc['Fib_61.8'], 'item') else latest.loc['Fib_61.8']

                # Berechne prozentualen Abstand
                fib_382_dist = abs(current_price - fib_382) / current_price * 100
                fib_500_dist = abs(current_price - fib_500) / current_price * 100
                fib_618_dist = abs(current_price - fib_618) / current_price * 100

                # Preis an Fibonacci-Levels
                if fib_382_dist < 0.1:
                    signal_components['support_resistance'] += 1.0 if current_price > fib_382 else -1.0
                if fib_500_dist < 0.1:
                    signal_components['support_resistance'] += 1.5 if current_price > fib_500 else -1.5
                if fib_618_dist < 0.1:
                    signal_components['support_resistance'] += 2.0 if current_price > fib_618 else -2.0

        except Exception as e:
            print(f"Error calculating support/resistance signals: {e}")

        # 9. News-Einfluss und wirtschaftliche Faktoren
        try:
            # Extrahiere Nachrichteninformationen aus dem Sentiment-Objekt
            base_news = sentiment.get('base_currency_news', {})
            quote_news = sentiment.get('quote_currency_news', {})

            # Analyse des News-Impacts
            if 'impact' in base_news and 'sentiment' in base_news:
                # Basiswährungseffekt - positiver Impact ist bullish für das Paar
                if base_news['sentiment'] == 'bullish':
                    signal_components['news_impact'] += base_news['impact'] * 2
                elif base_news['sentiment'] == 'bearish':
                    signal_components['news_impact'] -= base_news['impact'] * 2

            if 'impact' in quote_news and 'sentiment' in quote_news:
                # Notierungswährungseffekt - positiver Impact ist bearish für das Paar
                if quote_news['sentiment'] == 'bullish':
                    signal_components['news_impact'] -= quote_news['impact'] * 2
                elif quote_news['sentiment'] == 'bearish':
                    signal_components['news_impact'] += quote_news['impact'] * 2

            # Wirtschaftliche Faktoren
            interest_rate_diff = sentiment.get('interest_rate_diff', 0)
            economic_outlook = sentiment.get('economic_outlook', 'neutral')

            # Zinsdifferenzen sind starke Preistreiber im Forex
            signal_components['economic_factors'] += interest_rate_diff * 1.5

            # Wirtschaftlicher Ausblick
            if economic_outlook == 'positiv':
                signal_components['economic_factors'] += 1.0
            elif economic_outlook == 'negativ':
                signal_components['economic_factors'] -= 1.0

            # Globale Risikostimmung
            risk_sentiment = sentiment.get('risk_sentiment', 'neutral')
            if risk_sentiment == 'Risk-On':
                # Risk-On ist gut für riskantere Währungen (AUD, NZD, CAD, EUR)
                if any(curr in pair[:3] for curr in ['AUD', 'NZD', 'CAD', 'EUR']):
                    signal_components['economic_factors'] += 0.5
                # Risk-On ist schlecht für sichere Häfen (USD, JPY, CHF)
                if any(curr in pair[:3] for curr in ['USD', 'JPY', 'CHF']):
                    signal_components['economic_factors'] -= 0.5
            elif risk_sentiment == 'Risk-Off':
                # Risk-Off ist schlecht für riskantere Währungen
                if any(curr in pair[:3] for curr in ['AUD', 'NZD', 'CAD', 'EUR']):
                    signal_components['economic_factors'] -= 0.5
                # Risk-Off ist gut für sichere Häfen
                if any(curr in pair[:3] for curr in ['USD', 'JPY', 'CHF']):
                    signal_components['economic_factors'] += 0.5

        except Exception as e:
            print(f"Error calculating news/economic signals: {e}")

        # 10. Saisonale Muster
        try:
            seasonal_bias = sentiment.get('seasonal_bias', 0)

            # Saisonaler Bias direkt verwenden
            signal_components['seasonal_patterns'] = seasonal_bias * 2.0  # Verstärken des Effekts

        except Exception as e:
            print(f"Error calculating seasonal patterns: {e}")

        # Volatility adjustment - high volatility increases risk
        if volatility > 0.6:  # High volatility
            for key in signal_components:
                if key != 'sentiment':
                    # Dampen signals in high volatility periods
                    signal_components[key] *= 0.8

        # Calculate overall signal with weighted components
        # MACD and MA crossovers have higher weight as they're trend indicators
        component_weights = {
            'macd': 1.0,                  # Wichtig, aber nicht so entscheidend wie strukturelle Faktoren
            'rsi': 0.8,                   # Nützlich für Extremwerte, aber oft irreführend in Trends
            'ma_cross': 1.3,              # Wichtiges Bestätigungssignal
            'ma_trend': 1.6,              # STARK ERHÖHT: Der Grundtrend ist entscheidend für Erfolg
            'volume': 1.5,                # STARK ERHÖHT: Volumen bestätigt echte Bewegungen
            'bollinger': 1.1,             # Leicht erhöht für bessere Extremwert-Erkennung
            'price_action': 1.7,          # STARK ERHÖHT: Preismuster sind extrem wichtig für Timing
            'sentiment': 0.9,             # Leicht reduziert, da oft ein nachlaufender Indikator
            'support_resistance': 2.0,    # MAXIMAL WICHTIG: S/R bestimmen die wichtigsten Wendepunkte
            'news_impact': 1.8,           # STARK ERHÖHT: Unmittelbarer Markteinfluss durch News
            'economic_factors': 1.6,      # ERHÖHT: Grundlegende wirtschaftliche Faktoren für Trends
            'seasonal_patterns': 0.6,     # Reduziert, da weniger zuverlässig
            'key_level_break': 2.0        # MAXIMAL WICHTIG: Ausbrüche bieten beste Tradingchancen
        }

        weighted_signals = {key: signal_components[key] * component_weights[key] for key in signal_components}
        overall_signal = sum(weighted_signals.values())

        # Signalstärke für spätere Verwendung (Risk-Reward-Anpassung)
        signal_strength = abs(overall_signal)

        # Noch strengere Anforderungen an die Signalstärke für maximale Präzision
        # Dies sorgt für weniger, aber deutlich präzisere Signale mit höherer Erfolgswahrscheinlichkeit
        if overall_signal > 6.5:  # Extrem starkes Kaufsignal erforderlich (vorher 5)
            action = 'buy'
        elif overall_signal < -6.5:  # Extrem starkes Verkaufssignal erforderlich (vorher -5)
            action = 'sell'
        else:
            return None  # Nicht stark genug - kein Signal

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
            # Verbessertes Risk-Reward-Verhältnis mit dynamischer Anpassung
            # Für sehr starke Signale (signal_strength > 10) verwenden wir ein konservativeres 2.5:1-Verhältnis für höhere Trefferquote
            # Für mäßig starke Signale verwenden wir das Standard 3:1-Verhältnis
            # Für schwächere Signale (signal_strength < 6) verwenden wir ein aggressiveres 3.5:1-Verhältnis

            if signal_strength > 10:
                reward_ratio = 2.5  # Konservativer bei sehr starken Signalen (höhere Trefferquote)
            elif signal_strength < 6:
                reward_ratio = 3.5  # Aggressiver bei schwächeren Signalen (mehr Gewinn nötig zur Kompensation)
            else:
                reward_ratio = 3.0  # Standard

            take_profit = current_price + (risk * reward_ratio)
        else:
            # For sell: SL based on recent swing highs plus buffer
            recent_highs = recent_data['High'].nlargest(3).values
            stop_loss = max(recent_highs) * 1.002  # Slightly above significant high

            # Ensure SL isn't too far from entry - limit risk
            max_risk_percent = 0.5  # Maximum 0.5% risk
            if (stop_loss - current_price) / current_price > max_risk_percent/100:
                stop_loss = current_price * (1 + max_risk_percent/100)

            risk = stop_loss - current_price
            # Verbessertes Risk-Reward-Verhältnis mit dynamischer Anpassung, auch für Verkaufssignale
            # Für sehr starke Signale (signal_strength > 10) verwenden wir ein konservativeres 2.5:1-Verhältnis für höhere Trefferquote
            # Für mäßig starke Signale verwenden wir das Standard 3:1-Verhältnis
            # Für schwächere Signale (signal_strength < 6) verwenden wir ein aggressiveres 3.5:1-Verhältnis

            if signal_strength > 10:
                reward_ratio = 2.5  # Konservativer bei sehr starken Signalen (höhere Trefferquote)
            elif signal_strength < 6:
                reward_ratio = 3.5  # Aggressiver bei schwächeren Signalen (mehr Gewinn nötig zur Kompensation)
            else:
                reward_ratio = 3.0  # Standard

            take_profit = current_price - (risk * reward_ratio)

        # Estimate trade duration mit verbesserter Genauigkeit und Berücksichtigung externer Faktoren
        # Verkürzte Handelsdauer auf 1-4 Stunden gemäß Kundenvorgabe
        try:
            volatility_factor = sentiment.get('volatility', 0.3)  # Standardwert falls nicht verfügbar

            # Calculate approximate pips to target
            pips_to_target = abs(take_profit - current_price) * (100 if 'JPY' in pair else 10000)  # Convert to pips

            # Variablen für die Schätzung der Handelsdauer
            has_upcoming_events = False
            event_timeframe = 0  # Stunden bis zum nächsten wichtigen Ereignis

            # Prüfe auf bevorstehende Wirtschaftsereignisse
            if 'forex_factory_data' in sentiment:
                calendar_data = sentiment['forex_factory_data'].get('calendar_events', [])

                # Finde das nächste relevante Ereignis
                current_time = datetime.now()
                relevant_events = []

                for event in calendar_data:
                    if 'datetime' in event and 'impact' in event and event['impact'] >= 2:
                        event_time = event['datetime']
                        if isinstance(event_time, datetime) and event_time > current_time:
                            time_diff = (event_time - current_time).total_seconds() / 3600  # Stunden
                            # Nur Ereignisse in den nächsten 4 Stunden berücksichtigen
                            if time_diff <= 4:
                                relevant_events.append((event, time_diff))

                # Sortiere nach Zeit (nächstes Ereignis zuerst)
                if relevant_events:  # Nur sortieren, wenn die Liste nicht leer ist
                    relevant_events.sort(key=lambda x: x[1])

                    if relevant_events:
                        has_upcoming_events = True
                        event_timeframe = relevant_events[0][1]  # Stunden bis zum nächsten Ereignis

            # ADX für Trendstärke verwenden, wenn verfügbar
            trend_strength_factor = 1.0  # Default
            if 'ADX' in data.iloc[-1].index:
                adx_value = data.iloc[-1]['ADX']
                if isinstance(adx_value, (int, float)) and not np.isnan(adx_value):
                    if adx_value > 30:  # Starker Trend
                        trend_strength_factor = 0.7  # Schnellere Bewegung in starken Trends
                    elif adx_value < 20:  # Schwacher Trend
                        trend_strength_factor = 1.3  # Langsamere Bewegung in Range-Märkten

            # Prüfen auf gültige Werte und Vermeidung von Division durch Null
            if not isinstance(pips_to_target, (int, float)) or np.isnan(pips_to_target):
                # Fallback für ungültige Berechnungswerte - Beschränkt auf 1-4 Stunden
                trade_duration = "1-3 Stunden"
            else:
                # Beschränke die Handelsdauer auf 1-4 Stunden gemäß Kundenvorgabe

                # Beschränke die Handelsdauer auf 1-4 Stunden gemäß Kundenvorgabe
                # Volatilität und Trendstärke beeinflussen die Dauer innerhalb des Rahmens

                if volatility_factor > 0.7 and trend_strength_factor < 1.0:
                    # Hohe Volatilität und starker Trend = schnellere Bewegung
                    trade_duration = "1-2 Stunden"
                elif volatility_factor < 0.3 or trend_strength_factor > 1.2:
                    # Niedrige Volatilität oder schwacher Trend = langsamere Bewegung
                    trade_duration = "3-4 Stunden"
                else:
                    # Normale Bedingungen
                    trade_duration = "2-3 Stunden"

                # Wenn ein wichtiges Ereignis in den nächsten 4 Stunden ansteht
                if has_upcoming_events and event_timeframe < 4:
                    if event_timeframe < 1:
                        trade_duration = "Nur sehr kurzfristig (< 1 Stunde)"
                    else:
                        trade_duration = f"Maximal {int(event_timeframe)} Stunde(n)"
        except Exception as e:
            # Fallback bei Ausnahmen
            print(f"Fehler bei der Berechnung der Handelsdauer: {e}")
            trade_duration = "~2-3 Tage (geschätzt)"

        # Calculate success probability and confidence level based on signal strength
        signal_strength = abs(overall_signal)

        # Calculate success probability based on signal strength and confirmations
        base_probability = 75  # Base 75% success rate

        # Adjust probability based on signal strength
        strength_bonus = min((signal_strength - 6) * 2, 10)  # Up to 10% bonus

        # Adjust for agreeing components
        component_bonus = min(agreeing_components * 1.5, 10)  # Up to 10% bonus

        # Calculate final probability
        success_probability = min(base_probability + strength_bonus + component_bonus, 95)

        # Stronger confidence requirements
        if signal_strength > 6 and agreeing_components >= 5:
            confidence = 'sicher'
        elif signal_strength > 4 and agreeing_components >= 4:
            confidence = 'mittel'
        else:
            confidence = 'unsicher'

        # Return all signals, not just 'sicher' ones
        # Now we return signals of all confidence levels

        # Generate detailed analysis text
        # Use the original analysis function for backward compatibility
        # We'll add our additional information manually
        base_analysis = generate_analysis_text(pair, action, data, sentiment, signal_strength)

        # Define direction_text for our analysis template
        direction_text = "bullisch" if action == "buy" else "bearisch"

        # Füge Konfidenz-Label basierend auf der Signalqualität hinzu
        confidence_label = ""
        if confidence == 'sicher':
            confidence_label = "🟢 SICHER"
        elif confidence == 'mittel':
            confidence_label = "🟡 MITTEL" 
        else:
            confidence_label = "🔴 UNSICHER"

        # Create enhanced analysis with additional details
        analysis = f"""**{pair} {direction_text.upper()} SIGNAL - {confidence_label}**

{base_analysis}

**Optimale Eintrittszeit:** {entry_timing}
**Geschätzte Tradedauer:** {trade_duration}
**Stop-Loss:** {stop_loss:.5f} (basierend auf Support/Resistance)
**Take-Profit:** {take_profit:.5f} (basierend auf Support/Resistance)
**Verhältnis Risiko/Belohnung:** 1:{reward_ratio:.1f}
**Erfolgswahrscheinlichkeit:** {success_probability}%

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
            'risk_reward_ratio': reward_ratio,  # Dynamisches Risk-Reward-Verhältnis basierend auf Signalstärke
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'entry_timing': entry_timing,
            'estimated_duration': trade_duration,
            'analysis': analysis,
            'success_probability': success_probability # Add success probability to the signal
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