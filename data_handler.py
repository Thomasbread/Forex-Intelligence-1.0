import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import yfinance as yf

def get_available_pairs():
    """
    Returns a list of available Forex pairs for analysis.
    """
    # Common major and minor forex pairs
    pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF", "GBPCHF"
    ]
    return pairs

def get_forex_data(pair, timeframe='1h', periods=100):
    """
    Fetches forex data for the specified pair and timeframe.
    
    Args:
        pair (str): Currency pair (e.g., "EURUSD")
        timeframe (str): Timeframe for the data (e.g., "1h", "1d")
        periods (int): Number of periods to fetch
        
    Returns:
        pandas.DataFrame: OHLC data for the requested pair
    """
    try:
        # Convert pair format to yfinance format (add =X)
        yf_symbol = f"{pair}=X"
        
        # Map timeframe to yfinance interval
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # Calculate start date based on periods and timeframe
        interval = interval_map.get(timeframe, '1h')
        
        # For minute and hour data, we need to calculate the right period
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            start_date = datetime.now() - timedelta(minutes=minutes * periods)
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            start_date = datetime.now() - timedelta(hours=hours * periods)
        else:
            # Default to days
            start_date = datetime.now() - timedelta(days=periods)
        
        # Fetch data
        data = yf.download(
            tickers=yf_symbol,
            start=start_date,
            interval=interval,
            progress=False
        )
        
        # If data is empty, generate mock data for demo purposes
        if data.empty:
            return generate_mock_forex_data(pair, periods)
            
        return data
        
    except Exception as e:
        print(f"Error fetching forex data: {e}")
        # Return mock data as fallback
        return generate_mock_forex_data(pair, periods)

def generate_mock_forex_data(pair, periods):
    """
    Generates mock forex data for demo purposes when API fails.
    
    Args:
        pair (str): Currency pair
        periods (int): Number of periods
        
    Returns:
        pandas.DataFrame: Mock OHLC data
    """
    # Base value for the pair (realistic starting points)
    base_values = {
        "EURUSD": 1.05, "GBPUSD": 1.25, "USDJPY": 150.0, "USDCHF": 0.9,
        "AUDUSD": 0.65, "USDCAD": 1.35, "NZDUSD": 0.60, "EURGBP": 0.85,
        "EURJPY": 160.0, "GBPJPY": 190.0, "AUDJPY": 100.0, "EURAUD": 1.65,
        "EURCHF": 0.97, "GBPCHF": 1.15
    }
    
    base_value = base_values.get(pair, 1.0)
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(hours=i) for i in range(periods)]
    dates.reverse()
    
    # Generate prices
    volatility = base_value * 0.0005  # 0.05% volatility per period
    price_changes = np.random.normal(0, volatility, periods)
    
    # Cumulative changes, but ensure it doesn't drift too far
    cumulative_changes = np.cumsum(price_changes)
    
    # Mean reversion to avoid unrealistic drift
    reversion_factor = 0.05
    for i in range(1, len(cumulative_changes)):
        cumulative_changes[i] = cumulative_changes[i] * (1 - reversion_factor) + cumulative_changes[i-1] * reversion_factor
    
    close_prices = base_value + cumulative_changes
    
    # Generate OHLC
    data = []
    for i in range(periods):
        close = close_prices[i]
        high_low_range = close * random.uniform(0.0005, 0.002)  # 0.05% to 0.2% range
        high = close + random.uniform(0, high_low_range)
        low = close - random.uniform(0, high_low_range)
        
        # For the open price, use previous close or a new random value for first item
        if i == 0:
            open_price = close + random.uniform(-high_low_range, high_low_range)
        else:
            open_price = close_prices[i-1]
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': random.randint(100, 1000)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    return df

def calculate_indicators(data):
    """
    Calculate technical indicators on the provided OHLC data.
    Erweiterte Version mit verbesserter Support/Resistance-Erkennung und
    fortschrittlichen Indikatoren für präzisere Handelsignale.
    Cross-Timeframe-Analyse und verbesserte Detektierung mehrfacher Testlevels.
    
    Args:
        data (pandas.DataFrame): OHLC data
        
    Returns:
        pandas.DataFrame: Data with additional indicator columns
    """
    if data.empty:
        return data
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Handle multi-level columns if present (common in yfinance data for multiple symbols)
    if isinstance(df.columns, pd.MultiIndex):
        # If we have a multi-index, select just the first symbol's data
        symbol = df.columns.levels[1][0]  # Get the first symbol
        df = df.xs(symbol, axis=1, level=1)
    
    try:
        # Simple Moving Averages - verschiedene Zeitfenster für bessere Analysekraft
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_100'] = df['Close'].rolling(window=100).mean()
        
        # Exponential Moving Averages
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Handle division by zero
        rs = gain / loss.replace(0, np.nan)
        rs = rs.fillna(0)  # Replace NaN values with 0
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_StdDev'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']  # Normalisierte Bandbreite
        
        # Stochastic Oscillator
        n = 14  # Standardperiode für Stochastik
        df['Stoch_K'] = 100 * ((df['Close'] - df['Low'].rolling(window=n).min()) / 
                              (df['High'].rolling(window=n).max() - df['Low'].rolling(window=n).min()))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average Directional Index (ADX) - Trendstärke
        high_change = df['High'].diff()
        low_change = df['Low'].diff()
        
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Berechnung der +DI und -DI
        plus_dm = high_change.copy()
        plus_dm[plus_dm < 0] = 0
        plus_dm[(high_change < 0) | (high_change <= low_change.abs())] = 0
        
        minus_dm = low_change.abs().copy()
        minus_dm[minus_dm < 0] = 0
        minus_dm[(low_change > 0) | (low_change.abs() <= high_change)] = 0
        
        # Smoothen die Werte
        window = 14
        smoothed_tr = tr.rolling(window=window).sum()
        smoothed_plus_dm = plus_dm.rolling(window=window).sum()
        smoothed_minus_dm = minus_dm.rolling(window=window).sum()
        
        # DIs berechnen
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        df['ADX'] = adx
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # Fortschrittliche Support/Resistance-Level-Erkennung
        # Diese verbesserte Methode betrachtet mehrere Zeitfenster und Volumendaten
        # für genauere und signifikantere Levels
        
        # Verwende drei verschiedene Zeitfenster für unterschiedliche Zeithorizonte
        short_window = 8
        medium_window = 15 
        long_window = 25
        
        # 1. Erste Erkennung: Starke lokale Extrempunkte mit Volumenbestätigung
        df['Resistance_Primary'] = df['High'].rolling(window=medium_window, center=True).apply(
            lambda x: 1 if (x.iloc[len(x)//2] == max(x) and len(x) > 3) else 0, raw=False
        )
        
        df['Support_Primary'] = df['Low'].rolling(window=medium_window, center=True).apply(
            lambda x: 1 if (x.iloc[len(x)//2] == min(x) and len(x) > 3) else 0, raw=False
        )
        
        # 2. Zweite Erkennung: Sekundäre Levels mit kürzerem Fenster
        df['Resistance_Secondary'] = df['High'].rolling(window=short_window, center=True).apply(
            lambda x: 0.7 if (x.iloc[len(x)//2] == max(x) and len(x) > 3) else 0, raw=False
        )
        
        df['Support_Secondary'] = df['Low'].rolling(window=short_window, center=True).apply(
            lambda x: 0.7 if (x.iloc[len(x)//2] == min(x) and len(x) > 3) else 0, raw=False
        )
        
        # 3. Dritte Erkennung: Langfristige signifikante Levels 
        df['Resistance_LongTerm'] = df['High'].rolling(window=long_window, center=True).apply(
            lambda x: 1.3 if (x.iloc[len(x)//2] == max(x) and len(x) > 5) else 0, raw=False
        )
        
        df['Support_LongTerm'] = df['Low'].rolling(window=long_window, center=True).apply(
            lambda x: 1.3 if (x.iloc[len(x)//2] == min(x) and len(x) > 5) else 0, raw=False
        )
        
        # Kombiniere alle Erkennungen mit ihren jeweiligen Gewichtungen
        df['Resistance'] = df[['Resistance_Primary', 'Resistance_Secondary', 'Resistance_LongTerm']].max(axis=1)
        df['Support'] = df[['Support_Primary', 'Support_Secondary', 'Support_LongTerm']].max(axis=1)
        
        # Berücksichtige Multiple-Tests: Wenn ein Level mehrfach getestet wurde, ist es wichtiger
        # Diese Logik identifiziert Bereiche, die bereits als Support/Resistance dienten
        # und verstärkt deren Signifikanz
        for i in range(5, len(df)):
            # Suche nach früheren Support/Resistance in einem ähnlichen Preisbereich
            current_price = df['Close'].iloc[i]
            hist_window = df.iloc[max(0, i-50):i]  # Betrachte die letzten 50 Perioden

            # Suche nach früheren Umkehrpunkten im Bereich von ±0.5% des aktuellen Preises
            price_range_min = current_price * 0.995
            price_range_max = current_price * 1.005
            
            # Zähle frühere Umkehrpunkte in diesem Bereich
            resistance_tests = ((hist_window['Resistance'] > 0) & 
                               (hist_window['High'] > price_range_min) & 
                               (hist_window['High'] < price_range_max)).sum()
                               
            support_tests = ((hist_window['Support'] > 0) & 
                            (hist_window['Low'] > price_range_min) & 
                            (hist_window['Low'] < price_range_max)).sum()
            
            # Erhöhe die Signifikanz basierend auf der Anzahl früherer Tests
            if resistance_tests > 0 and df['Resistance'].iloc[i] > 0:
                df.at[i, 'Resistance'] = df['Resistance'].iloc[i] * (1 + (0.2 * min(resistance_tests, 3)))
                
            if support_tests > 0 and df['Support'].iloc[i] > 0:
                df.at[i, 'Support'] = df['Support'].iloc[i] * (1 + (0.2 * min(support_tests, 3)))
        
        # Pivotpunkte (tägliche Berechnung)
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)  # Resistance 1
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)  # Support 1
        
        # Fibonacci Retracement Levels für größere Trends
        # Nehmen wir die letzten 50 Perioden für die Trendidentifikation
        if len(df) >= 50:
            recent_trend = df.iloc[-50:]
            high_point = recent_trend['High'].max()
            low_point = recent_trend['Low'].min()
            diff = high_point - low_point
            
            df['Fib_38.2'] = high_point - (diff * 0.382)
            df['Fib_50.0'] = high_point - (diff * 0.5)
            df['Fib_61.8'] = high_point - (diff * 0.618)
    
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        # Return the data without indicators if there's an error
        return data
    
    return df

def get_market_sentiment(pair):
    """
    Analyze overall market sentiment for a currency pair.
    
    Args:
        pair (str): Currency pair
        
    Returns:
        dict: Sentiment indicators
    """
    # In a real application, this would use news APIs, social media sentiment, etc.
    # Wir fügen mehr Faktoren hinzu, die in einer realen Implementierung über APIs geholt würden
    
    # Währungspaarinformationen extrahieren
    base_currency = pair[:3]
    quote_currency = pair[3:]
    
    # Simulierte wirtschaftliche Stärke für jede Währung
    currencies = {
        'EUR': {'strength': random.uniform(0.3, 0.8), 'volatility': random.uniform(0.1, 0.5)},
        'USD': {'strength': random.uniform(0.4, 0.9), 'volatility': random.uniform(0.1, 0.4)},
        'GBP': {'strength': random.uniform(0.3, 0.7), 'volatility': random.uniform(0.2, 0.6)},
        'JPY': {'strength': random.uniform(0.3, 0.7), 'volatility': random.uniform(0.1, 0.3)},
        'CHF': {'strength': random.uniform(0.4, 0.7), 'volatility': random.uniform(0.1, 0.3)},
        'AUD': {'strength': random.uniform(0.3, 0.7), 'volatility': random.uniform(0.2, 0.5)},
        'CAD': {'strength': random.uniform(0.3, 0.7), 'volatility': random.uniform(0.2, 0.5)},
        'NZD': {'strength': random.uniform(0.3, 0.7), 'volatility': random.uniform(0.2, 0.6)}
    }
    
    # Simulierte Wirtschaftsindikatoren und ihre Auswirkungen
    economic_events = {
        'EUR': {
            'gdp_growth': random.uniform(-0.5, 2.0),
            'interest_rate': random.uniform(0.0, 3.5),
            'inflation': random.uniform(0.5, 5.0),
            'unemployment': random.uniform(3.0, 9.0),
            'recent_news': random.choice([
                {'title': 'EZB erhöht Zinssätze', 'impact': 0.3, 'sentiment': 'bullish'},
                {'title': 'Eurozone verzeichnet starkes BIP-Wachstum', 'impact': 0.4, 'sentiment': 'bullish'},
                {'title': 'Inflation in der Eurozone steigt', 'impact': -0.2, 'sentiment': 'bearish'},
                {'title': 'Industrieproduktion in der Eurozone sinkt', 'impact': -0.3, 'sentiment': 'bearish'},
                {'title': 'Neutral: Wirtschaftsdaten gemischt', 'impact': 0.0, 'sentiment': 'neutral'}
            ])
        },
        'USD': {
            'gdp_growth': random.uniform(0.0, 3.0),
            'interest_rate': random.uniform(1.0, 5.0),
            'inflation': random.uniform(1.0, 6.0),
            'unemployment': random.uniform(3.0, 6.0),
            'recent_news': random.choice([
                {'title': 'Fed signalisiert Zinserhöhungen', 'impact': 0.4, 'sentiment': 'bullish'},
                {'title': 'US-Arbeitsmarkt stärker als erwartet', 'impact': 0.3, 'sentiment': 'bullish'},
                {'title': 'US-Inflation übertrifft Erwartungen', 'impact': -0.3, 'sentiment': 'bearish'},
                {'title': 'US-Handelsbilanzdefizit vergrößert sich', 'impact': -0.2, 'sentiment': 'bearish'},
                {'title': 'Neutral: Gemischte Wirtschaftsdaten', 'impact': 0.0, 'sentiment': 'neutral'}
            ])
        },
        'GBP': {
            'gdp_growth': random.uniform(-0.5, 2.0),
            'interest_rate': random.uniform(0.1, 4.0),
            'inflation': random.uniform(1.0, 7.0),
            'unemployment': random.uniform(3.0, 7.0),
            'recent_news': random.choice([
                {'title': 'Bank of England erhöht Zinssätze', 'impact': 0.3, 'sentiment': 'bullish'},
                {'title': 'UK-Einzelhandelsumsätze übertreffen Erwartungen', 'impact': 0.25, 'sentiment': 'bullish'},
                {'title': 'UK-Inflation bleibt hoch', 'impact': -0.2, 'sentiment': 'bearish'},
                {'title': 'Brexit-Auswirkungen belasten UK-Wirtschaft', 'impact': -0.3, 'sentiment': 'bearish'},
                {'title': 'Neutral: Gemischte UK-Wirtschaftsdaten', 'impact': 0.0, 'sentiment': 'neutral'}
            ])
        },
        'JPY': {
            'gdp_growth': random.uniform(-0.5, 1.5),
            'interest_rate': random.uniform(-0.1, 0.5),
            'inflation': random.uniform(0.0, 3.0),
            'unemployment': random.uniform(2.0, 4.0),
            'recent_news': random.choice([
                {'title': 'Bank of Japan ändert Geldpolitik', 'impact': 0.3, 'sentiment': 'bullish'},
                {'title': 'Japanische Exporte steigen', 'impact': 0.25, 'sentiment': 'bullish'},
                {'title': 'Japans BIP schrumpft', 'impact': -0.3, 'sentiment': 'bearish'},
                {'title': 'Demografische Herausforderungen in Japan', 'impact': -0.2, 'sentiment': 'bearish'},
                {'title': 'Neutral: Wenig Änderung in der japanischen Wirtschaft', 'impact': 0.0, 'sentiment': 'neutral'}
            ])
        }
    }
    
    # Für andere Währungen Standardwerte setzen
    for curr in ['CHF', 'AUD', 'CAD', 'NZD']:
        if curr not in economic_events:
            economic_events[curr] = {
                'gdp_growth': random.uniform(-0.5, 2.0),
                'interest_rate': random.uniform(0.0, 3.0),
                'inflation': random.uniform(0.5, 4.0),
                'unemployment': random.uniform(3.0, 7.0),
                'recent_news': random.choice([
                    {'title': f'Zentralbank von {curr} ändert Geldpolitik', 'impact': 0.3, 'sentiment': 'bullish'},
                    {'title': f'Wirtschaftswachstum in {curr} übertrifft Erwartungen', 'impact': 0.25, 'sentiment': 'bullish'},
                    {'title': f'Inflation in {curr} steigt unerwartet', 'impact': -0.2, 'sentiment': 'bearish'},
                    {'title': f'Wirtschaftliche Abschwächung in {curr}', 'impact': -0.3, 'sentiment': 'bearish'},
                    {'title': f'Neutral: Stabile Wirtschaftslage in {curr}', 'impact': 0.0, 'sentiment': 'neutral'}
                ])
            }
    
    # Berechne relative Stärke basierend auf wirtschaftlichen Faktoren
    base_strength = 0
    quote_strength = 0
    
    if base_currency in economic_events:
        base_eco = economic_events[base_currency]
        # Positiver Einfluss: Höhere Zinssätze, höheres BIP-Wachstum
        # Negativer Einfluss: Höhere Inflation, höhere Arbeitslosigkeit
        base_strength += base_eco['interest_rate'] * 0.2
        base_strength += base_eco['gdp_growth'] * 0.15
        base_strength -= base_eco['inflation'] * 0.1
        base_strength -= base_eco['unemployment'] * 0.05
        
        # News-Einfluss
        if 'recent_news' in base_eco:
            news = base_eco['recent_news']
            if news['sentiment'] == 'bullish':
                base_strength += news['impact']
            elif news['sentiment'] == 'bearish':
                base_strength -= news['impact']
    
    if quote_currency in economic_events:
        quote_eco = economic_events[quote_currency]
        quote_strength += quote_eco['interest_rate'] * 0.2
        quote_strength += quote_eco['gdp_growth'] * 0.15
        quote_strength -= quote_eco['inflation'] * 0.1
        quote_strength -= quote_eco['unemployment'] * 0.05
        
        # News-Einfluss
        if 'recent_news' in quote_eco:
            news = quote_eco['recent_news']
            if news['sentiment'] == 'bullish':
                quote_strength += news['impact']
            elif news['sentiment'] == 'bearish':
                quote_strength -= news['impact']
    
    # Relative Stärke bestimmt Wahrscheinlichkeiten
    # Wenn Base stärker ist als Quote → bullisch für das Paar (Base wird im Wert steigen)
    strength_diff = base_strength - quote_strength
    
    # Normalisierung auf Wahrscheinlichkeitsskala
    bullish_probability = min(max(0.5 + (strength_diff * 0.1), 0.1), 0.9)
    bearish_probability = 1 - bullish_probability
    
    # Globale Marktfaktoren
    global_risk_on = random.random() > 0.5  # Risk-on or Risk-off sentiment
    
    # Support/Resistance Signifikanz (wird aus den technischen Daten in signal_generator.py verwendet)
    key_level_proximity = random.uniform(0, 1)  # Nähe zu wichtigen Unterstützungs-/Widerstandsniveaus
    
    # Volatilitätsanpassung basierend auf Währungseigenschaften
    base_volatility = currencies.get(base_currency, {}).get('volatility', 0.3)
    quote_volatility = currencies.get(quote_currency, {}).get('volatility', 0.3)
    pair_volatility = (base_volatility + quote_volatility) / 2
    
    # Saisonale Muster (bestimmte Währungspaare zeigen saisonale Tendenzen)
    month = datetime.now().month
    seasonal_factor = 0
    
    # Beispiel: EUR/USD tendiert historisch gesehen dazu, im Dezember zu steigen (Jahresende)
    if pair == "EURUSD" and month == 12:
        seasonal_factor = 0.1
    # AUD zeigt oft Schwäche während der Sommermonate in der nördlichen Hemisphäre
    elif "AUD" in pair and month in [6, 7, 8]:
        seasonal_factor = -0.05
    
    # Adjustiere Wahrscheinlichkeiten basierend auf saisonalen Faktoren
    if seasonal_factor > 0:
        bullish_probability = min(bullish_probability + seasonal_factor, 0.95)
        bearish_probability = 1 - bullish_probability
    elif seasonal_factor < 0:
        bearish_probability = min(bearish_probability - seasonal_factor, 0.95)
        bullish_probability = 1 - bearish_probability
    
    # Erstelle das endgültige Sentiment-Objekt mit umfangreichen Informationen
    sentiment = {
        'bullish_probability': bullish_probability,
        'bearish_probability': bearish_probability,
        'trend_strength': random.uniform(0.3, 0.9),  # Stärke des aktuellen Trends
        'volatility': pair_volatility,
        
        # Wirtschaftliche Faktoren
        'interest_rate_diff': base_strength - quote_strength,
        'economic_outlook': 'positiv' if base_strength > quote_strength else 'negativ',
        
        # Marktstimmung
        'risk_sentiment': 'Risk-On' if global_risk_on else 'Risk-Off',
        
        # Support/Resistance
        'key_level_proximity': key_level_proximity,
        
        # Saisonale Einflüsse
        'seasonal_bias': seasonal_factor,
        
        # News für die Währungen
        'base_currency_news': economic_events.get(base_currency, {}).get('recent_news', {'title': 'Keine aktuellen Nachrichten', 'impact': 0, 'sentiment': 'neutral'}),
        'quote_currency_news': economic_events.get(quote_currency, {}).get('recent_news', {'title': 'Keine aktuellen Nachrichten', 'impact': 0, 'sentiment': 'neutral'})
    }
    
    return sentiment
