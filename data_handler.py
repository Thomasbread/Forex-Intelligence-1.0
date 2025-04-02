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
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
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
    # For this demo, we'll generate random sentiment
    
    bullish_probability = random.uniform(0, 1)
    bearish_probability = 1 - bullish_probability
    
    sentiment = {
        'bullish_probability': bullish_probability,
        'bearish_probability': bearish_probability,
        'trend_strength': random.uniform(0.3, 0.9),
        'volatility': random.uniform(0.1, 0.8)
    }
    
    return sentiment
