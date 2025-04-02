def format_percentage(value):
    """
    Format a value as a percentage with one decimal place.
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted percentage
    """
    return f"{value:.1f}%"

def get_confidence_color(confidence):
    """
    Returns a color hex code based on confidence level.
    
    Args:
        confidence (str): Confidence level (unsicher, mittel, sicher)
        
    Returns:
        str: Hex color code
    """
    if confidence == 'sicher':
        return "#28a745"  # Green
    elif confidence == 'mittel':
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def format_pips(value):
    """
    Format a pip value with +/- sign and one decimal place.
    
    Args:
        value (float): Pip value
        
    Returns:
        str: Formatted pip value
    """
    return f"{value:+.1f}"
