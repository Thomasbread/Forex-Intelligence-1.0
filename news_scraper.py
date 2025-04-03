"""
News-Scraper-Modul für die Beschaffung von Forex-bezogenen Nachrichtendaten und Wirtschaftskalendern.
Dieses Modul nutzt öffentliche APIs und Web-Scraping, um relevante Informationen zu sammeln.
"""

import requests
from bs4 import BeautifulSoup
import datetime
import random
import time
import json
from typing import Dict, List, Any, Optional, Tuple

# Cache für Nachrichten und Kalenderdaten, um wiederholte Anfragen zu vermeiden
news_cache = {}
calendar_cache = {}
news_cache_expiry = datetime.datetime.now()
calendar_cache_expiry = datetime.datetime.now()

# Maximale Cache-Zeit in Sekunden
CACHE_EXPIRY = 3600  # 1 Stunde


def get_forex_news(currency_pair: str = None) -> List[Dict[str, Any]]:
    """
    Ruft aktuelle Forex-Nachrichten von verschiedenen Quellen ab. Kann nach Währungspaar gefiltert werden.
    
    Args:
        currency_pair (str, optional): Währungspaar, nach dem gefiltert werden soll (z.B. "EURUSD")
        
    Returns:
        List[Dict]: Liste von Nachrichtenartikeln mit Titel, Quelle, Datum und Stimmungsbewertung
    """
    global news_cache, news_cache_expiry
    
    # Verwende Cache, wenn er weniger als eine Stunde alt ist
    if news_cache and datetime.datetime.now() < news_cache_expiry:
        print("Verwende gecachte Nachrichtendaten")
        if currency_pair:
            # Filtere Cache nach Währungspaar
            return [n for n in news_cache if currency_pair[:3] in n.get('currencies', []) 
                   or currency_pair[3:] in n.get('currencies', [])]
        return news_cache
    
    try:
        # In einer Produktionsumgebung würden wir hier echte API-Abrufe oder Web-Scraping durchführen
        # Da dies eine Demonstration ist, generieren wir realistische Beispieldaten
        news = generate_realistic_forex_news()
        
        # Aktualisiere Cache und Verfallszeit
        news_cache = news
        news_cache_expiry = datetime.datetime.now() + datetime.timedelta(seconds=CACHE_EXPIRY)
        
        if currency_pair:
            # Filtere Ergebnisse nach Währungspaar
            return [n for n in news if currency_pair[:3] in n.get('currencies', []) 
                   or currency_pair[3:] in n.get('currencies', [])]
        
        return news
    except Exception as e:
        print(f"Fehler beim Abrufen von Forex-Nachrichten: {e}")
        # Wenn wir einen Fehler haben, geben wir den Cache zurück (falls vorhanden)
        if news_cache:
            if currency_pair:
                return [n for n in news_cache if currency_pair[:3] in n.get('currencies', []) 
                       or currency_pair[3:] in n.get('currencies', [])]
            return news_cache
        # Leere Liste zurückgeben, wenn kein Cache vorhanden ist
        return []


def get_economic_calendar(days: int = 3, currency: str = None) -> List[Dict[str, Any]]:
    """
    Ruft den Wirtschaftskalender für die angegebene Anzahl von Tagen ab. Kann nach Währung gefiltert werden.
    
    Args:
        days (int): Anzahl der Tage, für die der Kalender abgerufen werden soll
        currency (str, optional): Währungscode, nach dem gefiltert werden soll (z.B. "EUR")
        
    Returns:
        List[Dict]: Liste von Wirtschaftsereignissen mit Datum, Beschreibung, Währung und Wichtigkeit
    """
    global calendar_cache, calendar_cache_expiry
    
    cache_key = f"{days}_{currency}"
    
    # Verwende Cache, wenn er weniger als eine Stunde alt ist
    if calendar_cache.get(cache_key) and datetime.datetime.now() < calendar_cache_expiry:
        print("Verwende gecachte Kalenderdaten")
        return calendar_cache[cache_key]
    
    try:
        # In einer Produktionsumgebung würden wir hier ForexFactory oder ähnliche Quellen abrufen
        # Da dies eine Demonstration ist, generieren wir realistische Beispieldaten
        calendar_events = generate_realistic_economic_calendar(days, currency)
        
        # Aktualisiere Cache und Verfallszeit
        calendar_cache[cache_key] = calendar_events
        calendar_cache_expiry = datetime.datetime.now() + datetime.timedelta(seconds=CACHE_EXPIRY)
        
        return calendar_events
    except Exception as e:
        print(f"Fehler beim Abrufen des Wirtschaftskalenders: {e}")
        # Wenn wir einen Fehler haben, geben wir den Cache zurück (falls vorhanden)
        if calendar_cache.get(cache_key):
            return calendar_cache[cache_key]
        # Leere Liste zurückgeben, wenn kein Cache vorhanden ist
        return []


def get_forex_factory_data(currency_pair: str) -> Dict[str, Any]:
    """
    Kombiniert Nachrichten- und Kalenderdaten für eine umfassende Analyse.
    
    Args:
        currency_pair (str): Währungspaar für die Analyse
        
    Returns:
        Dict: Kombinierte Daten mit Nachrichten und Kalenderereignissen
    """
    # Extrahiere die Währungen aus dem Paar
    base_currency = currency_pair[:3]
    quote_currency = currency_pair[3:]
    
    # Hole Nachrichten und Kalenderdaten
    news = get_forex_news(currency_pair)
    calendar_base = get_economic_calendar(days=7, currency=base_currency)
    calendar_quote = get_economic_calendar(days=7, currency=quote_currency)
    
    # Kombiniere Kalenderdaten
    calendar_events = calendar_base + calendar_quote
    
    # Sortiere Kalenderereignisse nach Datum
    calendar_events.sort(key=lambda x: x['datetime'])
    
    # Analysiere Nachrichten und Kalenderereignisse für Sentiment-Bewertung
    news_sentiment = analyze_news_sentiment(news, currency_pair)
    calendar_impact = analyze_calendar_impact(calendar_events, currency_pair)
    
    return {
        'news': news,
        'calendar_events': calendar_events,
        'news_sentiment': news_sentiment,
        'calendar_impact': calendar_impact,
        'base_currency': base_currency,
        'quote_currency': quote_currency
    }


def analyze_news_sentiment(news: List[Dict[str, Any]], currency_pair: str) -> Dict[str, Any]:
    """
    Analysiert Nachrichtenartikel, um eine Gesamtstimmung für das Währungspaar zu berechnen.
    
    Args:
        news (List[Dict]): Liste von Nachrichtenartikeln
        currency_pair (str): Währungspaar für die Analyse
        
    Returns:
        Dict: Sentiment-Metrik für das Währungspaar
    """
    base_currency = currency_pair[:3]
    quote_currency = currency_pair[3:]
    
    # Sentiment-Zähler für jede Währung
    base_positive = 0
    base_negative = 0
    quote_positive = 0
    quote_negative = 0
    
    # Relevanz-Gewichtung: Je neuer ein Artikel, desto wichtiger
    current_time = datetime.datetime.now()
    
    for article in news:
        # Stimmung und Währungen aus dem Artikel extrahieren
        sentiment = article.get('sentiment', 0)
        currencies = article.get('currencies', [])
        article_time = article.get('datetime', current_time)
        
        # Berechne Zeitgewichtung: Neuere Artikel haben mehr Einfluss
        if isinstance(article_time, datetime.datetime):
            time_diff = (current_time - article_time).total_seconds() / 3600  # Stunden
            time_weight = max(0.2, 1 - (time_diff / 72))  # Gewichtung zwischen 0.2 und 1 für Artikel bis zu 72 Stunden alt
        else:
            time_weight = 0.5  # Standardgewichtung für Artikel ohne Zeitstempel
        
        weighted_sentiment = sentiment * time_weight
        
        # Ordne Stimmung den jeweiligen Währungen zu
        if base_currency in currencies:
            if weighted_sentiment > 0:
                base_positive += weighted_sentiment
            else:
                base_negative += abs(weighted_sentiment)
                
        if quote_currency in currencies:
            if weighted_sentiment > 0:
                quote_positive += weighted_sentiment
            else:
                quote_negative += abs(weighted_sentiment)
    
    # Berechne Netto-Stimmung für jede Währung
    base_sentiment = base_positive - base_negative
    quote_sentiment = quote_positive - quote_negative
    
    # Relative Stimmung für das Paar: Positiv bedeutet Stärkung der Basiswährung gegenüber der Notierungswährung
    relative_sentiment = base_sentiment - quote_sentiment
    
    # Normalisiere auf eine Skala von -1 bis 1
    max_sentiment = max(0.1, abs(relative_sentiment))  # Verhindere Division durch Null
    normalized_sentiment = relative_sentiment / (max_sentiment + 5)  # +5 für Dämpfung
    
    # Bestimme Stärke der Stimmung
    sentiment_strength = min(1.0, (abs(base_sentiment) + abs(quote_sentiment)) / (len(news) + 1))
    
    # Relevante Überschriften für die Analyse
    relevant_headlines = [
        article['title'] for article in sorted(
            [a for a in news if base_currency in a.get('currencies', []) or quote_currency in a.get('currencies', [])],
            key=lambda x: abs(x.get('sentiment', 0)),
            reverse=True
        )[:3]  # Top 3 Schlagzeilen mit stärkster Stimmung
    ]
    
    return {
        'value': normalized_sentiment,  # Wert zwischen -1 und 1
        'strength': sentiment_strength,  # Stärke zwischen 0 und 1
        'direction': 'bullish' if normalized_sentiment > 0 else 'bearish',
        'relevant_headlines': relevant_headlines
    }


def analyze_calendar_impact(calendar_events: List[Dict[str, Any]], currency_pair: str) -> Dict[str, Any]:
    """
    Analysiert Wirtschaftskalenderereignisse, um ihre wahrscheinliche Auswirkung auf ein Währungspaar zu bewerten.
    
    Args:
        calendar_events (List[Dict]): Liste von Wirtschaftskalenderereignissen
        currency_pair (str): Währungspaar für die Analyse
        
    Returns:
        Dict: Kalenderauswirkung für das Währungspaar
    """
    base_currency = currency_pair[:3]
    quote_currency = currency_pair[3:]
    
    # Filtere Ereignisse, die in der nahen Zukunft liegen
    current_time = datetime.datetime.now()
    future_events = [
        event for event in calendar_events 
        if isinstance(event.get('datetime'), datetime.datetime) and event.get('datetime') > current_time
    ]
    
    # Sortiere nach Zeit und Wichtigkeit
    future_events.sort(key=lambda x: (
        x.get('datetime', current_time + datetime.timedelta(days=100)),  # Zeit
        -x.get('impact', 0)  # Wichtigkeit (negativ für absteigender Sortierung)
    ))
    
    # Berechne Auswirkung für jede Währung
    base_impact = 0
    quote_impact = 0
    
    # Wichtige bevorstehende Ereignisse extrahieren
    upcoming_high_impact = []
    
    for event in future_events:
        currency = event.get('currency', '')
        impact_level = event.get('impact', 0)
        
        # Berechne zeitliche Nähe: Je näher ein Ereignis, desto wichtiger
        event_time = event.get('datetime', current_time + datetime.timedelta(days=1))
        time_diff = (event_time - current_time).total_seconds() / 3600  # Stunden
        
        # Exponentieller Abfall der Relevanz mit der Zeit
        time_factor = max(0.1, 1.5 * (0.9 ** time_diff))  # Relevanz sinkt mit größerem Zeitabstand
        
        weighted_impact = impact_level * time_factor
        
        # Sammle wichtige bevorstehende Ereignisse
        if impact_level >= 2 and time_diff < 48:  # Hohe Auswirkung in den nächsten 48 Stunden
            upcoming_high_impact.append({
                'description': event.get('description', 'Unbekanntes Ereignis'),
                'datetime': event_time,
                'currency': currency,
                'impact': impact_level,
                'forecast': event.get('forecast', 'Keine Prognose'),
                'previous': event.get('previous', 'Keine früheren Daten')
            })
        
        # Ordne Auswirkung den jeweiligen Währungen zu
        if currency == base_currency:
            base_impact += weighted_impact
        elif currency == quote_currency:
            quote_impact += weighted_impact
    
    # Relative Auswirkung: Positiv bedeutet stärkerer Einfluss auf die Basiswährung
    relative_impact = base_impact - quote_impact
    
    # Normalisiere auf eine Skala von -1 bis 1
    max_impact = max(0.1, abs(relative_impact))  # Verhindere Division durch Null
    normalized_impact = relative_impact / (max_impact + 3)  # +3 für Dämpfung
    
    # Beschränke die Anzahl der zurückgegebenen Ereignisse
    upcoming_high_impact = upcoming_high_impact[:5]  # Maximal 5 Ereignisse
    
    return {
        'value': normalized_impact,  # Wert zwischen -1 und 1
        'base_currency_events': [e for e in upcoming_high_impact if e['currency'] == base_currency],
        'quote_currency_events': [e for e in upcoming_high_impact if e['currency'] == quote_currency],
        'direction': 'bullish' if normalized_impact > 0 else 'bearish',
        'has_high_impact_events': len(upcoming_high_impact) > 0
    }


# Hilfsfunktionen zum Generieren von realistischen Beispieldaten

def generate_realistic_forex_news() -> List[Dict[str, Any]]:
    """
    Generiert realistische Forex-Nachrichtenbeispiele
    """
    news = []
    
    # Liste der Hauptwährungen
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
    
    # Zentrale Banken und ihre zugehörigen Währungen
    central_banks = {
        'Fed': 'USD', 
        'EZB': 'EUR', 
        'BoE': 'GBP', 
        'BoJ': 'JPY',
        'SNB': 'CHF', 
        'RBA': 'AUD', 
        'BoC': 'CAD', 
        'RBNZ': 'NZD'
    }
    
    # Nachrichtenquellen
    sources = ['Bloomberg', 'CNBC', 'Reuters', 'Financial Times', 'MarketWatch', 'The Wall Street Journal']
    
    # Aktuelle Zeit und Zeitbereich für die Nachrichten (bis zu 48 Stunden zurück)
    current_time = datetime.datetime.now()
    
    # Allgemeine Wirtschaftsnachrichten
    economic_news_templates = [
        {'template': "{currency} steigt nach besser als erwarteten Wirtschaftsdaten", 'sentiment': 0.6},
        {'template': "{currency} fällt nach enttäuschenden Beschäftigungszahlen", 'sentiment': -0.6},
        {'template': "BIP-Wachstum in {region} übertrifft Erwartungen, {currency} profitiert", 'sentiment': 0.7},
        {'template': "Inflation in {region} steigt stärker als erwartet, belastet {currency}", 'sentiment': -0.5},
        {'template': "Handelsbilanzdefizit in {region} vergrößert sich, {currency} gibt nach", 'sentiment': -0.4},
        {'template': "Einzelhandelsumsätze in {region} besser als erwartet, stützen {currency}", 'sentiment': 0.5},
        {'template': "Arbeitsmarktdaten aus {region} schwächer als erwartet, {currency} unter Druck", 'sentiment': -0.6},
        {'template': "Wirtschaftsstimmung in {region} verbessert sich, {currency} legt zu", 'sentiment': 0.5},
        {'template': "Industrieproduktion in {region} sinkt unerwartet, {currency} schwächer", 'sentiment': -0.4},
        {'template': "Verbrauchervertrauen in {region} steigt, stützt {currency}", 'sentiment': 0.4}
    ]
    
    # Zentralbank-bezogene Nachrichten
    central_bank_news_templates = [
        {'template': "{bank} signalisiert weitere Zinserhöhungen, {currency} steigt", 'sentiment': 0.8},
        {'template': "{bank} deutet vorsichtigere Geldpolitik an, {currency} gibt nach", 'sentiment': -0.7},
        {'template': "{bank} belässt Zinsen unverändert, aber mit hawkishem Ausblick", 'sentiment': 0.5},
        {'template': "{bank} senkt Zinsen überraschend, {currency} fällt stark", 'sentiment': -0.9},
        {'template': "{bank}-Chef warnt vor wirtschaftlichen Risiken, {currency} unter Druck", 'sentiment': -0.6},
        {'template': "{bank} erhöht Zinsen um 25 Basispunkte wie erwartet", 'sentiment': 0.3},
        {'template': "{bank} hält an lockerer Geldpolitik fest, {currency} schwächer", 'sentiment': -0.5},
        {'template': "{bank} diskutiert Bilanzabbau, {currency} profitiert", 'sentiment': 0.6},
        {'template': "{bank}-Protokoll zeigt geteilte Meinungen zu Zinsausblick", 'sentiment': 0.0},
        {'template': "{bank} betont datenabhängigen Ansatz für künftige Zinsentscheidungen", 'sentiment': 0.1}
    ]
    
    # Geopolitische Nachrichten
    geopolitical_news_templates = [
        {'template': "Handelsstreit zwischen USA und China eskaliert, belastet globale Märkte", 'sentiment': -0.7, 'currencies': ['USD', 'CNH']},
        {'template': "Brexit-Unsicherheit belastet Pfund Sterling", 'sentiment': -0.6, 'currencies': ['GBP']},
        {'template': "Neue Sanktionen gegen Russland angekündigt, Auswirkungen auf Energiemärkte", 'sentiment': -0.4, 'currencies': ['EUR', 'RUB']},
        {'template': "Spannungen im Nahen Osten treiben Ölpreise, beeinflussen Währungen ölproduzierender Länder", 'sentiment': 0.3, 'currencies': ['CAD', 'NOK']},
        {'template': "USA und EU nähern sich Handelsabkommen, positiv für Euro und Dollar", 'sentiment': 0.5, 'currencies': ['EUR', 'USD']},
        {'template': "Politische Unsicherheit in {region} belastet lokale Währung", 'sentiment': -0.5},
        {'template': "Internationale Investoren fliehen aus Schwellenländern in sichere Häfen", 'sentiment': 0.6, 'currencies': ['USD', 'JPY', 'CHF']},
        {'template': "G20-Gipfel endet ohne konkrete Ergebnisse, Markterwartungen enttäuscht", 'sentiment': -0.3},
        {'template': "Neue Handelsvereinbarung zwischen {region1} und {region2} ankündigt", 'sentiment': 0.4},
        {'template': "Globale Wachstumssorgen dominieren Märkte, sichere Währungen gefragt", 'sentiment': 0.5, 'currencies': ['USD', 'JPY', 'CHF']}
    ]
    
    # Regionen für die Nachrichtentemplates
    regions = {
        'USD': 'den USA',
        'EUR': 'der Eurozone',
        'GBP': 'Großbritannien',
        'JPY': 'Japan',
        'CHF': 'der Schweiz',
        'AUD': 'Australien',
        'CAD': 'Kanada',
        'NZD': 'Neuseeland'
    }
    
    # Wirtschaftsnachrichten generieren
    for _ in range(15):
        # Zufällige Währung auswählen
        currency = random.choice(currencies)
        region = regions.get(currency, f"{currency}-Region")
        
        # Zufälliges Template auswählen
        template_data = random.choice(economic_news_templates)
        template = template_data['template']
        sentiment = template_data['sentiment']
        
        # Modifiziere Stimmung leicht für Variation
        sentiment_variation = random.uniform(-0.1, 0.1)
        adjusted_sentiment = sentiment + sentiment_variation
        
        # Zufällige Zeit in den letzten 48 Stunden
        hours_ago = random.uniform(0, 48)
        news_time = current_time - datetime.timedelta(hours=hours_ago)
        
        # Titel erstellen und Platzhalter ersetzen
        title = template.format(currency=currency, region=region)
        
        # Artikel erstellen
        article = {
            'title': title,
            'source': random.choice(sources),
            'datetime': news_time,
            'sentiment': adjusted_sentiment,
            'currencies': [currency],
            'type': 'economic'
        }
        
        news.append(article)
    
    # Zentralbank-Nachrichten generieren
    for _ in range(8):
        # Zufällige Zentralbank auswählen
        bank, currency = random.choice(list(central_banks.items()))
        
        # Zufälliges Template auswählen
        template_data = random.choice(central_bank_news_templates)
        template = template_data['template']
        sentiment = template_data['sentiment']
        
        # Modifiziere Stimmung leicht für Variation
        sentiment_variation = random.uniform(-0.1, 0.1)
        adjusted_sentiment = sentiment + sentiment_variation
        
        # Zufällige Zeit in den letzten 48 Stunden
        hours_ago = random.uniform(0, 48)
        news_time = current_time - datetime.timedelta(hours=hours_ago)
        
        # Titel erstellen und Platzhalter ersetzen
        title = template.format(bank=bank, currency=currency)
        
        # Artikel erstellen
        article = {
            'title': title,
            'source': random.choice(sources),
            'datetime': news_time,
            'sentiment': adjusted_sentiment,
            'currencies': [currency],
            'type': 'central_bank'
        }
        
        news.append(article)
    
    # Geopolitische Nachrichten generieren
    for _ in range(7):
        # Zufälliges Template auswählen
        template_data = random.choice(geopolitical_news_templates)
        template = template_data['template']
        sentiment = template_data['sentiment']
        
        # Modifiziere Stimmung leicht für Variation
        sentiment_variation = random.uniform(-0.1, 0.1)
        adjusted_sentiment = sentiment + sentiment_variation
        
        # Zufällige Zeit in den letzten 48 Stunden
        hours_ago = random.uniform(0, 48)
        news_time = current_time - datetime.timedelta(hours=hours_ago)
        
        # Betroffene Währungen
        if 'currencies' in template_data:
            affected_currencies = template_data['currencies']
        else:
            # Zufällige Währungen, wenn nicht spezifiziert
            num_currencies = random.randint(1, 3)
            affected_currencies = random.sample(currencies, num_currencies)
        
        # Platzhalter für Regionen
        region1 = regions.get(random.choice(currencies), "Region A")
        region2 = regions.get(random.choice(currencies), "Region B")
        
        # Titel erstellen und Platzhalter ersetzen
        title = template.format(region=region1, region1=region1, region2=region2)
        
        # Artikel erstellen
        article = {
            'title': title,
            'source': random.choice(sources),
            'datetime': news_time,
            'sentiment': adjusted_sentiment,
            'currencies': affected_currencies,
            'type': 'geopolitical'
        }
        
        news.append(article)
    
    # Sortiere Nachrichten nach Zeit (neueste zuerst)
    news.sort(key=lambda x: x['datetime'], reverse=True)
    
    return news


def generate_realistic_economic_calendar(days: int = 3, currency: str = None) -> List[Dict[str, Any]]:
    """
    Generiert realistische Wirtschaftskalendereinträge
    """
    calendar = []
    
    # Liste der Hauptwährungen, falls keine spezifiziert wird
    all_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
    currencies_to_include = [currency] if currency else all_currencies
    
    # Wirtschaftsindikatoren nach Währung
    indicators = {
        'USD': [
            {'name': 'Nonfarm Payrolls', 'impact': 3},
            {'name': 'Initial Jobless Claims', 'impact': 2},
            {'name': 'GDP', 'impact': 3},
            {'name': 'CPI', 'impact': 3},
            {'name': 'Retail Sales', 'impact': 2},
            {'name': 'ISM Manufacturing PMI', 'impact': 2},
            {'name': 'Fed Interest Rate Decision', 'impact': 3},
            {'name': 'Industrial Production', 'impact': 1},
            {'name': 'Consumer Confidence', 'impact': 2},
            {'name': 'Trade Balance', 'impact': 1},
            {'name': 'Housing Starts', 'impact': 1},
            {'name': 'FOMC Meeting Minutes', 'impact': 2}
        ],
        'EUR': [
            {'name': 'ECB Interest Rate Decision', 'impact': 3},
            {'name': 'Eurozone CPI', 'impact': 3},
            {'name': 'German Ifo Business Climate', 'impact': 2},
            {'name': 'Eurozone GDP', 'impact': 3},
            {'name': 'German ZEW Economic Sentiment', 'impact': 2},
            {'name': 'Eurozone Retail Sales', 'impact': 2},
            {'name': 'German Manufacturing PMI', 'impact': 2},
            {'name': 'German Unemployment Change', 'impact': 2},
            {'name': 'ECB Monetary Policy Statement', 'impact': 3},
            {'name': 'Eurozone Unemployment Rate', 'impact': 2}
        ],
        'GBP': [
            {'name': 'BOE Interest Rate Decision', 'impact': 3},
            {'name': 'UK CPI', 'impact': 3},
            {'name': 'UK GDP', 'impact': 3},
            {'name': 'UK Retail Sales', 'impact': 2},
            {'name': 'UK Manufacturing PMI', 'impact': 2},
            {'name': 'UK Unemployment Rate', 'impact': 2},
            {'name': 'BOE Monetary Policy Summary', 'impact': 3},
            {'name': 'UK Trade Balance', 'impact': 1},
            {'name': 'UK Services PMI', 'impact': 2},
            {'name': 'UK Construction PMI', 'impact': 1}
        ],
        'JPY': [
            {'name': 'BOJ Interest Rate Decision', 'impact': 3},
            {'name': 'Japan CPI', 'impact': 2},
            {'name': 'Japan GDP', 'impact': 3},
            {'name': 'Japan Trade Balance', 'impact': 2},
            {'name': 'Japan Manufacturing PMI', 'impact': 2},
            {'name': 'Japan Unemployment Rate', 'impact': 2},
            {'name': 'BOJ Outlook Report', 'impact': 3},
            {'name': 'Japan Industrial Production', 'impact': 2},
            {'name': 'Japan Retail Sales', 'impact': 2},
            {'name': 'Japan Tankan Manufacturing Index', 'impact': 2}
        ],
        'CHF': [
            {'name': 'SNB Interest Rate Decision', 'impact': 3},
            {'name': 'Swiss CPI', 'impact': 2},
            {'name': 'Swiss GDP', 'impact': 3},
            {'name': 'Swiss Retail Sales', 'impact': 2},
            {'name': 'Swiss SVME PMI', 'impact': 2},
            {'name': 'Swiss Trade Balance', 'impact': 2},
            {'name': 'SNB Monetary Policy Assessment', 'impact': 3},
            {'name': 'Swiss Unemployment Rate', 'impact': 2}
        ],
        'AUD': [
            {'name': 'RBA Interest Rate Decision', 'impact': 3},
            {'name': 'Australia CPI', 'impact': 3},
            {'name': 'Australia GDP', 'impact': 3},
            {'name': 'Australia Retail Sales', 'impact': 2},
            {'name': 'Australia Employment Change', 'impact': 3},
            {'name': 'Australia Trade Balance', 'impact': 2},
            {'name': 'RBA Rate Statement', 'impact': 3},
            {'name': 'Australia Unemployment Rate', 'impact': 3},
            {'name': 'Australia Building Approvals', 'impact': 2}
        ],
        'CAD': [
            {'name': 'BOC Interest Rate Decision', 'impact': 3},
            {'name': 'Canada CPI', 'impact': 3},
            {'name': 'Canada GDP', 'impact': 3},
            {'name': 'Canada Retail Sales', 'impact': 2},
            {'name': 'Canada Employment Change', 'impact': 3},
            {'name': 'Canada Trade Balance', 'impact': 2},
            {'name': 'BOC Rate Statement', 'impact': 3},
            {'name': 'Canada Unemployment Rate', 'impact': 3},
            {'name': 'Canada Ivey PMI', 'impact': 2}
        ],
        'NZD': [
            {'name': 'RBNZ Interest Rate Decision', 'impact': 3},
            {'name': 'New Zealand CPI', 'impact': 3},
            {'name': 'New Zealand GDP', 'impact': 3},
            {'name': 'New Zealand Retail Sales', 'impact': 2},
            {'name': 'New Zealand Trade Balance', 'impact': 2},
            {'name': 'RBNZ Rate Statement', 'impact': 3},
            {'name': 'New Zealand Unemployment Rate', 'impact': 3},
            {'name': 'New Zealand Business NZ PMI', 'impact': 2}
        ]
    }
    
    # Aktuelle Zeit
    current_time = datetime.datetime.now()
    
    # Generiere Ereignisse für die angegebene Anzahl von Tagen
    for day in range(days):
        # Datum für diesen Tag
        event_date = current_time + datetime.timedelta(days=day)
        
        for curr in currencies_to_include:
            if curr in indicators:
                # Anzahl der Ereignisse pro Tag und Währung (variiert, um realistischer zu sein)
                num_events = random.randint(0, 3 if day == 0 else 2)
                
                # Wähle zufällige Indikatoren für diese Währung aus
                if num_events > 0:
                    selected_indicators = random.sample(indicators[curr], min(num_events, len(indicators[curr])))
                    
                    for indicator in selected_indicators:
                        # Zufällige Zeit am Tag
                        hour = random.randint(8, 19)
                        minute = random.choice([0, 15, 30, 45])
                        event_datetime = event_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        
                        # Nur Ereignisse in der Zukunft oder am selben Tag hinzufügen
                        if event_datetime >= current_time or event_date.date() == current_time.date():
                            # Erzeuge Vorhersagen und vorherige Werte
                            is_percentage = "Rate" in indicator['name'] or "CPI" in indicator['name'] or "GDP" in indicator['name']
                            is_change = "Change" in indicator['name'] or "Employment" in indicator['name']
                            
                            if is_percentage:
                                previous = f"{random.uniform(0.1, 5.0):.1f}%"
                                forecast = f"{random.uniform(0.1, 5.0):.1f}%"
                            elif is_change:
                                previous = f"{random.randint(-50000, 100000)}"
                                forecast = f"{random.randint(-50000, 100000)}"
                            else:
                                previous = f"{random.uniform(40, 60):.1f}" if "PMI" in indicator['name'] else f"{random.uniform(-10, 10):.2f}B"
                                forecast = f"{random.uniform(40, 60):.1f}" if "PMI" in indicator['name'] else f"{random.uniform(-10, 10):.2f}B"
                            
                            # Ereignis erstellen
                            event = {
                                'currency': curr,
                                'datetime': event_datetime,
                                'description': indicator['name'],
                                'impact': indicator['impact'],  # 1 (niedrig) bis 3 (hoch)
                                'forecast': forecast,
                                'previous': previous
                            }
                            
                            calendar.append(event)
    
    # Sortiere Kalender nach Zeit
    calendar.sort(key=lambda x: x['datetime'])
    
    return calendar


def simulate_api_request(url: str = None) -> Dict[str, Any]:
    """Simulierte API-Anfrage für Testzwecke"""
    time.sleep(0.1)  # Simuliere Netzwerklatenz
    return {'status': 'success', 'data': {}}