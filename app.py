import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

from data_handler import get_forex_data, get_available_pairs
from signal_generator import generate_signals
from performance_tracker import get_performance_history, update_performance
from utils import format_percentage, get_confidence_color

# Page configuration
st.set_page_config(
    page_title="Forex Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für ein epischeres Erscheinungsbild
st.markdown("""
<style>
    /* Allgemeine Stylingverbesserungen */
    .stApp {
        background-image: linear-gradient(to bottom, #0f1c2e, #162236);
    }

    /* Verbesserte Überschriften */
    h1 {
        background: linear-gradient(90deg, #00c7b7, #5ce1e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        padding-bottom: 10px;
        border-bottom: 2px solid #00c7b7;
        margin-bottom: 30px !important;
    }

    h2 {
        color: #00c7b7 !important;
        font-weight: 700 !important;
        margin-top: 30px !important;
    }

    h3 {
        font-weight: 600 !important;
        margin-top: 20px !important;
        background: linear-gradient(90deg, #f1f1f1, #00c7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Container für Signale */
    .stExpander {
        border-left: 3px solid #00c7b7 !important;
        padding-left: 10px !important;
    }

    /* Bessere Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00c7b7, #00a99d) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
    }

    /* Verbesserte Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1a2c42 !important;
        border-right: 1px solid #304a6d !important;
    }

    /* Infoboxen */
    .stAlert {
        border-radius: 5px !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }

    /* Metrics mit Glow-Effekt */
    [data-testid="stMetric"] {
        background-color: rgba(26, 44, 66, 0.7) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 0 15px rgba(0, 199, 183, 0.2) !important;
        border: 1px solid rgba(0, 199, 183, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 0 20px rgba(0, 199, 183, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Datentabellenverbesserungen */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* Trading Signal Container */
    .element-container {
        background-color: rgba(26, 44, 66, 0.5) !important;
        border-radius: 8px !important;
        margin-bottom: 20px !important;
        padding: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }

    /* Animationen für Neue Signale */
    @keyframes newSignalGlow {
        0% { box-shadow: 0 0 5px rgba(0, 199, 183, 0.2); }
        50% { box-shadow: 0 0 20px rgba(0, 199, 183, 0.5); }
        100% { box-shadow: 0 0 5px rgba(0, 199, 183, 0.2); }
    }
    .new-signal {
        animation: newSignalGlow 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tracking viewed signals
if 'viewed_signals' not in st.session_state:
    st.session_state.viewed_signals = set()

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(hours=1)

# Initialize recent signals list to store signals from the last 20 minutes
if 'recent_signals' not in st.session_state:
    st.session_state.recent_signals = []

# Initialize last signal state to keep showing the most recent signal
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = None

# Sidebar navigation
# Verbesserte Sidebar-Navigation mit Icons und Styling
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <div style="background: linear-gradient(90deg, #00c7b7, #5ce1e6); height: 3px; margin-bottom: 20px;"></div>
    <h3 style="color: #00c7b7; font-weight: 700; letter-spacing: 1px;">NAVIGATION</h3>
    <div style="background: linear-gradient(90deg, #5ce1e6, #00c7b7); height: 3px; margin-top: 10px;"></div>
</div>
""", unsafe_allow_html=True)

# Erstelle Navigation mit Icons
page_icons = {
    "Trading Signals": "📊",
    "Performance History": "📈",
    "Contact": "✉️"
}

# Navigation-Optionen
pages = list(page_icons.keys())

# Erzeuge die Auswahlbox mit Icons
page_options = [f"{icon} {page}" for page, icon in page_icons.items()]
selected_page = st.sidebar.selectbox("Navigation", page_options, label_visibility="collapsed")

# Extrahiere den Seitennamen ohne Icon
page = selected_page.split(" ", 1)[1]

st.sidebar.markdown("<div style='background: linear-gradient(90deg, #00c7b7, #5ce1e6, #00c7b7); height: 2px; margin: 20px 0;'></div>", unsafe_allow_html=True)

# Verbesserte Markt-Update-Sektion
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, rgba(26, 44, 66, 0.8), rgba(15, 28, 46, 0.9));
            padding: 15px; border-radius: 10px; 
            border-left: 3px solid #00c7b7; margin-bottom: 20px;">
    <h3 style="color: #00c7b7; font-weight: 600; margin-bottom: 15px; display: flex; align-items: center;">
        <span style="margin-right: 10px;">⏱️</span> MARKT UPDATE
    </h3>
""", unsafe_allow_html=True)

# Show last update time mit formatierter Zeit
last_update_time = st.session_state.last_update.strftime("%d.%m.%Y %H:%M:%S")
st.sidebar.markdown(f"""
<div style="display: flex; justify-content: space-between; margin-bottom: 10px; padding: 8px; background-color: rgba(0,0,0,0.2); border-radius: 5px;">
    <span style="color: rgba(255,255,255,0.8);">Letztes Update:</span>
    <span style="color: #00c7b7; font-weight: bold;">{last_update_time}</span>
</div>
""", unsafe_allow_html=True)

# Update button mit info text
st.sidebar.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 15px;">
    <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; margin-bottom: 5px;">
        Die App generiert bis zu 5 Trades mit unterschiedlichen Konfidenzstufen (sicher, mittel, unsicher).
    </p>
</div>
""", unsafe_allow_html=True)

# Schließe die Container-Div
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Stylischer Update-Button
if st.sidebar.button("🔄 Markt Aktualisieren", use_container_width=True):
    st.session_state.last_update = datetime.now()
    st.rerun()

# Auto-refresh every minute
if datetime.now() - st.session_state.last_update > timedelta(minutes=1):
    st.session_state.last_update = datetime.now()
    st.rerun()

# Disclaimer in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Disclaimer
Die angezeigten Signale basieren auf KI-Analysen und stellen keine professionelle Finanzberatung dar. 
Handeln Sie auf eigenes Risiko.
""")

# Main content
if page == "Trading Signals":
    # Epischer Header mit Hintergrundeffekt und animiertem Logo
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2a2a2a, #404040);
                padding: 30px; border-radius: 15px; 
                margin-bottom: 30px; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2);">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
            <div style="width: 60px; height: 60px; margin-right: 20px; background: linear-gradient(135deg, #00c7b7, #5ce1e6); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(0, 199, 183, 0.5);">
                <span style="font-size: 30px;">📊</span>
            </div>
            <h1 style="margin: 0; background: linear-gradient(90deg, #00c7b7, #5ce1e6); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                      font-size: 48px; font-weight: 800; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                FOREX INTELLIGENCE
            </h1>
        </div>
        <p style="color: #f1f1f1; font-size: 18px; max-width: 800px; margin: 0 auto; letter-spacing: 1px;">
            Fortschrittliche KI-gestützte <span style="color: #00c7b7; font-weight: bold;">Handelsanalyse und Signalgenerierung</span> 
            für optimale Entscheidungen auf dem Forex-Markt.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Informationsbereich mit hübscher Grafik
    st.markdown("""
    <div style="display: flex; background: linear-gradient(135deg, rgba(0, 199, 183, 0.1), rgba(26, 44, 66, 0.3)); 
                border-radius: 10px; margin-bottom: 30px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <div style="padding: 20px; flex: 1;">
            <h3 style="color: #00c7b7; margin-bottom: 15px; font-weight: 600;">KI-gesteuerte Trading Signale</h3>
            <p style="color: #f1f1f1; margin-bottom: 15px;">
                Unsere fortschrittlichen Algorithmen analysieren kontinuierlich die Forex-Märkte, 
                identifizieren potenzielle Trading-Chancen und generieren präzise Handelssignale mit:
            </p>
            <ul style="color: #f1f1f1; padding-left: 20px;">
                <li>Optimalen Einstiegspunkten</li>
                <li>Strategischen Stop-Loss und Take-Profit Levels</li>
                <li>Risiko/Gewinn-Verhältnis von 1:3</li>
                <li>Konfidenzbewertungen für jedes Signal</li>
                <li>Empfehlungen zum besten Eintrittszeitpunkt</li>
            </ul>
        </div>
        <div style="width: 200px; display: flex; align-items: center; justify-content: center;">
            <div style="width: 150px; height: 150px; border-radius: 50%; 
                     background: linear-gradient(135deg, #00c7b7, #0f1c2e); 
                     display: flex; align-items: center; justify-content: center; 
                     box-shadow: 0 0 30px rgba(0, 199, 183, 0.3);">
                <span style="font-size: 70px;">🤖</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get real forex data for available pairs
    available_pairs = get_available_pairs()

    # Generate signals - multiple signals with different confidence levels
    with st.spinner("Analysiere Marktdaten..."):
        signals = generate_signals(available_pairs, max_signals=5)  # Generate up to 5 signals

    # If we have new signals, store them
    if not signals.empty:
        # Convert all signals to dict and add to recent signals
        for _, signal_row in signals.iterrows():
            signal_dict = signal_row.to_dict()

            # Store the best signal (highest confidence) as last_signal
            if not st.session_state.last_signal or signal_row.name == 0:
                st.session_state.last_signal = signal_dict

            # Add timestamp for recent signals tracking
            signal_dict['generated_at'] = datetime.now()

            # Add to recent signals list
            st.session_state.recent_signals.append(signal_dict)

    # Clean up old signals (keep only signals from the last 20 minutes)
    current_time = datetime.now()
    st.session_state.recent_signals = [
        signal for signal in st.session_state.recent_signals 
        if current_time - signal.get('generated_at', current_time) < timedelta(minutes=20)
    ]

    # Sort signals by timestamp (newest first)
    st.session_state.recent_signals.sort(key=lambda x: x.get('generated_at', datetime.now()), reverse=True)

    # Display signals from the last 20 minutes
    if st.session_state.recent_signals:
        st.markdown("## Aktuelle Trading Signale")
        st.markdown("##### Signale der letzten 20 Minuten")

        # Display each signal
        for i, signal in enumerate(st.session_state.recent_signals):
            # Determine confidence level
            confidence = signal['confidence']
            confidence_color = get_confidence_color(confidence)

            # Create a unique ID for the signal
            signal_id = f"{signal['pair']}-{signal['timestamp']}"

            # Check if this is a new signal
            is_new = signal_id not in st.session_state.viewed_signals

            with st.container():
                cols = st.columns([3, 1])

                with cols[0]:
                    # Signal header with NEW badge if applicable
                    header_text = f"{signal['pair']} - {signal['action'].upper()}"
                    if is_new:
                        header_text = f"{header_text} 🆕"
                        # Add to viewed signals
                        st.session_state.viewed_signals.add(signal_id)

                    st.markdown(f"### {header_text}")

                    # Signal details
                    timestamp = signal['timestamp']
                    entry_price = signal['entry_price']
                    stop_loss = signal['stop_loss']
                    take_profit = signal['take_profit']
                    risk_reward = signal['risk_reward_ratio']
                    trade_duration = signal.get('estimated_duration', 'Nicht verfügbar')
                    confidence_label = {
                        'sicher': "🟢 Hohe Konfidenz",
                        'mittel': "🟡 Mittlere Konfidenz",
                        'unsicher': "🔴 Niedrige Konfidenz"
                    }[signal['confidence']]

                    # Calculate pips for SL and TP
                    pip_multiplier = 100 if 'JPY' in signal['pair'] else 10000
                    sl_pips = abs(entry_price - stop_loss) * pip_multiplier
                    tp_pips = abs(take_profit - entry_price) * pip_multiplier

                    # Detailed trade info box
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(26, 44, 66, 0.8), rgba(15, 28, 46, 0.95));
                                border-left: 4px solid #00c7b7;
                                padding: 20px;
                                border-radius: 8px;
                                margin-bottom: 20px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                        <div style="display: grid; grid-template-columns: auto 1fr; gap: 10px; align-items: center;">
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Zeitstempel:</span>
                            <span>{timestamp}</span>
                            
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Konfidenz:</span>
                            <span>{confidence_label}</span>
                            
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Entry:</span>
                            <span>{entry_price:.5f}</span>
                            
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Stop-Loss:</span>
                            <span>{stop_loss:.5f} ({sl_pips:.1f} Pips)</span>
                            
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Take-Profit:</span>
                            <span>{take_profit:.5f} ({tp_pips:.1f} Pips)</span>
                            
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Risiko/Belohnung:</span>
                            <span>1:{risk_reward:.1f}</span>
                            
                            <span style="font-weight: bold; color: rgba(255,255,255,0.8);">Geschätzte Dauer:</span>
                            <span>{trade_duration}</span>
                        </div>

                        <div style="margin-top: 20px; padding: 15px; background: rgba(0, 199, 183, 0.1); border-radius: 5px; border-left: 3px solid #00c7b7;">
                            <p style="margin: 0; color: #f1f1f1; font-size: 0.9em;">
                                <strong>💡 Trading-Tipp:</strong> Wenn du erfolgreich traden willst, nutze maximal 1–2 % deines Kapitals pro Trade 
                                und strebe ein Chance-Risiko-Verhältnis von mindestens 1.5:1 an – so gewinnst du langfristig auch bei einzelnen Verlusten. 
                                Nutze Leverage nur vorsichtig (2–5x) und nur dann, wenn du deinen Stop-Loss klar definiert hast. 
                                Der Schlüssel zum Erfolg ist Disziplin, nicht Größe – kleine, kontrollierte Schritte schlagen wilde Wetten.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Signal analysis
                    with st.expander("Analyse anzeigen"):
                        st.write(signal['analysis'])

                    # Entfernt: Kein Disclaimer mehr unterhalb jedes Signals

                with cols[1]:
                    # Get price chart for the pair
                    pair_data = get_forex_data(signal['pair'], '1h', 24)

                    if not pair_data.empty:
                        # Erstelle einen eindrucksvolleren Chart mit besserer Optik
                        fig = go.Figure(data=[go.Candlestick(
                            x=pair_data.index,
                            open=pair_data['Open'],
                            high=pair_data['High'],
                            low=pair_data['Low'],
                            close=pair_data['Close'],
                            increasing_line_color='#00c7b7',  # Türkis für steigende Kerzen
                            decreasing_line_color='#ef4056',  # Rot für fallende Kerzen
                            increasing_fillcolor='rgba(0, 199, 183, 0.3)',  # Transparentes Türkis
                            decreasing_fillcolor='rgba(239, 64, 86, 0.3)',  # Transparentes Rot
                            line=dict(width=1)
                        )])

                        # Add entry, SL and TP lines mit besseren Farben und Beschriftungen
                        entry_price = signal['entry_price'].item() if hasattr(signal['entry_price'], 'item') else signal['entry_price']
                        stop_loss = signal['stop_loss'].item() if hasattr(signal['stop_loss'], 'item') else signal['stop_loss']
                        take_profit = signal['take_profit'].item() if hasattr(signal['take_profit'], 'item') else signal['take_profit']

                        # Berechne möglichen Gewinn und Verlust
                        risk_pips = abs(entry_price - stop_loss)
                        reward_pips = abs(take_profit - entry_price)

                        # Add fancy lines mit Annotationen
                        action_color = "#00c7b7" if signal['action'] == 'buy' else "#ef4056"

                        # Berechne Pip-Werte für Chart-Anzeige
                        sl_pips_chart = abs(entry_price - stop_loss) * (100 if 'JPY' in signal['pair'] else 10000)
                        tp_pips_chart = abs(take_profit - entry_price) * (100 if 'JPY' in signal['pair'] else 10000)

                        # Entry-Linie
                        fig.add_hline(
                            y=entry_price, 
                            line_dash="solid", 
                            line_color="#ffcc00",  # Gold für Entry
                            line_width=2,
                            annotation_text=f"ENTRY {entry_price:.5f}",
                            annotation_position="right",
                            annotation_font_color="#ffcc00",
                            annotation_font_size=14,
                            annotation_bgcolor="rgba(15, 28, 46, 0.8)"  # Halbtransparenter dunkler Hintergrund
                        )

                        # Stop-Loss-Linie
                        fig.add_hline(
                            y=stop_loss, 
                            line_dash="dash", 
                            line_color="#ef4056",  # Rot für Stop Loss
                            line_width=2,
                            annotation_text=f"SL {stop_loss:.5f} ({sl_pips_chart:.1f} pips)",
                            annotation_position="right",
                            annotation_font_color="#ef4056",
                            annotation_font_size=14,
                            annotation_bgcolor="rgba(15, 28, 46, 0.8)"
                        )

                        # Take-Profit-Linie
                        fig.add_hline(
                            y=take_profit, 
                            line_dash="dash", 
                            line_color="#00c7b7",  # Türkis für Take Profit
                            line_width=2,
                            annotation_text=f"TP {take_profit:.5f} ({tp_pips_chart:.1f} pips)",
                            annotation_position="right",
                            annotation_font_color="#00c7b7",
                            annotation_font_size=14,
                            annotation_bgcolor="rgba(15, 28, 46, 0.8)"
                        )

                        # Diagramm-Layout verbessern
                        fig.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_rangeslider_visible=False,
                            plot_bgcolor='rgba(15, 28, 46, 1)',  # Dunkler Hintergrund
                            paper_bgcolor='rgba(15, 28, 46, 0)',  # Transparenter Hintergrund um das Diagramm
                            font=dict(color='#f1f1f1'),  # Weiße Schrift
                            title=dict(
                                text=f"{signal['pair']} Chart | {signal['action'].upper()}", 
                                font=dict(size=16, color="#00c7b7")
                            ),
                            xaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(255, 255, 255, 0.1)',
                                title=None
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(255, 255, 255, 0.1)',
                                title=None
                            )
                        )

                        # Füge Wasserzeichen hinzu
                        fig.add_annotation(
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            text="FOREX INTELLIGENCE",
                            showarrow=False,
                            font=dict(size=25, color="rgba(255, 255, 255, 0.05)"),
                            align="center",
                            opacity=0.6,
                            textangle=30
                        )

                        # Plaziere den Chart mit einem einzigartigen Key für jedes Signal
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{signal_id}")

                        # Zeige ein visuelles Konfidenz-Badge an
                        confidence_badge = {
                            'sicher': "🟢 Hohe Konfidenz", 
                            'mittel': "🟠 Mittlere Konfidenz", 
                            'unsicher': "🔴 Niedrige Konfidenz"
                        }
                        st.markdown(f"<div style='text-align: center; padding: 5px; background-color: rgba(26, 44, 66, 0.7); border-radius: 5px;'>{confidence_badge.get(confidence, 'Unbekannt')}</div>", unsafe_allow_html=True)
                    else:
                        # Verbesserte Fehlermeldung
                        st.error("📊 Chart Daten nicht verfügbar - Bitte aktualisieren Sie die Seite.")

                st.markdown("---")
    else:
        # This should only happen on the very first run
        st.info("Aktuell sind keine Trading Signale verfügbar. Bitte klicken Sie auf 'Aktualisieren', um ein Signal zu generieren.")

elif page == "Performance History":
    # Epischer Header für die Performance-Seite
    st.markdown("""
    <div style="background-image: linear-gradient(to right, rgba(15, 28, 46, 0.8), rgba(26, 44, 66, 0.8)), 
                url('https://i.imgur.com/7Ty2kJM.png'); 
                background-size: cover; padding: 30px; border-radius: 15px; 
                margin-bottom: 30px; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2);">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
            <div style="width: 60px; height: 60px; margin-right: 20px; background: linear-gradient(135deg, #00c7b7, #5ce1e6); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(0, 199, 183, 0.5);">
                <span style="font-size: 30px;">📈</span>
            </div>
            <h1 style="margin: 0; background: linear-gradient(90deg, #00c7b7, #5ce1e6); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                      font-size: 48px; font-weight: 800; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                PERFORMANCE HISTORY
            </h1>
        </div>
        <p style="color: #f1f1f1; font-size: 18px; max-width: 800px; margin: 0 auto; letter-spacing: 1px;">
            <span style="color: #00c7b7; font-weight: bold;">Detaillierte Analyse und Tracking</span> 
            der historischen Signalperformance und Handelsgewinne im Zeitverlauf.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get performance data
    performance_data = get_performance_history()

    if not performance_data.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            success_rate = len(performance_data[performance_data['result'] == 'success']) / len(performance_data) * 100
            st.metric("Erfolgsrate", f"{success_rate:.1f}%")

        with col2:
            total_profit = performance_data['profit_loss_pips'].sum()
            st.metric("Gesamtgewinn/-verlust", f"{total_profit:.1f} Pips")

        with col3:
            average_profit = performance_data[performance_data['result'] == 'success']['profit_loss_pips'].mean()
            st.metric("Ø Gewinn", f"{average_profit:.1f} Pips")

        with col4:
            average_loss = performance_data[performance_data['result'] == 'failure']['profit_loss_pips'].mean()
            st.metric("Ø Verlust", f"{average_loss:.1f} Pips")

        # Performance by confidence level
        st.subheader("Performance nach Konfidenz")

        confidence_df = performance_data.groupby('confidence').agg({
            'result': lambda x: (x == 'success').mean() * 100,
            'profit_loss_pips': 'sum'
        }).reset_index()

        confidence_df.columns = ['Konfidenz', 'Erfolgsrate (%)', 'Pips']

        col1, col2 = st.columns(2)

        with col1:
            # Verbesserte Erfolgsraten-Visualisierung
            fig_success = go.Figure()

            # Balkendiagramm mit Farbverlauf nach Konfidenz
            color_map = {
                'unsicher': '#ef4056',  # Rot
                'mittel': '#ffcc00',    # Gold
                'sicher': '#00c7b7'     # Türkis
            }

            # Sortieren nach Zuverlässigkeit (niedrig zu hoch)
            confidence_order = ['unsicher', 'mittel', 'sicher']
            confidence_df_sorted = confidence_df.set_index('Konfidenz').loc[confidence_order].reset_index()

            # Hinzufügen der Balken
            fig_success.add_trace(go.Bar(
                x=confidence_df_sorted['Konfidenz'],
                y=confidence_df_sorted['Erfolgsrate (%)'],
                marker=dict(
                    color=[color_map.get(conf, '#777777') for conf in confidence_df_sorted['Konfidenz']],
                    line=dict(width=1, color='rgba(255, 255, 255, 0.5)')
                ),
                text=confidence_df_sorted['Erfolgsrate (%)'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
            ))

            # Styling verbessern
            fig_success.update_layout(
                title=dict(
                    text="Erfolgsrate nach Konfidenz",
                    font=dict(size=24, color="#00c7b7"),
                    x=0.5,
                    y=0.95
                ),
                xaxis=dict(
                    title="Konfidenz",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14),
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    title="Erfolgsrate (%)",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14),
                    range=[0, 100],
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                plot_bgcolor='rgba(15, 28, 46, 1)',
                paper_bgcolor='rgba(15, 28, 46, 0)',
                font=dict(color='#f1f1f1'),
                margin=dict(l=40, r=40, t=80, b=40),
                bargap=0.2
            )

            # Linien für visuelle Referenz hinzufügen
            fig_success.add_shape(
                type="line",
                x0=-0.5, x1=2.5,
                y0=50, y1=50,
                line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash")
            )

            # Hinzufügen von Beschriftung für die Referenzlinie
            fig_success.add_annotation(
                x=2.4, y=50,
                text="50%",
                showarrow=False,
                font=dict(color="rgba(255, 255, 255, 0.5)")
            )

            # Wasserzeichen hinzufügen
            fig_success.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                text="FOREX INTELLIGENCE",
                showarrow=False,
                font=dict(size=25, color="rgba(255, 255, 255, 0.05)"),
                align="center",
                opacity=0.6,
                textangle=30
            )

            st.plotly_chart(fig_success, use_container_width=True, key="confidence_success_chart")

        with col2:
            # Verbesserte Pips-Visualisierung
            fig_pips = go.Figure()

            # Berechne positive und negative Werte
            confidence_df_sorted['color'] = confidence_df_sorted['Pips'].apply(
                lambda x: '#00c7b7' if x > 0 else '#ef4056'
            )

            # Balkendiagramm
            fig_pips.add_trace(go.Bar(
                x=confidence_df_sorted['Konfidenz'],
                y=confidence_df_sorted['Pips'],
                marker=dict(
                    color=confidence_df_sorted['color'],
                    line=dict(width=1, color='rgba(255, 255, 255, 0.5)')
                ),
                text=confidence_df_sorted['Pips'].apply(lambda x: f"{x:.1f}"),
                textposition='auto',
                hovertemplate='%{x}: %{y:.1f} Pips<extra></extra>'
            ))

            # Styling verbessern
            fig_pips.update_layout(
                title=dict(
                    text="Gewinn/Verlust nach Konfidenz",
                    font=dict(size=24, color="#00c7b7"),
                    x=0.5,
                    y=0.95
                ),
                xaxis=dict(
                    title="Konfidenz",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14),
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    title="Pips",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14),
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    zeroline=True,
                    zerolinecolor='rgba(255, 255, 255, 0.3)',
                    zerolinewidth=2
                ),
                plot_bgcolor='rgba(15, 28, 46, 1)',
                paper_bgcolor='rgba(15, 28, 46, 0)',
                font=dict(color='#f1f1f1'),
                margin=dict(l=40, r=40, t=80, b=40),
                bargap=0.2
            )

            # Wasserzeichen hinzufügen
            fig_pips.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                text="FOREX INTELLIGENCE",
                showarrow=False,
                font=dict(size=25, color="rgba(255, 255, 255, 0.05)"),
                align="center",
                opacity=0.6,
                textangle=30
            )

            st.plotly_chart(fig_pips, use_container_width=True, key="confidence_pips_chart")

        # Performance over time
        st.subheader("Performance im Zeitverlauf")

        # Group data by date and calculate cumulative performance
        performance_data['date'] = pd.to_datetime(performance_data['timestamp']).dt.date
        time_performance = performance_data.groupby('date')['profit_loss_pips'].sum().reset_index()
        time_performance['cumulative'] = time_performance['profit_loss_pips'].cumsum()

        # Erstelle ein modernes und beeindruckendes Performance-Diagramm
        fig_time = go.Figure()

        # Füge Balken für tägliche Performance mit verbesserten Farben hinzu
        fig_time.add_trace(go.Bar(
            x=time_performance['date'],
            y=time_performance['profit_loss_pips'],
            name='Täglicher Gewinn/Verlust',
            marker_color=time_performance['profit_loss_pips'].apply(
                lambda x: '#00c7b7' if x > 0 else '#ef4056'  # Türkis für positiv, Rot für negativ
            ),
            opacity=0.8,
            hovertemplate='%{x|%d.%m.%Y}: %{y:.1f} Pips<extra></extra>'
        ))

        # Füge eine Area-Füllung unter der kumulativen Linie hinzu
        cumulative_color = '#00c7b7' if time_performance['cumulative'].iloc[-1] >= 0 else '#ef4056'

        # Transparenter Farbverlauf unter der Linie 
        fig_time.add_trace(go.Scatter(
            x=time_performance['date'],
            y=time_performance['cumulative'],
            mode='none',
            fill='tozeroy',
            fillcolor=f'rgba({", ".join(str(int(cumulative_color[i:i+2], 16)) for i in (1, 3, 5))}, 0.2)',
            hoverinfo='skip',
            showlegend=False
        ))

        # Füge die Hauptlinie für kumulativen Gewinn/Verlust hinzu
        fig_time.add_trace(go.Scatter(
            x=time_performance['date'],
            y=time_performance['cumulative'],
            mode='lines+markers',
            name='Kumulativer Gewinn/Verlust',
            line=dict(
                color=cumulative_color,
                width=3,
                shape='spline',  # Sanfte Kurven
                smoothing=1.3
            ),
            marker=dict(
                size=8,
                color=cumulative_color,
                line=dict(
                    color='white',
                    width=1
                )
            ),
            hovertemplate='%{x|%d.%m.%Y}: %{y:.1f} Pips (Gesamt)<extra></extra>'
        ))

        # Füge eine horizontale Linie bei 0 hinzu
        fig_time.add_shape(
            type="line",
            x0=time_performance['date'].min(),
            x1=time_performance['date'].max(),
            y0=0, y1=0,
            line=dict(
                color="rgba(255, 255, 255, 0.5)",
                width=1,
                dash="dash"
            )
        )

        # Füge Wasserzeichen hinzu
        fig_time.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="FOREX INTELLIGENCE",
            showarrow=False,
            font=dict(size=30, color="rgba(255, 255, 255, 0.05)"),
            align="center"
        )

        # Verbessere das Layout
        fig_time.update_layout(
            title=dict(
                text="Performance im Zeitverlauf",
                font=dict(size=24, color="#00c7b7"),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="Datum",
                titlefont=dict(size=16),
                tickfont=dict(size=14),
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickformat='%d.%m.%Y'
            ),
            yaxis=dict(
                title="Pips",
                titlefont=dict(size=16),
                tickfont=dict(size=14),
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
                zerolinewidth=2
            ),
            plot_bgcolor='rgba(15, 28, 46, 1)',
            paper_bgcolor='rgba(15, 28, 46, 0)',
            font=dict(color='#f1f1f1'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(15, 28, 46, 0.5)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode="x unified"
        )

        # Zeige das Diagram
        st.plotly_chart(fig_time, use_container_width=True, key="performance_time_chart")

        # Historical signals table
        st.subheader("Historische Signale")

        # Add status column for display
        performance_data['status'] = performance_data['result'].apply(
            lambda x: "✅ Erfolg" if x == 'success' else "❌ Fehlschlag"
        )

        # Format the table
        table_data = performance_data[['timestamp', 'pair', 'action', 'confidence', 'status', 'profit_loss_pips']]
        table_data.columns = ['Zeitstempel', 'Währungspaar', 'Aktion', 'Konfidenz', 'Status', 'Gewinn/Verlust (Pips)']

        st.dataframe(
            table_data.sort_values('Zeitstempel', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Noch keine Performance-Daten verfügbar. Es werden Daten gesammelt, sobald Signale generiert werden.")

elif page == "Contact":
    st.title("Kontakt")

    # Erstelle ein 2-spaltiges Layout für die Kontaktseite
    contact_col1, contact_col2 = st.columns([3, 2])

    with contact_col1:
        # Stylisches Kontaktformular
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(26, 44, 66, 0.8), rgba(15, 28, 46, 0.9)); 
                    padding: 25px; border-radius: 15px; 
                    border-left: 4px solid #00c7b7; box-shadow: 0 10px 20px rgba(0,0,0,0.3);">
            <h3 style="color: #00c7b7; margin-bottom: 20px; font-weight: 600;">Haben Sie Fragen oder Anregungen?</h3>
            <p style="margin-bottom: 20px;">Forex Intelligence ist ein KI-basiertes Tool zur Analyse von Forex-Märkten. Wir freuen uns über Ihr Feedback!</p>
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background-color: #00c7b7; border-radius: 50%; height: 36px; width: 36px; display: flex; justify-content: center; align-items: center; margin-right: 15px;">
                    <span style="color: white; font-size: 18px;">✉️</span>
                </div>
                <div>
                    <p style="margin: 0; font-weight: bold; color: white;">E-Mail:</p>
                    <p style="margin: 0; color: #00c7b7;"><a href="mailto:thomasbrot@proton.me" style="color: #00c7b7; text-decoration: none;">thomasbrot@proton.me</a></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Verbessertes Formular
        st.markdown("<h3 style='color: #00c7b7; margin-top: 30px;'>Feedback-Formular</h3>", unsafe_allow_html=True)

        with st.form("feedback_form"):
            # Formularfelder mit verbesserten Beschreibungen
            name = st.text_input("Name", placeholder="Ihr vollständiger Name")
            email = st.text_input("E-Mail", placeholder="ihre.email@beispiel.de")
            subject = st.selectbox("Betreff", options=[
                "Allgemeines Feedback",
                "Feature-Vorschlag",
                "Technisches Problem melden",
                "Partnerschaftsanfrage",
                "Sonstiges"
            ])
            message = st.text_area("Nachricht", placeholder="Beschreiben Sie Ihr Anliegen detailliert...", height=150)

            # Datenschutzhinweis
            st.markdown("""
            <small style='color: rgba(255, 255, 255, 0.7);'>
            Mit dem Absenden stimmen Sie zu, dass Ihre Daten zur Bearbeitung Ihrer Anfrage gespeichert werden.
            Ihre Daten werden vertraulich behandelt und nicht an Dritte weitergegeben.
            </small>
            """, unsafe_allow_html=True)

            # Verbesserter Submit-Button
            submit_button = st.form_submit_button("Nachricht senden")

            if submit_button:
                if name and email and message:
                    st.success("✅ Vielen Dank für Ihr Feedback! Wir werden uns innerhalb von 48 Stunden bei Ihnen melden.")
                else:
                    st.error("⚠️ Bitte füllen Sie alle erforderlichen Felder aus.")

    with contact_col2:
        # Epische Visualisierung für die Kontaktseite
        # Erstelle ein Plotly-Diagramm, das Währungssymbole visualisiert

        import random
        import numpy as np

        # Erstelle ein beeindruckendes 3D-Scatter-Diagramm mit Währungssymbolen
        forex_symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'EUR/GBP', 'AUD/USD', 'USD/CAD', 'NZD/USD']
        n_points = 100

        # Generiere Zufallsdaten für 3D-Scatter
        random.seed(42)  # Für konsistente Visualisierung
        x = np.random.normal(0, 1, n_points)
        y = np.random.normal(0, 1, n_points)
        z = np.random.normal(0, 1, n_points)

        # Wähle zufällige Symbole
        symbols = [random.choice(forex_symbols) for _ in range(n_points)]

        # Erstelle 3D-Scatter-Plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers+text',
            marker=dict(
                size=8,
                color=z,
                colorscale='Turbo',
                opacity=0.8,
                colorbar=dict(title="Market Volatility"),
                symbol='circle',
            ),
            text=symbols,
            hoverinfo='text'
        )])

        # Layout anpassen
        fig.update_layout(
            title=dict(
                text="Forex Markt Visualisierung",
                font=dict(size=22, color="#00c7b7", family="Arial Black"),
                x=0.5,
                y=0.95
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    showticklabels=False,
                    title=''
                ),
                yaxis=dict(
                    showbackground=False,
                    showticklabels=False,
                    title=''
                ),
                zaxis=dict(
                    showbackground=False,
                    showticklabels=False,
                    title=''
                ),
                bgcolor='rgba(15, 28, 46, 0)'
            ),
            paper_bgcolor='rgba(15, 28, 46, 0)',
            plot_bgcolor='rgba(15, 28, 46, 0)',
            font=dict(color='#f1f1f1'),
            autosize=True,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Über-Box mit Schattierung und Highlight-Effekten
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(26, 44, 66, 0.8), rgba(15, 28, 46, 0.9));
                    padding: 25px; border-radius: 15px; margin-top: 20px; 
                    border-right: 4px solid #00c7b7; box-shadow: 0 10px 20px rgba(0,0,0,0.3);">
            <h3 style="color: #00c7b7; margin-bottom: 20px; font-weight: 600;">Über Forex Intelligence</h3>

            <p style="margin-bottom: 15px;">Forex Intelligence nutzt fortschrittliche KI-Algorithmen, um Forex-Märkte zu analysieren und präzise Handelssignale zu generieren.</p>

            <h4 style="color: #ffcc00; margin: 20px 0 10px 0; font-weight: 500;">Features:</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="padding: 6px 0; display: flex; align-items: center;">
                    <span style="color: #00c7b7; margin-right: 10px;">▶</span> KI-gestützte Marktanalyse in Echtzeit
                </li>
                <li style="padding: 6px 0; display: flex; align-items: center;">
                    <span style="color: #00c7b7; margin-right: 10px;">▶</span> Trading Signale mit Stop-Loss und Take-Profit
                </li>
                <li style="padding: 6px 0; display: flex; align-items: center;">
                    <span style="color: #00c7b7; margin-right: 10px;">▶</span> Risikoanalyse mit Konfidenz-Bewertungen
                </li>
                <li style="padding: 6px 0; display: flex; align-items: center;">
                    <span style="color: #00c7b7; margin-right: 10px;">▶</span> Performance-Tracking und -Analyse
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer am Ende der Seite
    st.markdown("""
    <div style="margin-top: 40px; padding: 15px; background-color: rgba(0, 0, 0, 0.2); border-radius: 10px; text-align: center;">
        <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9em;">
            <strong>Haftungsausschluss:</strong> Forex Intelligence bietet keine professionelle Finanzberatung. 
            Alle Signale basieren auf Algorithmen und historischen Daten. 
            Trading auf den Forex-Märkten beinhaltet Risiken und kann zum Verlust Ihres investierten Kapitals führen.
        </p>
    </div>
    """, unsafe_allow_html=True)