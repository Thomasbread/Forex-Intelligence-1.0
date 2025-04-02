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

# Initialize session state for tracking viewed signals
if 'viewed_signals' not in st.session_state:
    st.session_state.viewed_signals = set()

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(hours=1)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Trading Signals", "Performance History", "Contact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Markt Update")
# Show last update time
last_update_time = st.session_state.last_update.strftime("%d.%m.%Y %H:%M:%S")
st.sidebar.text(f"Letztes Update: {last_update_time}")

# Update button with info text
st.sidebar.info("Die App generiert nur 1 Trade pro Minute mit hoher Sicherheit (\"sicher\" Konfidenz).")
if st.sidebar.button("Aktualisieren"):
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
    st.title("Forex Intelligence")
    st.subheader("KI-generierte Forex Trading Signale")
    
    # Get real forex data for available pairs
    available_pairs = get_available_pairs()
    
    # Generate signals
    with st.spinner("Analysiere Marktdaten..."):
        signals = generate_signals(available_pairs)
    
    if not signals.empty:
        # Group signals by confidence level
        confidence_groups = {
            "Sicher": signals[signals['confidence'] == 'sicher'],
            "Mittel": signals[signals['confidence'] == 'mittel'],
            "Unsicher": signals[signals['confidence'] == 'unsicher']
        }
        
        # Display signals by confidence level
        for confidence, group_signals in confidence_groups.items():
            if not group_signals.empty:
                st.markdown(f"## {confidence} Signale")
                
                for idx, signal in group_signals.iterrows():
                    # Create a unique ID for each signal
                    signal_id = f"{signal['pair']}-{signal['timestamp']}"
                    
                    # Check if this is a new signal
                    is_new = signal_id not in st.session_state.viewed_signals
                    
                    # Signal card with border color based on confidence
                    confidence_color = get_confidence_color(signal['confidence'])
                    
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
                            # Convert Series values to Python scalar values if needed
                            entry_price = signal['entry_price'].item() if hasattr(signal['entry_price'], 'item') else signal['entry_price']
                            stop_loss = signal['stop_loss'].item() if hasattr(signal['stop_loss'], 'item') else signal['stop_loss']
                            take_profit = signal['take_profit'].item() if hasattr(signal['take_profit'], 'item') else signal['take_profit']
                            risk_reward = signal['risk_reward_ratio'].item() if hasattr(signal['risk_reward_ratio'], 'item') else signal['risk_reward_ratio']
                            timestamp = signal['timestamp']
                            
                            st.markdown(f"""
                            **Einstiegspunkt:** {entry_price:.5f}  
                            **Stop Loss:** {stop_loss:.5f}  
                            **Take Profit:** {take_profit:.5f}  
                            **Risiko/Gewinn Verhältnis:** 1:{risk_reward}  
                            **Zeitstempel:** {timestamp}
                            """)
                            
                            # Signal analysis
                            st.markdown("#### Analyse")
                            st.write(signal['analysis'])
                            
                            # Disclaimer for each signal
                            st.info("⚠️ Hinweis: Dies ist keine professionelle Finanzberatung, sondern basiert auf KI-Analysen.")
                        
                        with cols[1]:
                            # Get price chart for the pair
                            pair_data = get_forex_data(signal['pair'], '1h', 24)
                            
                            if not pair_data.empty:
                                fig = go.Figure(data=[go.Candlestick(
                                    x=pair_data.index,
                                    open=pair_data['Open'],
                                    high=pair_data['High'],
                                    low=pair_data['Low'],
                                    close=pair_data['Close']
                                )])
                                
                                # Add entry, SL and TP lines
                                entry_price = signal['entry_price'].item() if hasattr(signal['entry_price'], 'item') else signal['entry_price']
                                stop_loss = signal['stop_loss'].item() if hasattr(signal['stop_loss'], 'item') else signal['stop_loss']
                                take_profit = signal['take_profit'].item() if hasattr(signal['take_profit'], 'item') else signal['take_profit']
                                
                                fig.add_hline(y=entry_price, line_dash="solid", 
                                             line_color="yellow", annotation_text="Entry")
                                fig.add_hline(y=stop_loss, line_dash="dash", 
                                             line_color="red", annotation_text="SL")
                                fig.add_hline(y=take_profit, line_dash="dash", 
                                             line_color="green", annotation_text="TP")
                                
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    xaxis_rangeslider_visible=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Chart Daten nicht verfügbar")
                        
                        st.markdown("---")
    else:
        st.info("Aktuell sind keine Trading Signale verfügbar. Bitte versuchen Sie es später erneut.")

elif page == "Performance History":
    st.title("Performance History")
    
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
            # Create bar chart for success rate by confidence
            fig_success = go.Figure()
            fig_success.add_trace(go.Bar(
                x=confidence_df['Konfidenz'],
                y=confidence_df['Erfolgsrate (%)'],
                marker_color=['#ff9999', '#ffcc99', '#99ff99'],
                text=confidence_df['Erfolgsrate (%)'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            ))
            fig_success.update_layout(
                title="Erfolgsrate nach Konfidenz",
                xaxis_title="Konfidenz",
                yaxis_title="Erfolgsrate (%)",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            # Create bar chart for pips by confidence
            fig_pips = go.Figure()
            fig_pips.add_trace(go.Bar(
                x=confidence_df['Konfidenz'],
                y=confidence_df['Pips'],
                marker_color=['#ff9999', '#ffcc99', '#99ff99'],
                text=confidence_df['Pips'].apply(lambda x: f"{x:.1f}"),
                textposition='auto'
            ))
            fig_pips.update_layout(
                title="Gesamtgewinn/-verlust nach Konfidenz",
                xaxis_title="Konfidenz",
                yaxis_title="Pips"
            )
            st.plotly_chart(fig_pips, use_container_width=True)
        
        # Performance over time
        st.subheader("Performance im Zeitverlauf")
        
        # Group data by date and calculate cumulative performance
        performance_data['date'] = pd.to_datetime(performance_data['timestamp']).dt.date
        time_performance = performance_data.groupby('date')['profit_loss_pips'].sum().reset_index()
        time_performance['cumulative'] = time_performance['profit_loss_pips'].cumsum()
        
        # Create line chart for cumulative performance
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_performance['date'],
            y=time_performance['cumulative'],
            mode='lines+markers',
            name='Kumulativ',
            line=dict(color='#0A2463', width=2)
        ))
        fig_time.add_trace(go.Bar(
            x=time_performance['date'],
            y=time_performance['profit_loss_pips'],
            name='Täglich',
            marker_color=time_performance['profit_loss_pips'].apply(
                lambda x: 'green' if x > 0 else 'red'
            )
        ))
        fig_time.update_layout(
            title="Performance im Zeitverlauf",
            xaxis_title="Datum",
            yaxis_title="Pips",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
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
    
    st.markdown("""
    ### Haben Sie Fragen oder Anregungen?

    Forex Intelligence ist ein KI-basiertes Tool zur Analyse von Forex-Märkten. Wir freuen uns über Ihr Feedback!

    **E-Mail:** [thomasbrot@proton.me](mailto:thomasbrot@proton.me)

    **Feedback:**
    """)
    
    with st.form("feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("E-Mail")
        message = st.text_area("Nachricht")
        submit_button = st.form_submit_button("Absenden")
        
        if submit_button:
            if name and email and message:
                st.success("Vielen Dank für Ihr Feedback! Wir werden uns bald bei Ihnen melden.")
            else:
                st.error("Bitte füllen Sie alle Felder aus.")
    
    st.markdown("""
    ### Über Forex Intelligence

    Forex Intelligence ist ein KI-Tool, das fortschrittliche Algorithmen nutzt, um Forex-Märkte zu analysieren und Handelssignale zu generieren.

    **Features:**
    - KI-gestützte Marktanalyse in Echtzeit
    - Trading Signale mit Stop-Loss und Take-Profit Werten
    - Risikoanalyse mit Konfidenz-Bewertungen
    - Performance-Tracking und -Analyse

    **Haftungsausschluss:** 
    Forex Intelligence bietet keine professionelle Finanzberatung. Alle Signale basieren auf Algorithmen und historischen Daten. 
    Trading auf den Forex-Märkten beinhaltet Risiken und kann zum Verlust Ihres investierten Kapitals führen.
    """)
