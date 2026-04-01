import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="ProQuant | AI Terminal", layout="wide")

# --- 2. THE HIGH-CONTRAST CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0B0E11;
        color: #FFFFFF;
    }

    /* Premium Glass Cards */
    .premium-card {
        background: #151921; 
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #2D3446;
        box-shadow: 0 8px 16px rgba(0,0,0,0.5);
        margin-bottom: 15px;
    }
    
    /* --- SIDEBAR CUSTOM COLOR --- */
    [data-testid="stSidebar"] {
        background-color: #bdb76b !important; /* Darker background for sidebar */
        border-right: 1px solid #2D3446; /* Sidebar aur main screen ke beech ki line */
    }

    /* Sidebar ke titles aur labels ka color white karne ke liye */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h2 {
        color: black !important;
    }



    /* FIXED: Label color changed from Grey to Bright White/Silver */
    .metric-label { 
        color: #FFFFFF !important; /* Pure White Labels */
        font-size: 14px; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        margin-bottom: 8px;
        opacity: 0.9;
    }

    .metric-value { 
        color: #00FFC2; 
        font-size: 32px; 
        font-weight: 800; 
        text-shadow: 0 0 15px rgba(0, 255, 194, 0.3);
    }

    /* FIXED: Sub-text visibility */
    .sub-text {
        color: #E2E8F0 !important; /* Bright Silver for secondary info */
        font-size: 13px;
        margin-top: 5px;
        font-weight: 500;
    }

    .header-banner {
        background: #1C212D;
        padding: 20px 25px;
        border-radius: 12px;
        border-left: 6px solid #00FFC2;
        margin-bottom: 30px;
        color: #FFFFFF; /* Banner title White */
    }

    [data-testid="stMetric"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & MODEL ENGINES ---
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss)))

@st.cache_data(ttl=600)
def get_clean_data(ticker, period):
    df = yf.download(ticker, period=period)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df['RSI'] = compute_rsi(df['Close'])
    return df


# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#BF2A2A;'>PRO-QUANT AI</h2>", unsafe_allow_html=True)
    
    # Yahan aap apne pasand ke tickers add kar sakte hain
    ticker_options = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "RELIANCE.NS", "TCS.NS"]
    
    # Text input ki jagah Selectbox ka use
    symbol = st.selectbox("Select Asset Ticker", options=ticker_options, index=0)
    
    timeframe = st.selectbox("Historical Range", ["1y", "2y", "5y"], index=0)
    forecast_days = st.slider("Prediction Horizon (Days)", 30, 365, 90)
    flexibility = st.select_slider("AI Sensitivity", options=[0.01, 0.05, 0.1, 0.5], value=0.05)
    
    st.markdown("---")
    st.caption("Terminal Status: Active")

# --- 5. DASHBOARD EXECUTION ---
try:
    data = get_clean_data(symbol, timeframe)
    
    if data is not None:
        # AI Modeling
        df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet(changepoint_prior_scale=flexibility, daily_seasonality=False, yearly_seasonality=True)
        model.fit(df_train)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Calculations
        last_p = float(data['Close'].iloc[-1])
        prev_p = float(data['Close'].iloc[-2])
        diff = last_p - prev_p
        rsi_val = float(data['RSI'].iloc[-1])
        target_p = float(forecast['yhat'].iloc[-1])
        upside = ((target_p / last_p) - 1) * 100
        vol = float(data['Close'].pct_change().std() * np.sqrt(252) * 100)

        # HEADER BANNER
        st.markdown(f"""<div class="header-banner">
            <span style="font-size: 28px; font-weight: 800; color: #FFFFFF;">{symbol} Institutional Analysis</span>
            <span style="background:#00FFC2; color:#000000; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:bold; margin-left:20px;">LIVE AI DATA</span>
        </div>""", unsafe_allow_html=True)

        # METRIC CARDS (With White Text Fix)
        m1, m2, m3, m4 = st.columns(4)

        def metric_box(col, label, value, sub="", color="#00FFC2"):
            col.markdown(f"""<div class="premium-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="sub-text">{sub}</div>
            </div>""", unsafe_allow_html=True)

        metric_box(m1, "Live Market Price", f"${last_p:,.2f}", f"{diff:+.2f} ({(diff/prev_p)*100:+.2f}%)")
        metric_box(m2, "Momentum (RSI)", f"{rsi_val:.1f}", "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL", color="#FFB800")
        metric_box(m3, "Annual Volatility", f"{vol:.1f}%", "Historical Variance")
        metric_box(m4, "AI Target Price", f"${target_p:,.2f}", f"{upside:+.1f}% Forecasted Change")

        # CHARTS
        # --- 6. CHARTS (UPDATED WITH BLUE/RED/LIGHT-BLUE PALETTE) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                            subplot_titles=("", ""), # Titles removed for cleaner look
                            row_heights=[0.75, 0.25])

        # 1. AI Confidence Range (Light Blue Shaded Area)
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], 
            line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], 
            fill='tonexty', fillcolor='rgba(135, 206, 235, 0.08)', # Ultra-light Sky Blue
            line_color='rgba(0,0,0,0)', name='AI Range'), row=1, col=1)

        # 2. Actual Price Line (Main Visibility: Deep Blue)
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'], 
            name="Market Price", 
            line=dict(color="#1A73E8", width=1.8)), row=1, col=1)

        # 3. AI Prediction (Highlight: Deep Red Dashed)
        fig.add_trace(go.Scatter(
            x=forecast['ds'].iloc[-forecast_days:], 
            y=forecast['yhat'].iloc[-forecast_days:], 
            name="AI Forecast", 
            line=dict(color="#EA4335", width=2.5, dash='dot')), row=1, col=1)

        # 4. RSI Plot (Clean Grey/Blue Color)
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['RSI'], 
            name="RSI", 
            line=dict(color="#8E94A5", width=1.5)), row=2, col=1)

        # RSI Threshold Lines (Subtle red/green dashed)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 75, 75, 0.4)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 255, 162, 0.4)", row=2, col=1)

        # --- HIGH VISIBILITY LAYOUT FIX ---
        fig.update_layout(
            height=700,
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)', # Glass effect
            plot_bgcolor='rgba(0,0,0,0)',
            # Legend moved below for clarity
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="right", x=1, font=dict(color="white")),
            margin=dict(t=30, b=50, l=0, r=0)
        )

        # Subplot Titles Stylization (Custom placement)
        fig.add_annotation(text="PRICE & AI FORECAST", x=0, y=1.02, xref="paper", yref="paper", 
                           showarrow=False, font=dict(color="white", size=14, weight=700))
        fig.add_annotation(text="MOMENTUM (RSI)", x=0, y=0.28, xref="paper", yref="paper", 
                           showarrow=False, font=dict(color="white", size=14, weight=700))

        # Clean Grid Lines (Ultra-Subtle Grey for clarity)
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='#1E232E', 
            zeroline=False, tickfont=dict(color="#A5ACB9")
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='#1E232E', 
            zeroline=False, tickfont=dict(color="#A5ACB9"),
            tickprefix="$"
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        
        

        # ⚡ REPORT SECTION
        st.markdown("### ⚡ AI Intelligence Report")
        r_col1, r_col2 = st.columns([1, 1])
        
        with r_col1:
            st.markdown(f"""<div class="premium-card" style="min-height: 180px;">
                <h4 style="margin:0; color:#FFFFFF; font-size:16px; font-weight:700;">STRATEGIC SUMMARY</h4>
                <p style="font-size:18px; margin-top:15px; line-height:1.6; color:#FFFFFF;">
                    Asset is showing a <b style="color:#00FFC2;">{'Strong' if abs(upside) > 10 else 'Moderate'} 
                    {'Uptrend' if upside > 0 else 'Downtrend'}</b> based on historical seasonality. 
                    Structure indicates a <b style="color:#FFB800;">{flexibility*100:.1f}% sensitivity</b> structural drift.
                </p>
                <div style="margin-top:10px; background:#232730; display:inline-block; padding:5px 12px; border-radius:4px; font-size:12px; color:#00FFC2; font-weight:bold;">SYSTEM READY</div>
            </div>""", unsafe_allow_html=True)
            
        with r_col2:
            # Table with white text headers
            st.dataframe(forecast[['ds', 'yhat', 'trend']].tail(5).style.format(precision=2), use_container_width=True)

except Exception as e:
    st.error(f"Please enter a valid Ticker. (Error: {e})")