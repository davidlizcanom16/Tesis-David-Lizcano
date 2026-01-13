# pages/05_📊_Risk_Analysis.py
"""
Risk Analysis Dashboard
Early identification of underperforming products with negative trends and low rotation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import cargar_datos, get_restaurante_color

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Risk Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PROFESSIONAL STYLING
# ==========================================

st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8fafc;
    }
    
    /* Professional Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .dashboard-subtitle {
        color: #e0e7ff;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Filter Section */
    .filter-section {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk Level Cards */
    .risk-summary-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #ef4444;
    }
    
    .risk-summary-card.high {
        border-top-color: #ef4444;
    }
    
    .risk-summary-card.medium {
        border-top-color: #f59e0b;
    }
    
    .risk-summary-card.low {
        border-top-color: #10b981;
    }
    
    .risk-summary-card.info {
        border-top-color: #3b82f6;
    }
    
    .risk-summary-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .risk-summary-value {
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .risk-summary-subtitle {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    /* Alert Box */
    .alert-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .alert-box strong {
        color: #92400e;
    }
    
    .alert-box-critical {
        background: #fee2e2;
        border-left-color: #ef4444;
    }
    
    .alert-box-critical strong {
        color: #991b1b;
    }
    
    /* Risk Score Card */
    .risk-score-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ef4444;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .risk-score-card.high {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .risk-score-card.medium {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    .risk-score-card.low {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    .risk-score-label {
        color: #64748b;
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
    }
    
    .risk-score-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .risk-score-value.high {
        color: #ef4444;
    }
    
    .risk-score-value.medium {
        color: #f59e0b;
    }
    
    .risk-score-value.low {
        color: #10b981;
    }
    
    .risk-score-subtitle {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    /* Recommendations Box */
    .recommendations-box {
        background: white;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .recommendations-box h4 {
        color: #1e293b;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
    }
    
    .recommendations-box ul {
        margin: 0;
        padding-left: 1.5rem;
        color: #475569;
    }
    
    .recommendations-box li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Warning Indicator */
    .warning-indicator {
        background: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 0.375rem;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.875rem;
        color: #92400e;
        font-weight: 500;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

df = cargar_datos()

if df is None:
    st.error("⚠️ Unable to load data. Please check data sources.")
    st.stop()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='dashboard-header'>
    <h1 class='dashboard-title'>Risk Analysis Dashboard</h1>
    <p class='dashboard-subtitle'>Early identification of underperforming products with declining trends, low rotation, and high variability</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FILTERS
# ==========================================

st.markdown("<div class='filter-section'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    años = ['All Years'] + sorted(df['año'].unique().tolist(), reverse=True)
    año_sel = st.selectbox("Fiscal Year", años, index=0, help="Filter by specific year")

with col2:
    restaurantes = ['All Locations'] + sorted(df['restaurante'].unique().tolist())
    restaurante_sel = st.selectbox("Location", restaurantes, index=0, help="Filter by restaurant")

with col3:
    umbral_dias = st.selectbox("Recent Period", [30, 60, 90], index=1, help="Days for recent trend analysis")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FILTER DATA
# ==========================================

df_filtrado = df.copy()

if año_sel != 'All Years':
    df_filtrado = df_filtrado[df_filtrado['año'] == int(año_sel)]

if restaurante_sel != 'All Locations':
    df_filtrado = df_filtrado[df_filtrado['restaurante'] == restaurante_sel]

if len(df_filtrado) == 0:
    st.warning("No data available for selected filters")
    st.stop()

# Get recent N days
fecha_max = df_filtrado['fecha'].max()
fecha_min_reciente = fecha_max - pd.Timedelta(days=umbral_dias)
df_reciente = df_filtrado[df_filtrado['fecha'] >= fecha_min_reciente]

# ==========================================
# CALCULATE RISK INDICATORS
# ==========================================

productos_riesgo = []

for producto in df_filtrado['producto'].unique():
    df_prod = df_filtrado[df_filtrado['producto'] == producto]
    df_prod_reciente = df_reciente[df_reciente['producto'] == producto]
    
    # Basic metrics
    ventas_total = df_prod['venta_pesos'].sum()
    ventas_recientes = df_prod_reciente['venta_pesos'].sum()
    dias_total = len(df_prod)
    dias_recientes = len(df_prod_reciente)
    
    if dias_total < 10:  # Insufficient history
        continue
    
    # 1. TREND (linear regression)
    ventas_diarias = df_prod.groupby('fecha')['venta_pesos'].sum().reset_index()
    if len(ventas_diarias) >= 5:
        x = np.arange(len(ventas_diarias))
        y = ventas_diarias['venta_pesos'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        tendencia = slope
        r_squared = r_value ** 2
    else:
        tendencia = 0
        r_squared = 0
    
    # 2. RECENT VS HISTORICAL SALES
    venta_prom_historico = ventas_total / dias_total
    venta_prom_reciente = ventas_recientes / max(dias_recientes, 1)
    caida_reciente = ((venta_prom_historico - venta_prom_reciente) / venta_prom_historico) * 100 if venta_prom_historico > 0 else 0
    
    # 3. VARIABILITY
    std_ventas = df_prod.groupby('fecha')['venta_pesos'].sum().std()
    mean_ventas = df_prod.groupby('fecha')['venta_pesos'].sum().mean()
    cv = (std_ventas / mean_ventas) * 100 if mean_ventas > 0 else 0
    
    # 4. SALES FREQUENCY
    dias_unicos_venta = df_prod['fecha'].nunique()
    dias_disponibles = df_filtrado['fecha'].nunique()
    frecuencia = (dias_unicos_venta / dias_disponibles) * 100
    
    # 5. DAYS WITHOUT SALES (recent period)
    dias_sin_venta = umbral_dias - dias_recientes
    
    # CALCULATE RISK SCORE (0-100, higher = more risk)
    score_riesgo = 0
    
    # Negative trend (0-30 points)
    if tendencia < 0:
        score_riesgo += min(abs(tendencia) / 10000, 30)
    
    # Recent decline (0-25 points)
    if caida_reciente > 0:
        score_riesgo += min(caida_reciente / 4, 25)
    
    # High variability (0-20 points)
    if cv > 50:
        score_riesgo += min((cv - 50) / 5, 20)
    
    # Low frequency (0-15 points)
    if frecuencia < 50:
        score_riesgo += min((50 - frecuencia) / 3.33, 15)
    
    # Recent days without sales (0-10 points)
    if dias_sin_venta > 0:
        score_riesgo += min(dias_sin_venta / 3, 10)
    
    productos_riesgo.append({
        'producto': producto,
        'ventas_total': ventas_total,
        'ventas_recientes': ventas_recientes,
        'venta_prom_historico': venta_prom_historico,
        'venta_prom_reciente': venta_prom_reciente,
        'caida_reciente_%': caida_reciente,
        'tendencia': tendencia,
        'r_squared': r_squared,
        'cv_%': cv,
        'frecuencia_%': frecuencia,
        'dias_sin_venta': dias_sin_venta,
        'score_riesgo': score_riesgo
    })

df_riesgo = pd.DataFrame(productos_riesgo).sort_values('score_riesgo', ascending=False)

# Classify risk level
def clasificar_riesgo(score):
    if score >= 60:
        return 'High'
    elif score >= 30:
        return 'Medium'
    else:
        return 'Low'

df_riesgo['nivel_riesgo'] = df_riesgo['score_riesgo'].apply(clasificar_riesgo)

# ==========================================
# RISK SUMMARY
# ==========================================

st.markdown("<h2 class='section-header'>Risk Assessment Summary</h2>", unsafe_allow_html=True)

riesgo_alto = len(df_riesgo[df_riesgo['nivel_riesgo'] == 'High'])
riesgo_medio = len(df_riesgo[df_riesgo['nivel_riesgo'] == 'Medium'])
riesgo_bajo = len(df_riesgo[df_riesgo['nivel_riesgo'] == 'Low'])
ventas_en_riesgo = df_riesgo[df_riesgo['nivel_riesgo'] == 'High']['ventas_total'].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='risk-summary-card high'>
        <div class='risk-summary-label'>High Risk</div>
        <div class='risk-summary-value'>{riesgo_alto}</div>
        <div class='risk-summary-subtitle'>Immediate attention required</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='risk-summary-card medium'>
        <div class='risk-summary-label'>Medium Risk</div>
        <div class='risk-summary-value'>{riesgo_medio}</div>
        <div class='risk-summary-subtitle'>Close monitoring needed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='risk-summary-card low'>
        <div class='risk-summary-label'>Low Risk</div>
        <div class='risk-summary-value'>{riesgo_bajo}</div>
        <div class='risk-summary-subtitle'>Stable performance</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='risk-summary-card info'>
        <div class='risk-summary-label'>Revenue at Risk</div>
        <div class='risk-summary-value'>${ventas_en_riesgo/1e6:.1f}M</div>
        <div class='risk-summary-subtitle'>High-risk products</div>
    </div>
    """, unsafe_allow_html=True)

# Critical alert
if riesgo_alto > 0:
    st.markdown(f"""
    <div class='alert-box alert-box-critical'>
        <strong>⚠ CRITICAL ALERT:</strong> {riesgo_alto} product(s) identified with high risk level. 
        Immediate review of pricing strategy, availability, or menu removal is recommended.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TOP RISK PRODUCTS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Top 10 Highest Risk Products</h2>", unsafe_allow_html=True)

top_riesgo = df_riesgo.head(10)

# Chart
fig = go.Figure()

colors = ['#ef4444' if r == 'High' else '#f59e0b' if r == 'Medium' else '#10b981' 
          for r in top_riesgo['nivel_riesgo']]

fig.add_trace(go.Bar(
    y=top_riesgo['producto'],
    x=top_riesgo['score_riesgo'],
    orientation='h',
    marker=dict(color=colors, line=dict(color='#64748b', width=1)),
    text=[f'{s:.0f}' for s in top_riesgo['score_riesgo']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Risk Score: %{x:.0f}<extra></extra>'
))

fig.update_layout(
    title=dict(
        text='Risk Score by Product (0-100 scale)',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(title='Risk Score', showgrid=True, gridcolor='#f1f5f9'),
    yaxis=dict(title='', showgrid=False, categoryorder='total ascending'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    showlegend=False,
    margin=dict(l=250)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# DETAILED ANALYSIS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Detailed Product Analysis</h2>", unsafe_allow_html=True)

# Show top 5 highest risk
for idx, (_, producto) in enumerate(top_riesgo.head(5).iterrows(), 1):
    
    nivel = producto['nivel_riesgo']
    color_nivel = '#ef4444' if nivel == 'High' else '#f59e0b' if nivel == 'Medium' else '#10b981'
    risk_class = 'high' if nivel == 'High' else 'medium' if nivel == 'Medium' else 'low'
    
    with st.expander(f"#{idx} - {producto['producto']} (Risk Score: {producto['score_riesgo']:.0f})", expanded=(idx == 1)):
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Temporal evolution
            df_prod = df_filtrado[df_filtrado['producto'] == producto['producto']]
            ventas_tiempo = df_prod.groupby('fecha')['venta_pesos'].sum().reset_index()
            
            fig = go.Figure()
            
            # Sales line
            fig.add_trace(go.Scatter(
                x=ventas_tiempo['fecha'],
                y=ventas_tiempo['venta_pesos'],
                mode='lines',
                name='Daily Revenue',
                line=dict(color=color_nivel, width=2),
                hovertemplate='<b>%{x|%b %d}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            
            # Trend line
            if len(ventas_tiempo) >= 5:
                x_num = np.arange(len(ventas_tiempo))
                y = ventas_tiempo['venta_pesos'].values
                z = np.polyfit(x_num, y, 1)
                p = np.poly1d(z)
                
                fig.add_trace(go.Scatter(
                    x=ventas_tiempo['fecha'],
                    y=p(x_num),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', color='#64748b', width=2)
                ))
            
            fig.update_layout(
                title=dict(text='Revenue Evolution', font=dict(size=14, color='#1e293b')),
                xaxis=dict(title='Date', showgrid=False),
                yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=300,
                hovermode='x unified',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Indicators**")
            
            # Risk score
            st.markdown(f"""
            <div class='risk-score-card {risk_class}'>
                <div class='risk-score-label'>Risk Score</div>
                <div class='risk-score-value {risk_class}'>{producto['score_riesgo']:.0f}/100</div>
                <div class='risk-score-subtitle'>Level: {nivel}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Warnings
            warnings = []
            
            if producto['caida_reciente_%'] > 0:
                warnings.append(f"Recent decline: -{producto['caida_reciente_%']:.1f}%")
            
            if producto['cv_%'] > 50:
                warnings.append(f"High variability: CV = {producto['cv_%']:.0f}%")
            
            if producto['frecuencia_%'] < 50:
                warnings.append(f"Low frequency: {producto['frecuencia_%']:.0f}%")
            
            if producto['dias_sin_venta'] > 0:
                warnings.append(f"{producto['dias_sin_venta']} days without sales")
            
            for warning in warnings:
                st.markdown(f"<div class='warning-indicator'>⚠ {warning}</div>", unsafe_allow_html=True)
        
        # Comparative metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Historical Avg",
                f"${producto['venta_prom_historico']/1e3:.1f}K",
                help="Average daily revenue (all time)"
            )
        
        with col2:
            cambio = producto['venta_prom_reciente'] - producto['venta_prom_historico']
            st.metric(
                f"Recent Avg ({umbral_dias}d)",
                f"${producto['venta_prom_reciente']/1e3:.1f}K",
                delta=f"${cambio/1e3:+.1f}K",
                delta_color="normal" if cambio >= 0 else "inverse"
            )
        
        with col3:
            st.metric(
                "Total Revenue",
                f"${producto['ventas_total']/1e6:.2f}M",
                help="Cumulative revenue"
            )
        
        # Specific recommendations
        st.markdown("""
        <div class='recommendations-box'>
            <h4>Recommended Actions</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        if producto['score_riesgo'] >= 60:
            st.markdown("<li><strong>Critical:</strong> Evaluate profitability and consider discontinuation</li>", unsafe_allow_html=True)
        
        if producto['caida_reciente_%'] > 30:
            st.markdown("<li>Implement temporary promotion or discount strategy</li>", unsafe_allow_html=True)
        
        if producto['cv_%'] > 70:
            st.markdown("<li>Review availability and preparation consistency</li>", unsafe_allow_html=True)
        
        if producto['frecuencia_%'] < 30:
            st.markdown("<li>Consider menu repositioning or combo inclusion</li>", unsafe_allow_html=True)
        
        if producto['dias_sin_venta'] > 15:
            st.markdown("<li>Verify ingredient availability and supply chain</li>", unsafe_allow_html=True)
        
        if producto['score_riesgo'] < 30:
            st.markdown("<li>Monitor evolution over next 30 days</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)

# ==========================================
# RISK MATRIX
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Risk Matrix: Recent Decline vs Variability</h2>", unsafe_allow_html=True)

fig = px.scatter(
    df_riesgo.head(30),
    x='cv_%',
    y='caida_reciente_%',
    size='ventas_total',
    color='nivel_riesgo',
    hover_name='producto',
    labels={
        'cv_%': 'Coefficient of Variation (%)',
        'caida_reciente_%': 'Recent Decline (%)',
        'nivel_riesgo': 'Risk Level'
    },
    title='Products by Recent Decline vs Variability',
    color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
)

# Reference lines
fig.add_hline(y=20, line_dash="dash", line_color="#94a3b8", annotation_text="Decline threshold: 20%", annotation_position="right")
fig.add_vline(x=50, line_dash="dash", line_color="#94a3b8", annotation_text="CV threshold: 50%", annotation_position="top")

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SUMMARY TABLE
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Risk Summary Table</h2>", unsafe_allow_html=True)

tabla_riesgo = df_riesgo[['producto', 'score_riesgo', 'nivel_riesgo', 'caida_reciente_%', 
                           'cv_%', 'frecuencia_%', 'ventas_total']].head(20).copy()

# Rename columns
tabla_riesgo.columns = ['Product', 'Risk Score', 'Risk Level', 'Recent Decline (%)', 
                        'CV (%)', 'Frequency (%)', 'Total Revenue']

st.dataframe(
    tabla_riesgo.style.format({
        'Risk Score': '{:.0f}',
        'Recent Decline (%)': '{:.1f}%',
        'CV (%)': '{:.0f}%',
        'Frequency (%)': '{:.0f}%',
        'Total Revenue': '${:,.0f}'
    }).background_gradient(subset=['Risk Score'], cmap='Reds'),
    use_container_width=True,
    height=500
)

# Footer
st.markdown("<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.875rem;'>", unsafe_allow_html=True)
st.markdown(f"Risk analysis based on {len(df_riesgo)} products • Score calculation: Trend (30%), Recent decline (25%), Variability (20%), Frequency (15%), Days without sales (10%)")
st.markdown("</div>", unsafe_allow_html=True)
