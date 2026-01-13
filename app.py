# app.py
"""
Executive Overview Dashboard
High-level performance metrics and strategic insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import cargar_datos, get_restaurante_color

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Executive Overview",
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
    
    /* Main Header */
    .executive-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .executive-title {
        color: white;
        font-size: 2.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .executive-subtitle {
        color: #e0e7ff;
        font-size: 1.15rem;
        margin: 0.75rem 0 0 0;
        font-weight: 400;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.75rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    
    .kpi-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .kpi-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1;
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
    
    /* Restaurant Quick View Card */
    .restaurant-quick-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s;
    }
    
    .restaurant-quick-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transform: translateX(4px);
    }
    
    .restaurant-name {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .restaurant-revenue {
        font-size: 1.25rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    /* Info Box */
    .info-box {
        background: white;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #1e293b;
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 0.75rem 0;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e2e8f0;
        margin: 2rem 0;
    }
    
    /* Welcome Message */
    .welcome-message {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 0.375rem;
        margin: 1.5rem 0;
    }
    
    .welcome-message p {
        margin: 0;
        color: #1e40af;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

df = cargar_datos()

if df is None:
    st.error("⚠️ Unable to load data. Please verify the /data/ directory.")
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;
                text-align: center;'>
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>Executive Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # General Information
    st.markdown("<h4 style='color: #1e293b; font-weight: 700; margin-bottom: 1rem;'>General Information</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Locations", df['restaurante'].nunique())
    
    with col2:
        st.metric("Products", df['producto'].nunique())
    
    st.metric(
        "Analysis Period", 
        f"{df['fecha'].min().date()} to {df['fecha'].max().date()}"
    )
    
    st.metric("Operating Days", df['fecha'].nunique())
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Quick View by Location
    st.markdown("<h4 style='color: #1e293b; font-weight: 700; margin-bottom: 1rem;'>Revenue by Location</h4>", unsafe_allow_html=True)
    
    for restaurante in sorted(df['restaurante'].unique()):
        df_rest = df[df['restaurante'] == restaurante]
        ventas = df_rest['venta_pesos'].sum()
        color = get_restaurante_color(restaurante)
        
        st.markdown(f"""
        <div class='restaurant-quick-card' style='border-left-color: {color};'>
            <div class='restaurant-name'>{restaurante}</div>
            <div class='restaurant-revenue' style='color: {color};'>${ventas/1e6:.2f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Navigation hint
    st.markdown("""
    <div style='background: #f8fafc; padding: 1rem; border-radius: 0.5rem; 
                border: 1px solid #e2e8f0; margin-top: 1rem;'>
        <p style='margin: 0; color: #64748b; font-size: 0.875rem; text-align: center;'>
            Use the sidebar navigation to access detailed analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN CONTENT
# ==========================================

# Header
st.markdown("""
<div class='executive-header'>
    <h1 class='executive-title'>Executive Overview Dashboard</h1>
    <p class='executive-subtitle'>Strategic performance analytics for premium restaurant operations in Barranquilla</p>
</div>
""", unsafe_allow_html=True)

# Welcome message
st.markdown("""
<div class='welcome-message'>
    <p><strong>Welcome to the Executive Dashboard.</strong> This interface provides high-level insights into operational performance, 
    revenue trends, and strategic metrics across all locations. Navigate through the sidebar to access detailed analysis modules 
    including monthly trends, daily performance, product analytics, risk assessment, and ML-powered demand forecasting.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# KEY PERFORMANCE INDICATORS
# ==========================================

st.markdown("<h2 class='section-header'>Key Performance Indicators</h2>", unsafe_allow_html=True)

# Calculate KPIs
ventas_totales = df['venta_pesos'].sum()
unidades_totales = df['cantidad_vendida_diaria'].sum()
ticket_promedio = df.groupby('fecha')['venta_pesos'].sum().mean()
dias_operacion = df['fecha'].nunique()

# Display KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Revenue",
        value=f"${ventas_totales/1e6:.2f}M",
        help="Cumulative revenue across all locations"
    )

with col2:
    st.metric(
        label="Units Sold",
        value=f"{unidades_totales:,.0f}",
        help="Total units sold across all products"
    )

with col3:
    st.metric(
        label="Avg Daily Revenue",
        value=f"${ticket_promedio/1e3:.1f}K",
        help="Average daily revenue per operating day"
    )

with col4:
    st.metric(
        label="Operating Days",
        value=dias_operacion,
        help="Total days with recorded transactions"
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# LOCATION PERFORMANCE COMPARISON
# ==========================================

st.markdown("<h2 class='section-header'>Location Performance Comparison</h2>", unsafe_allow_html=True)

# Calculate revenue by location
ventas_rest = df.groupby('restaurante')['venta_pesos'].sum().sort_values(ascending=False)

# Performance comparison chart
fig = go.Figure()

colors = [get_restaurante_color(r) for r in ventas_rest.index]

fig.add_trace(go.Bar(
    x=ventas_rest.index,
    y=ventas_rest.values,
    marker=dict(
        color=colors,
        line=dict(color='#64748b', width=1)
    ),
    text=[f'${v/1e6:.2f}M' for v in ventas_rest.values],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig.update_layout(
    title=dict(
        text='Total Revenue by Location',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(
        title='Location',
        showgrid=False
    ),
    yaxis=dict(
        title='Revenue (COP)',
        showgrid=True,
        gridcolor='#f1f5f9'
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=450,
    showlegend=False,
    hovermode='x'
)

st.plotly_chart(fig, use_container_width=True)

# Performance table
col1, col2 = st.columns([2, 1])

with col1:
    # Detailed metrics by location
    st.markdown("### Performance Metrics by Location")
    
    metrics_by_location = []
    
    for restaurante in ventas_rest.index:
        df_rest = df[df['restaurante'] == restaurante]
        
        metrics_by_location.append({
            'Location': restaurante,
            'Revenue': f"${df_rest['venta_pesos'].sum()/1e6:.2f}M",
            'Units Sold': f"{df_rest['cantidad_vendida_diaria'].sum():,.0f}",
            'Products': df_rest['producto'].nunique(),
            'Avg Daily': f"${df_rest.groupby('fecha')['venta_pesos'].sum().mean()/1e3:.1f}K"
        })
    
    df_metrics = pd.DataFrame(metrics_by_location)
    
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

with col2:
    # Revenue distribution pie chart
    st.markdown("### Revenue Distribution")
    
    fig_pie = go.Figure()
    
    fig_pie.add_trace(go.Pie(
        labels=ventas_rest.index,
        values=ventas_rest.values,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textposition='inside',
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
    ))
    
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# HISTORICAL TREND
# ==========================================

st.markdown("<h2 class='section-header'>Revenue Trend Analysis</h2>", unsafe_allow_html=True)

# Monthly aggregation
df['año_mes'] = df['fecha'].dt.to_period('M')
ventas_mensuales = df.groupby('año_mes')['venta_pesos'].sum().reset_index()
ventas_mensuales['año_mes'] = ventas_mensuales['año_mes'].dt.to_timestamp()

# Trend chart
fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=ventas_mensuales['año_mes'],
    y=ventas_mensuales['venta_pesos'],
    mode='lines+markers',
    name='Monthly Revenue',
    line=dict(color='#3b82f6', width=3),
    marker=dict(size=8, color='#3b82f6', line=dict(width=2, color='white')),
    fill='tozeroy',
    fillcolor='rgba(59, 130, 246, 0.1)',
    hovertemplate='<b>%{x|%B %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Average line
promedio_mensual = ventas_mensuales['venta_pesos'].mean()
fig_trend.add_hline(
    y=promedio_mensual,
    line_dash="dash",
    line_color="#ef4444",
    line_width=2,
    annotation_text=f"Average: ${promedio_mensual/1e6:.2f}M",
    annotation_position="right"
)

fig_trend.update_layout(
    title=dict(
        text='Monthly Revenue Trend',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(
        title='Month',
        showgrid=False
    ),
    yaxis=dict(
        title='Revenue (COP)',
        showgrid=True,
        gridcolor='#f1f5f9'
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=400,
    hovermode='x unified',
    showlegend=False
)

st.plotly_chart(fig_trend, use_container_width=True)

# Trend insights
col1, col2, col3 = st.columns(3)

with col1:
    best_month = ventas_mensuales.loc[ventas_mensuales['venta_pesos'].idxmax()]
    st.metric(
        "Best Month",
        f"{best_month['año_mes'].strftime('%B %Y')}",
        f"${best_month['venta_pesos']/1e6:.2f}M"
    )

with col2:
    growth = ((ventas_mensuales['venta_pesos'].iloc[-1] - ventas_mensuales['venta_pesos'].iloc[0]) / 
              ventas_mensuales['venta_pesos'].iloc[0] * 100)
    st.metric(
        "Period Growth",
        f"{growth:+.1f}%",
        help="Revenue growth from first to last month"
    )

with col3:
    avg_monthly = ventas_mensuales['venta_pesos'].mean()
    st.metric(
        "Avg Monthly Revenue",
        f"${avg_monthly/1e6:.2f}M"
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# TOP PRODUCTS
# ==========================================

st.markdown("<h2 class='section-header'>Top 10 Products by Revenue</h2>", unsafe_allow_html=True)

top_productos = df.groupby('producto')['venta_pesos'].sum().nlargest(10).sort_values(ascending=True)

fig_top = go.Figure()

fig_top.add_trace(go.Bar(
    y=top_productos.index,
    x=top_productos.values,
    orientation='h',
    marker=dict(
        color=top_productos.values,
        colorscale=[[0, '#93c5fd'], [1, '#1e3a8a']],
        showscale=False,
        line=dict(color='#1e40af', width=1)
    ),
    text=[f'${v/1e6:.2f}M' for v in top_productos.values],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>'
))

fig_top.update_layout(
    title=dict(
        text='Revenue by Product',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(
        title='Revenue (COP)',
        showgrid=True,
        gridcolor='#f1f5f9'
    ),
    yaxis=dict(
        title='',
        showgrid=False
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    showlegend=False,
    margin=dict(l=250)
)

st.plotly_chart(fig_top, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# QUICK INSIGHTS
# ==========================================

st.markdown("<h2 class='section-header'>Quick Insights</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    # Busiest day of week
    dia_semana_ventas = df.groupby('dia_semana')['venta_pesos'].sum()
    mejor_dia = dia_semana_ventas.idxmax()
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>Busiest Day of Week</h4>
        <p style='font-size: 1.5rem; font-weight: 700; color: #3b82f6; margin: 0.5rem 0;'>{mejor_dia}</p>
        <p style='margin: 0; color: #64748b; font-size: 0.875rem;'>
            ${dia_semana_ventas[mejor_dia]/1e6:.2f}M total revenue
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Average units per transaction
    avg_units = df['cantidad_vendida_diaria'].mean()
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>Avg Units per Day</h4>
        <p style='font-size: 1.5rem; font-weight: 700; color: #3b82f6; margin: 0.5rem 0;'>{avg_units:.1f}</p>
        <p style='margin: 0; color: #64748b; font-size: 0.875rem;'>
            Across all products and locations
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Revenue per unit
    revenue_per_unit = ventas_totales / unidades_totales
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>Revenue per Unit</h4>
        <p style='font-size: 1.5rem; font-weight: 700; color: #3b82f6; margin: 0.5rem 0;'>${revenue_per_unit:,.0f}</p>
        <p style='margin: 0; color: #64748b; font-size: 0.875rem;'>
            Average across all products
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.875rem;'>", unsafe_allow_html=True)
st.markdown(f"Executive Overview • {df['restaurante'].nunique()} locations • {df['producto'].nunique()} products • {dias_operacion} operating days")
st.markdown("</div>", unsafe_allow_html=True)
