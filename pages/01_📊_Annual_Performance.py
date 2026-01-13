# pages/01_📊_Annual_Performance.py
"""
Annual Performance Dashboard
Comprehensive yearly analytics for restaurant operations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import cargar_datos, get_restaurante_color, formatear_numero
from utils.metrics import calcular_metricas_anuales

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Annual Performance",
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
    
    /* Professional Metrics */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .metric-delta {
        color: #10b981;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
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
    
    /* Restaurant Badge */
    .restaurant-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 600;
        font-size: 0.875rem;
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    /* Insight Box */
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .insight-box p {
        margin: 0;
        color: #0c4a6e;
        font-weight: 500;
    }
    
    /* Ranking Card */
    .ranking-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    
    .ranking-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .ranking-position {
        font-size: 1.5rem;
        font-weight: 700;
        color: #64748b;
    }
    
    .ranking-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .ranking-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .ranking-percentage {
        font-size: 0.875rem;
        color: #64748b;
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

# Add helper column for month names
meses_nombres = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}
df['mes_nombre'] = df['mes'].map(meses_nombres)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='dashboard-header'>
    <h1 class='dashboard-title'>Annual Performance Dashboard</h1>
    <p class='dashboard-subtitle'>Comprehensive yearly analytics and key performance indicators</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FILTERS
# ==========================================

st.markdown("<div class='filter-section'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    años_disponibles = sorted(df['año'].unique(), reverse=True)
    año_seleccionado = st.selectbox(
        "Fiscal Year",
        años_disponibles,
        index=0,
        help="Select the fiscal year for analysis"
    )

with col2:
    restaurantes = ['All Locations'] + sorted(df['restaurante'].unique().tolist())
    restaurante_sel = st.selectbox(
        "Location",
        restaurantes,
        index=0,
        help="Filter by specific restaurant location"
    )

with col3:
    if restaurante_sel != 'All Locations':
        color = get_restaurante_color(restaurante_sel)
        st.markdown(f"""
        <div style='margin-top: 1.8rem;'>
            <span class='restaurant-badge'>{restaurante_sel}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FILTER DATA
# ==========================================

if restaurante_sel == 'All Locations':
    df_filtrado = df[df['año'] == año_seleccionado]
else:
    df_filtrado = df[(df['año'] == año_seleccionado) & (df['restaurante'] == restaurante_sel)]

if len(df_filtrado) == 0:
    st.warning(f"No data available for {restaurante_sel} in {año_seleccionado}")
    st.stop()

# ==========================================
# KEY PERFORMANCE INDICATORS
# ==========================================

st.markdown("<h2 class='section-header'>Key Performance Indicators</h2>", unsafe_allow_html=True)

# Calculate metrics
ventas_totales = df_filtrado['venta_pesos'].sum()
unidades_totales = df_filtrado['cantidad_vendida_diaria'].sum()
dias_operacion = df_filtrado['fecha'].nunique()
productos_activos = df_filtrado['producto'].nunique()

# Calculate YoY change if previous year exists
año_anterior = año_seleccionado - 1
if año_anterior in df['año'].unique():
    if restaurante_sel == 'All Locations':
        df_año_anterior = df[df['año'] == año_anterior]
    else:
        df_año_anterior = df[(df['año'] == año_anterior) & (df['restaurante'] == restaurante_sel)]
    
    ventas_año_anterior = df_año_anterior['venta_pesos'].sum()
    if ventas_año_anterior > 0:
        variacion_yoy = ((ventas_totales - ventas_año_anterior) / ventas_año_anterior) * 100
    else:
        variacion_yoy = 0
else:
    variacion_yoy = None

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_str = f"{variacion_yoy:+.1f}% YoY" if variacion_yoy is not None else None
    st.metric(
        label="Total Revenue",
        value=f"${ventas_totales/1e6:.2f}M",
        delta=delta_str,
        help="Total revenue for the selected period"
    )

with col2:
    st.metric(
        label="Units Sold",
        value=f"{unidades_totales:,.0f}",
        help="Total number of units sold"
    )

with col3:
    st.metric(
        label="Operating Days",
        value=f"{dias_operacion}",
        help="Number of days with recorded sales"
    )

with col4:
    st.metric(
        label="Active Products",
        value=f"{productos_activos}",
        help="Number of unique products sold"
    )

# ==========================================
# MONTHLY TREND ANALYSIS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Monthly Revenue Trend</h2>", unsafe_allow_html=True)

# Prepare monthly data
ventas_mes = df_filtrado.groupby(['mes', 'mes_nombre'])['venta_pesos'].sum().reset_index()
ventas_mes = ventas_mes.sort_values('mes')

# Create professional chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=ventas_mes['mes_nombre'],
    y=ventas_mes['venta_pesos'],
    marker=dict(
        color=ventas_mes['venta_pesos'],
        colorscale=[[0, '#3b82f6'], [1, '#1e3a8a']],
        line=dict(color='#1e40af', width=1)
    ),
    text=[f'${v/1e6:.1f}M' for v in ventas_mes['venta_pesos']],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Highlight best and worst months
if len(ventas_mes) > 0:
    mejor_idx = ventas_mes['venta_pesos'].idxmax()
    peor_idx = ventas_mes['venta_pesos'].idxmin()
    
    fig.add_annotation(
        x=ventas_mes.loc[mejor_idx, 'mes_nombre'],
        y=ventas_mes.loc[mejor_idx, 'venta_pesos'],
        text="Peak",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#10b981",
        bgcolor="#10b981",
        font=dict(color="white", size=11),
        bordercolor="#10b981",
        borderwidth=2
    )
    
    fig.add_annotation(
        x=ventas_mes.loc[peor_idx, 'mes_nombre'],
        y=ventas_mes.loc[peor_idx, 'venta_pesos'],
        text="Low",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        bgcolor="#ef4444",
        font=dict(color="white", size=11),
        bordercolor="#ef4444",
        borderwidth=2
    )

fig.update_layout(
    title=dict(
        text=f'Monthly Revenue Distribution - {año_seleccionado}',
        font=dict(size=16, color='#1e293b', family='Arial')
    ),
    xaxis=dict(
        title='Month',
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title='Revenue (COP)',
        showgrid=True,
        gridcolor='#f1f5f9',
        zeroline=False
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=450,
    showlegend=False,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# LOCATION COMPARISON (IF ALL LOCATIONS)
# ==========================================

if restaurante_sel == 'All Locations':
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Location Performance Comparison</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Monthly evolution by location
        ventas_rest_mes = df_filtrado.groupby(['mes_nombre', 'restaurante', 'mes'])['venta_pesos'].sum().reset_index()
        ventas_rest_mes = ventas_rest_mes.sort_values('mes')
        
        fig = go.Figure()
        
        for restaurante in df_filtrado['restaurante'].unique():
            data = ventas_rest_mes[ventas_rest_mes['restaurante'] == restaurante]
            color = get_restaurante_color(restaurante)
            
            fig.add_trace(go.Scatter(
                x=data['mes_nombre'],
                y=data['venta_pesos'],
                name=restaurante,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color, line=dict(width=2, color='white')),
                hovertemplate=f'<b>{restaurante}</b><br>%{{x}}: $%{{y:,.0f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='Monthly Revenue by Location',
                font=dict(size=16, color='#1e293b')
            ),
            xaxis=dict(title='Month', showgrid=False),
            yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Annual ranking
        st.markdown("**Annual Ranking**")
        
        ventas_rest = df_filtrado.groupby('restaurante')['venta_pesos'].sum().sort_values(ascending=False)
        
        for i, (rest, venta) in enumerate(ventas_rest.items(), 1):
            position_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            porcentaje = (venta / ventas_rest.sum()) * 100
            color = get_restaurante_color(rest)
            
            st.markdown(f"""
            <div class='ranking-card' style='border-left: 4px solid {color}'>
                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                    <span class='ranking-position'>{position_emoji}</span>
                    <div style='flex: 1;'>
                        <div class='ranking-name'>{rest}</div>
                        <div class='ranking-value'>${venta/1e6:.2f}M</div>
                        <div class='ranking-percentage'>{porcentaje:.1f}% of total</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# DAY OF WEEK ANALYSIS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Day of Week Performance</h2>", unsafe_allow_html=True)

# Map day names
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_cortos = {
    'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed',
    'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'
}

ventas_dia = df_filtrado.groupby('dia_semana')['venta_pesos'].sum()
ventas_dia = ventas_dia.reindex(dias_orden)
ventas_dia.index = [dias_cortos[d] for d in ventas_dia.index]

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    
    colors = ['#3b82f6' if v == ventas_dia.max() else '#94a3b8' for v in ventas_dia.values]
    
    fig.add_trace(go.Bar(
        x=ventas_dia.index,
        y=ventas_dia.values,
        marker=dict(color=colors, line=dict(color='#1e40af', width=1)),
        text=[f'${v/1e6:.1f}M' for v in ventas_dia.values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Revenue by Day of Week',
            font=dict(size=16, color='#1e293b')
        ),
        xaxis=dict(title='Day', showgrid=False),
        yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    mejor_dia = ventas_dia.idxmax()
    peor_dia = ventas_dia.idxmin()
    
    st.markdown("**Performance Summary**")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(
            label="Peak Day",
            value=mejor_dia,
            delta=f"${ventas_dia[mejor_dia]/1e6:.1f}M"
        )
    
    with col_b:
        st.metric(
            label="Low Day",
            value=peor_dia,
            delta=f"${ventas_dia[peor_dia]/1e6:.1f}M",
            delta_color="inverse"
        )
    
    diferencia = ((ventas_dia[mejor_dia] - ventas_dia[peor_dia]) / ventas_dia[peor_dia]) * 100
    
    st.markdown(f"""
    <div class='insight-box'>
        <p><strong>Insight:</strong> {mejor_dia} outperforms {peor_dia} by {diferencia:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TOP PRODUCTS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Top 10 Products by Revenue</h2>", unsafe_allow_html=True)

top_productos = df_filtrado.groupby('producto').agg({
    'venta_pesos': 'sum',
    'cantidad_vendida_diaria': 'sum'
}).sort_values('venta_pesos', ascending=False).head(10)

fig = go.Figure()

fig.add_trace(go.Bar(
    y=top_productos.index[::-1],
    x=top_productos['venta_pesos'][::-1],
    orientation='h',
    marker=dict(
        color=top_productos['venta_pesos'][::-1],
        colorscale=[[0, '#93c5fd'], [1, '#1e3a8a']],
        line=dict(color='#1e40af', width=1)
    ),
    text=[f'${v/1e6:.2f}M' for v in top_productos['venta_pesos'][::-1]],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<br>Units: %{customdata:,.0f}<extra></extra>',
    customdata=top_productos['cantidad_vendida_diaria'][::-1]
))

fig.update_layout(
    title=dict(
        text='Top Revenue Generators',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
    yaxis=dict(title='', showgrid=False),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    showlegend=False,
    margin=dict(l=200)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# EXECUTIVE SUMMARY
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Executive Summary</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

ventas_promedio_dia = df_filtrado.groupby('fecha')['venta_pesos'].sum().mean()
producto_estrella = df_filtrado.groupby('producto')['venta_pesos'].sum().idxmax()
ticket_promedio = ventas_totales / unidades_totales if unidades_totales > 0 else 0

with col1:
    st.metric(
        label="Average Daily Revenue",
        value=f"${ventas_promedio_dia/1e3:.1f}K",
        help="Mean daily revenue across operating days"
    )

with col2:
    st.metric(
        label="Top Product",
        value=producto_estrella[:30] + "..." if len(producto_estrella) > 30 else producto_estrella,
        help="Highest revenue generating product"
    )

with col3:
    st.metric(
        label="Average Transaction Value",
        value=f"${ticket_promedio:,.0f}",
        help="Average revenue per unit sold"
    )

# Footer
st.markdown("<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.875rem;'>", unsafe_allow_html=True)
st.markdown(f"Report generated for fiscal year {año_seleccionado} • Data updated through {df_filtrado['fecha'].max().strftime('%B %d, %Y')}")
st.markdown("</div>", unsafe_allow_html=True)
