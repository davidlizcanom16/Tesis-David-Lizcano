# pages/03_📊_Daily_Performance.py
"""
Daily Performance Dashboard
Detailed analysis of specific day operations and product performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import cargar_datos, get_restaurante_color

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Daily Performance",
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
    
    /* Date Badge */
    .date-badge {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .date-badge .day-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .date-badge .date-text {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Event Badge */
    .event-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
        margin-bottom: 1rem;
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
    
    /* Product Card */
    .product-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s;
    }
    
    .product-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transform: translateX(4px);
    }
    
    .product-card .rank {
        display: inline-block;
        width: 1.5rem;
        height: 1.5rem;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 1.5rem;
        font-weight: 700;
        font-size: 0.875rem;
        margin-right: 0.5rem;
    }
    
    .product-card .name {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .product-card .value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 0.25rem;
    }
    
    .product-card .units {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Restaurant Card */
    .restaurant-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s;
    }
    
    .restaurant-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .restaurant-card .position {
        display: inline-block;
        font-size: 1.25rem;
        margin-right: 0.5rem;
    }
    
    .restaurant-card .name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .restaurant-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .restaurant-card .percentage {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Performance Badge */
    .performance-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .performance-excellent {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    
    .performance-normal {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    .performance-low {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
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
    <h1 class='dashboard-title'>Daily Performance Dashboard</h1>
    <p class='dashboard-subtitle'>Detailed operational analysis for specific dates</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FILTERS
# ==========================================

st.markdown("<div class='filter-section'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    fecha_min = df['fecha'].min().date()
    fecha_max = df['fecha'].max().date()
    
    fecha_sel = st.date_input(
        "Select Date",
        value=fecha_max,
        min_value=fecha_min,
        max_value=fecha_max,
        help="Choose a specific date for analysis"
    )

with col2:
    restaurantes = ['All Locations'] + sorted(df['restaurante'].unique().tolist())
    restaurante_sel = st.selectbox(
        "Location",
        restaurantes,
        index=0,
        help="Filter by restaurant location"
    )

with col3:
    if restaurante_sel != 'All Locations':
        color = get_restaurante_color(restaurante_sel)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background: {color}; color: white; padding: 0.5rem 1rem; 
                    border-radius: 0.375rem; text-align: center; font-weight: 600;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            {restaurante_sel}
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FILTER DATA
# ==========================================

if restaurante_sel == 'All Locations':
    df_dia = df[df['fecha'] == pd.to_datetime(fecha_sel)]
else:
    df_dia = df[(df['fecha'] == pd.to_datetime(fecha_sel)) & (df['restaurante'] == restaurante_sel)]

if len(df_dia) == 0:
    st.warning(f"No data available for {restaurante_sel} on {fecha_sel}")
    st.stop()

# ==========================================
# DATE INFO AND EVENT STATUS
# ==========================================

# Day mapping
dia_semana_map = {
    'Monday': 'Monday', 'Tuesday': 'Tuesday', 'Wednesday': 'Wednesday',
    'Thursday': 'Thursday', 'Friday': 'Friday', 'Saturday': 'Saturday', 'Sunday': 'Sunday'
}

dia_nombre = pd.to_datetime(fecha_sel).day_name()

# Display date
st.markdown(f"""
<div class='date-badge'>
    <div class='day-name'>{dia_nombre}</div>
    <div class='date-text'>{fecha_sel.strftime('%B %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)

# Event indicator
if df_dia['tiene_evento'].max() == 1:
    st.markdown("""
    <div class='event-badge'>
        Special Event Day
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# KEY METRICS
# ==========================================

st.markdown("<h2 class='section-header'>Daily Performance Metrics</h2>", unsafe_allow_html=True)

# Calculate metrics
ventas_dia = df_dia['venta_pesos'].sum()
unidades_dia = df_dia['cantidad_vendida_diaria'].sum()
productos_dia = df_dia['producto'].nunique()
ticket_prom = ventas_dia / len(df_dia) if len(df_dia) > 0 else 0

# Compare with similar days
if restaurante_sel == 'All Locations':
    df_dias_similares = df[df['dia_semana'] == dia_nombre]
else:
    df_dias_similares = df[(df['dia_semana'] == dia_nombre) & (df['restaurante'] == restaurante_sel)]

ventas_promedio_similar = df_dias_similares.groupby('fecha')['venta_pesos'].sum().mean()
vs_promedio = ((ventas_dia - ventas_promedio_similar) / ventas_promedio_similar * 100) if ventas_promedio_similar > 0 else 0

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Daily Revenue",
        value=f"${ventas_dia/1e3:.1f}K",
        delta=f"{vs_promedio:+.1f}% vs avg",
        help=f"Compared to average {dia_nombre} revenue"
    )

with col2:
    st.metric(
        label="Units Sold",
        value=f"{unidades_dia:,.0f}",
        help="Total units sold on this day"
    )

with col3:
    st.metric(
        label="Products Sold",
        value=productos_dia,
        help="Number of unique products sold"
    )

with col4:
    st.metric(
        label="Avg Transaction",
        value=f"${ticket_prom:,.0f}",
        help="Average revenue per product line"
    )

# ==========================================
# LOCATION BREAKDOWN (IF ALL)
# ==========================================

if restaurante_sel == 'All Locations':
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Revenue by Location</h2>", unsafe_allow_html=True)
    
    ventas_rest = df_dia.groupby('restaurante')['venta_pesos'].sum().sort_values(ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        fig = go.Figure()
        
        colors = [get_restaurante_color(r) for r in ventas_rest.index]
        
        fig.add_trace(go.Bar(
            x=ventas_rest.index,
            y=ventas_rest.values,
            marker=dict(
                color=colors,
                line=dict(color='#1e40af', width=1)
            ),
            text=[f'${v/1e3:.1f}K' for v in ventas_rest.values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Location Performance Comparison',
                font=dict(size=16, color='#1e293b')
            ),
            xaxis=dict(title='', showgrid=False),
            yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Performance Ranking**")
        
        for i, (rest, venta) in enumerate(ventas_rest.items(), 1):
            porcentaje = (venta / ventas_rest.sum()) * 100
            color = get_restaurante_color(rest)
            
            position_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            
            st.markdown(f"""
            <div class='restaurant-card' style='border-left-color: {color};'>
                <div>
                    <span class='position'>{position_emoji}</span>
                    <span class='name'>{rest}</span>
                </div>
                <div class='value' style='color: {color};'>${venta/1e3:.1f}K</div>
                <div class='percentage'>{porcentaje:.1f}% of daily total</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# PRODUCT PERFORMANCE
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Product Performance Analysis</h2>", unsafe_allow_html=True)

productos_dia_data = df_dia.groupby('producto').agg({
    'cantidad_vendida_diaria': 'sum',
    'venta_pesos': 'sum'
}).sort_values('venta_pesos', ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 5 by Revenue**")
    
    top_5_ventas = productos_dia_data.head(5)
    
    for i, (producto, row) in enumerate(top_5_ventas.iterrows(), 1):
        st.markdown(f"""
        <div class='product-card' style='border-left-color: #3b82f6;'>
            <div>
                <span class='rank'>{i}</span>
                <span class='name'>{producto[:45]}{'...' if len(producto) > 45 else ''}</span>
            </div>
            <div class='value'>${row['venta_pesos']/1e3:.1f}K</div>
            <div class='units'>{row['cantidad_vendida_diaria']:.0f} units sold</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("**Top 5 by Volume**")
    
    top_5_cantidad = productos_dia_data.sort_values('cantidad_vendida_diaria', ascending=False).head(5)
    
    for i, (producto, row) in enumerate(top_5_cantidad.iterrows(), 1):
        st.markdown(f"""
        <div class='product-card' style='border-left-color: #10b981;'>
            <div>
                <span class='rank' style='background: #10b981;'>{i}</span>
                <span class='name'>{producto[:45]}{'...' if len(producto) > 45 else ''}</span>
            </div>
            <div class='value' style='color: #10b981;'>{row['cantidad_vendida_diaria']:.0f} units</div>
            <div class='units'>${row['venta_pesos']/1e3:.1f}K revenue</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# HISTORICAL COMPARISON
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(f"<h2 class='section-header'>Comparison with Previous {dia_nombre}s</h2>", unsafe_allow_html=True)

ventas_mismo_dia = df_dias_similares.groupby('fecha')['venta_pesos'].sum().sort_index().tail(8)

fig = go.Figure()

# Historical trend line
fig.add_trace(go.Scatter(
    x=ventas_mismo_dia.index,
    y=ventas_mismo_dia.values,
    mode='lines+markers',
    name='Historical Performance',
    line=dict(color='#94a3b8', width=2),
    marker=dict(size=8, color='#94a3b8'),
    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Highlight selected day
if pd.to_datetime(fecha_sel) in ventas_mismo_dia.index:
    venta_sel = ventas_mismo_dia[pd.to_datetime(fecha_sel)]
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(fecha_sel)],
        y=[venta_sel],
        mode='markers',
        name='Selected Day',
        marker=dict(size=16, color='#3b82f6', symbol='star', line=dict(width=2, color='white')),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Revenue: $%{y:,.0f}<br><i>Selected Day</i><extra></extra>'
    ))

# Average line
promedio_similares = ventas_mismo_dia.mean()
fig.add_hline(
    y=promedio_similares,
    line_dash="dash",
    line_color="#ef4444",
    line_width=2,
    annotation_text=f"Average: ${promedio_similares/1e3:.1f}K",
    annotation_position="right",
    annotation=dict(font=dict(size=12, color="#ef4444"))
)

fig.update_layout(
    title=dict(
        text=f'Last 8 {dia_nombre}s Performance',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(title='Date', showgrid=False),
    yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=400,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Performance assessment
diferencia = ventas_dia - promedio_similares
porcentaje_dif = (diferencia / promedio_similares) * 100 if promedio_similares > 0 else 0

if porcentaje_dif > 10:
    performance_class = "performance-excellent"
    performance_label = "Excellent Performance"
    performance_icon = "✓"
    st.markdown(f"""
    <div style='background: #d1fae5; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.375rem; margin-top: 1rem;'>
        <p style='margin: 0; color: #065f46; font-weight: 600;'>
            <strong>{performance_icon} {performance_label}:</strong> Revenue was <strong>{porcentaje_dif:.1f}% above</strong> 
            the average for {dia_nombre}s (${promedio_similares/1e3:.1f}K)
        </p>
    </div>
    """, unsafe_allow_html=True)
elif porcentaje_dif < -10:
    performance_class = "performance-low"
    performance_label = "Below Average Performance"
    performance_icon = "⚠"
    st.markdown(f"""
    <div style='background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 0.375rem; margin-top: 1rem;'>
        <p style='margin: 0; color: #991b1b; font-weight: 600;'>
            <strong>{performance_icon} {performance_label}:</strong> Revenue was <strong>{abs(porcentaje_dif):.1f}% below</strong> 
            the average for {dia_nombre}s (${promedio_similares/1e3:.1f}K)
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    performance_class = "performance-normal"
    performance_label = "Normal Performance"
    performance_icon = "○"
    st.markdown(f"""
    <div style='background: #dbeafe; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 0.375rem; margin-top: 1rem;'>
        <p style='margin: 0; color: #1e40af; font-weight: 600;'>
            <strong>{performance_icon} {performance_label}:</strong> Revenue was within normal range 
            of the average for {dia_nombre}s (${promedio_similares/1e3:.1f}K)
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.875rem;'>", unsafe_allow_html=True)
st.markdown(f"Daily report for {fecha_sel.strftime('%B %d, %Y')} ({dia_nombre}) • {productos_dia} products sold • {unidades_dia:,.0f} total units")
st.markdown("</div>", unsafe_allow_html=True)
