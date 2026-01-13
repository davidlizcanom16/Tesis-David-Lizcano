# pages/04_📊_Top_Performers.py
"""
Top Performers Analysis Dashboard
Identification and analysis of highest-performing products
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import cargar_datos, get_restaurante_color
from utils.metrics import calcular_top_productos

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Top Performers",
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
    
    /* Metric Cards */
    .metric-card-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .metric-card-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card-value {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    
    /* Product Detail Card */
    .product-detail-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .product-detail-label {
        color: #64748b;
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
    }
    
    .product-detail-value {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .product-detail-subtitle {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: white;
        border-left: 4px solid #10b981;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-box h3 {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    .recommendation-box ul {
        margin: 0;
        padding-left: 1.5rem;
        color: #475569;
    }
    
    .recommendation-box li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Star Product Box */
    .star-product-box {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .star-product-box h3 {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        opacity: 0.95;
    }
    
    .star-product-box h2 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .star-product-box .value {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .star-product-box .details {
        opacity: 0.9;
        line-height: 1.6;
    }
    
    /* Insight Box */
    .insight-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 0.375rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .insight-box p {
        margin: 0;
        color: #1e40af;
        font-weight: 500;
        line-height: 1.6;
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
    <h1 class='dashboard-title'>Top Performers Analysis</h1>
    <p class='dashboard-subtitle'>Strategic analysis of highest-performing products by revenue, consistency, and frequency</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FILTERS
# ==========================================

st.markdown("<div class='filter-section'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    años = ['All Years'] + sorted(df['año'].unique().tolist(), reverse=True)
    año_sel = st.selectbox("Fiscal Year", años, index=0, help="Filter by specific year or analyze all periods")

with col2:
    restaurantes = ['All Locations'] + sorted(df['restaurante'].unique().tolist())
    restaurante_sel = st.selectbox("Location", restaurantes, index=0, help="Filter by restaurant location")

with col3:
    top_n = st.selectbox("Top N", [5, 10, 15, 20], index=1, help="Number of top products to analyze")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FILTER DATA
# ==========================================

df_filtrado = df.copy()

if año_sel != 'All Years':
    df_filtrado = df_filtrado[df_filtrado['año'] == int(año_sel)]

if restaurante_sel != 'All Locations':
    df_filtrado = df_filtrado[df_filtrado['restaurante'] == restaurante_sel]
    color_principal = get_restaurante_color(restaurante_sel)
else:
    color_principal = '#3b82f6'

if len(df_filtrado) == 0:
    st.warning("No data available for selected filters")
    st.stop()

# ==========================================
# CALCULATE PRODUCT METRICS
# ==========================================

# Group by product
productos_metricas = df_filtrado.groupby('producto').agg({
    'venta_pesos': ['sum', 'mean', 'std'],
    'cantidad_vendida_diaria': ['sum', 'mean'],
    'fecha': 'count'
}).reset_index()

productos_metricas.columns = ['producto', 'ventas_totales', 'venta_promedio', 'venta_std', 
                               'unidades_totales', 'unidad_promedio', 'dias_venta']

# Calculate additional metrics
productos_metricas['ticket_promedio'] = productos_metricas['ventas_totales'] / productos_metricas['unidades_totales']
productos_metricas['consistencia'] = 1 / (1 + productos_metricas['venta_std'])
productos_metricas['frecuencia'] = productos_metricas['dias_venta'] / df_filtrado['fecha'].nunique()

# Composite score (sales + consistency + frequency)
productos_metricas['score'] = (
    (productos_metricas['ventas_totales'] / productos_metricas['ventas_totales'].max()) * 40 +
    (productos_metricas['consistencia'] / productos_metricas['consistencia'].max()) * 30 +
    (productos_metricas['frecuencia'] / productos_metricas['frecuencia'].max()) * 30
)

# Top N products
top_productos = productos_metricas.nlargest(top_n, 'score')

# ==========================================
# EXECUTIVE SUMMARY
# ==========================================

st.markdown("<h2 class='section-header'>Executive Summary</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

ventas_top = top_productos['ventas_totales'].sum()
participacion = (ventas_top / df_filtrado['venta_pesos'].sum()) * 100
unidades_top = top_productos['unidades_totales'].sum()
ticket_prom_top = top_productos['ticket_promedio'].mean()

with col1:
    st.metric(
        label=f"Top {top_n} Revenue",
        value=f"${ventas_top/1e6:.2f}M",
        help=f"Total revenue from top {top_n} products"
    )

with col2:
    st.metric(
        label="Revenue Share",
        value=f"{participacion:.1f}%",
        help="Percentage of total revenue from top performers"
    )

with col3:
    st.metric(
        label="Units Sold",
        value=f"{unidades_top:,.0f}",
        help="Total units sold across top products"
    )

with col4:
    st.metric(
        label="Avg Transaction",
        value=f"${ticket_prom_top/1e3:.1f}K",
        help="Average revenue per unit"
    )

# ==========================================
# PRODUCT RANKING
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(f"<h2 class='section-header'>Top {top_n} Product Ranking</h2>", unsafe_allow_html=True)

# Main ranking chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=top_productos['producto'],
    x=top_productos['ventas_totales'],
    orientation='h',
    marker=dict(
        color=top_productos['score'],
        colorscale=[[0, '#93c5fd'], [1, '#1e3a8a']],
        showscale=True,
        colorbar=dict(title="Performance Score", x=1.15)
    ),
    text=[f'${v/1e6:.2f}M' for v in top_productos['ventas_totales']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<br>Score: %{marker.color:.1f}<extra></extra>'
))

fig.update_layout(
    title=dict(
        text=f'Top {top_n} Products by Total Revenue',
        font=dict(size=16, color='#1e293b')
    ),
    xaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
    yaxis=dict(title='', showgrid=False, categoryorder='total ascending'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=max(400, top_n * 40),
    showlegend=False,
    margin=dict(l=250, r=150)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MULTIDIMENSIONAL ANALYSIS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Multidimensional Performance Analysis</h2>", unsafe_allow_html=True)

# Scatter plot: Revenue vs Frequency
fig = px.scatter(
    top_productos,
    x='frecuencia',
    y='ventas_totales',
    size='unidades_totales',
    color='score',
    hover_name='producto',
    labels={
        'frecuencia': 'Sales Frequency (% of days)',
        'ventas_totales': 'Total Revenue (COP)',
        'score': 'Performance Score'
    },
    title='Revenue vs Frequency Analysis (bubble size = units sold)',
    color_continuous_scale=[[0, '#93c5fd'], [1, '#1e3a8a']]
)

fig.update_traces(marker=dict(line=dict(width=2, color='white')))
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    xaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickformat='.0%'),
    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
)

st.plotly_chart(fig, use_container_width=True)

# Insight
productos_alto_valor = top_productos[top_productos['ticket_promedio'] > top_productos['ticket_promedio'].median()]

st.markdown(f"""
<div class='insight-box'>
    <p><strong>Key Insight:</strong> {len(productos_alto_valor)} of the top {top_n} products have an average transaction value 
    above ${top_productos['ticket_promedio'].median()/1e3:.1f}K. These are <strong>premium products</strong> 
    that generate high value per sale and should be prioritized in inventory and marketing strategies.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# DETAILED ANALYSIS - TOP 5
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Detailed Product Analysis - Top 5</h2>", unsafe_allow_html=True)

for idx, (_, producto) in enumerate(top_productos.head(5).iterrows(), 1):
    
    # Product data
    df_producto = df_filtrado[df_filtrado['producto'] == producto['producto']]
    
    position_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "•"
    
    with st.expander(f"{position_emoji} #{idx} - {producto['producto']}", expanded=(idx <= 2)):
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Temporal evolution
            ventas_tiempo = df_producto.groupby('fecha')['venta_pesos'].sum().reset_index()
            
            fig = go.Figure()
            
            # Line chart with area fill
            fig.add_trace(go.Scatter(
                x=ventas_tiempo['fecha'],
                y=ventas_tiempo['venta_pesos'],
                mode='lines',
                name='Daily Revenue',
                line=dict(color=color_principal, width=2),
                fill='tozeroy',
                fillcolor=f'rgba(59, 130, 246, 0.1)',
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            
            # Average line
            promedio = ventas_tiempo['venta_pesos'].mean()
            fig.add_hline(
                y=promedio,
                line_dash="dash",
                line_color="#ef4444",
                line_width=2,
                annotation_text=f"Average: ${promedio/1e3:.1f}K",
                annotation_position="right"
            )
            
            fig.update_layout(
                title=dict(
                    text='Revenue Evolution',
                    font=dict(size=14, color='#1e293b')
                ),
                xaxis=dict(title='Date', showgrid=False),
                yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=300,
                hovermode='x unified',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key metrics
            st.markdown("**Key Metrics**")
            
            metrics_data = [
                ("Total Revenue", f"${producto['ventas_totales']/1e6:.2f}M", color_principal),
                ("Units Sold", f"{producto['unidades_totales']:,.0f}", "#64748b"),
                ("Avg Transaction", f"${producto['ticket_promedio']/1e3:.1f}K", "#64748b"),
                ("Frequency", f"{producto['frecuencia']*100:.0f}%", "#64748b")
            ]
            
            for label, value, color in metrics_data:
                st.markdown(f"""
                <div class='product-detail-card'>
                    <div class='product-detail-label'>{label}</div>
                    <div class='product-detail-value' style='color: {color};'>{value}</div>
                    {f"<div class='product-detail-subtitle'>{producto['dias_venta']:.0f} sales days</div>" if label == "Frequency" else ""}
                </div>
                """, unsafe_allow_html=True)
        
        # Distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # By day of week
            ventas_dia_semana = df_producto.groupby('dia_semana')['venta_pesos'].sum()
            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ventas_dia_semana = ventas_dia_semana.reindex(dias_orden, fill_value=0)
            
            dias_short = {
                'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed',
                'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'
            }
            ventas_dia_semana.index = [dias_short[d] for d in ventas_dia_semana.index]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=ventas_dia_semana.index,
                y=ventas_dia_semana.values,
                marker=dict(
                    color=ventas_dia_semana.values,
                    colorscale=[[0, '#93c5fd'], [1, '#1e3a8a']],
                    showscale=False
                ),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text='Revenue by Day of Week', font=dict(size=14, color='#1e293b')),
                xaxis=dict(title='', showgrid=False),
                yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=250,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # By restaurant or by month
            if restaurante_sel == 'All Locations':
                ventas_rest = df_producto.groupby('restaurante')['venta_pesos'].sum()
                
                fig = go.Figure()
                
                colors = [get_restaurante_color(r) for r in ventas_rest.index]
                
                fig.add_trace(go.Pie(
                    labels=ventas_rest.index,
                    values=ventas_rest.values,
                    marker=dict(colors=colors, line=dict(color='white', width=2)),
                    textposition='inside',
                    textinfo='percent',
                    hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text='Distribution by Location', font=dict(size=14, color='#1e293b')),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    height=250
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Monthly distribution
                ventas_mes = df_producto.groupby('mes')['venta_pesos'].sum().sort_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=ventas_mes.index,
                    y=ventas_mes.values,
                    marker=dict(color='#3b82f6', line=dict(color='#1e40af', width=1)),
                    hovertemplate='<b>Month %{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text='Revenue by Month', font=dict(size=14, color='#1e293b')),
                    xaxis=dict(title='Month', showgrid=False),
                    yaxis=dict(title='Revenue (COP)', showgrid=True, gridcolor='#f1f5f9'),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# STRATEGIC RECOMMENDATIONS
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Strategic Recommendations</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='recommendation-box'>
        <h3>Recommended Actions</h3>
        <ul>
            <li>Ensure consistent availability of top-performing products</li>
            <li>Implement targeted promotions during low-performance periods</li>
            <li>Feature these products prominently in menus and marketing materials</li>
            <li>Train staff on effective upselling techniques for top products</li>
            <li>Conduct margin analysis to optimize profitability</li>
            <li>Monitor inventory levels to prevent stockouts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Star product
    producto_estrella = top_productos.iloc[0]
    
    st.markdown(f"""
    <div class='star-product-box'>
        <h3>Top Performer #1</h3>
        <h2>{producto_estrella['producto'][:50]}{'...' if len(producto_estrella['producto']) > 50 else ''}</h2>
        <div class='value'>${producto_estrella['ventas_totales']/1e6:.2f}M</div>
        <div class='details'>
            • {producto_estrella['unidades_totales']:.0f} units sold<br>
            • Performance score: {producto_estrella['score']:.1f}/100<br>
            • Sales frequency: {producto_estrella['frecuencia']*100:.0f}% of days<br>
            • Avg transaction: ${producto_estrella['ticket_promedio']/1e3:.1f}K
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SUMMARY TABLE
# ==========================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Complete Summary Table</h2>", unsafe_allow_html=True)

tabla_resumen = top_productos[['producto', 'ventas_totales', 'unidades_totales', 
                                 'ticket_promedio', 'frecuencia', 'score']].copy()
tabla_resumen['frecuencia'] = tabla_resumen['frecuencia'] * 100

# Rename columns for display
tabla_resumen.columns = ['Product', 'Total Revenue', 'Units Sold', 'Avg Transaction', 'Frequency (%)', 'Score']

st.dataframe(
    tabla_resumen.style.format({
        'Total Revenue': '${:,.0f}',
        'Units Sold': '{:,.0f}',
        'Avg Transaction': '${:,.0f}',
        'Frequency (%)': '{:.0f}%',
        'Score': '{:.1f}'
    }).background_gradient(subset=['Score'], cmap='Blues'),
    use_container_width=True,
    height=400
)

# Footer
st.markdown("<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.875rem;'>", unsafe_allow_html=True)
st.markdown(f"Analysis of top {top_n} performing products • {len(df_filtrado)} total transactions analyzed • Performance score based on revenue (40%), consistency (30%), and frequency (30%)")
st.markdown("</div>", unsafe_allow_html=True)
