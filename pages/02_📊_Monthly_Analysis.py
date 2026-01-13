# pages/02_📊_Monthly_Analysis.py
"""
Monthly Analysis Dashboard
Detailed monthly performance metrics and trends
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
    page_title="Monthly Analysis",
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
    
    /* Highlight Cards */
    .highlight-card-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }
    
    .highlight-card-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
    }
    
    .highlight-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.95;
    }
    
    .highlight-card .date {
        font-size: 1rem;
        margin: 0;
        opacity: 0.9;
    }
    
    .highlight-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0 0 0;
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
        transform: translateX(4px);
    }
    
    .restaurant-card .name {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .restaurant-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 0.25rem;
    }
    
    .restaurant-card .percentage {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Insight Box */
    .insight-box {
        background: white;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 0.375rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .insight-box h4 {
        margin: 0 0 0.75rem 0;
        color: #1e293b;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .insight-box ul {
        margin: 0;
        padding-left: 1.25rem;
    }
    
    .insight-box li {
        color: #475569;
        margin-bottom: 0.375rem;
        line-height: 1.5;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-stable {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-moderate {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-volatile {
        background: #fee2e2;
        color: #991b1b;
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
    <h1 class='dashboard-title'>Monthly Analysis Dashboard</h1>
    <p class='dashboard-subtitle'>Detailed performance metrics and daily trends analysis</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FILTERS
# ==========================================

st.markdown("<div class='filter-section'>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

with col1:
    años = sorted(df['año'].unique(), reverse=True)
    año_sel = st.selectbox("Year", años, index=0, help="Select fiscal year")

with col2:
    meses_disponibles = sorted(df[df['año'] == año_sel]['mes'].unique(), reverse=True)
    mes_sel = st.selectbox("Month", meses_disponibles, index=0, help="Select month for analysis")

with col3:
    restaurantes = ['All Locations'] + sorted(df['restaurante'].unique().tolist())
    restaurante_sel = st.selectbox("Location", restaurantes, index=0, help="Filter by restaurant location")

with col4:
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
    df_mes = df[(df['año'] == año_sel) & (df['mes'] == mes_sel)]
else:
    df_mes = df[(df['año'] == año_sel) & (df['mes'] == mes_sel) & (df['restaurante'] == restaurante_sel)]

if len(df_mes) == 0:
    st.warning(f"No data available for {restaurante_sel} in this period")
    st.stop()

# Previous month for comparison
if mes_sel > 1:
    df_mes_anterior = df[(df['año'] == año_sel) & (df['mes'] == mes_sel - 1)]
    if restaurante_sel != 'All Locations':
        df_mes_anterior = df_mes_anterior[df_mes_anterior['restaurante'] == restaurante_sel]
else:
    df_mes_anterior = df[(df['año'] == año_sel - 1) & (df['mes'] == 12)]
    if restaurante_sel != 'All Locations':
        df_mes_anterior = df_mes_anterior[df_mes_anterior['restaurante'] == restaurante_sel]

# ==========================================
# KEY METRICS
# ==========================================

# Month names
meses_nombres = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

st.markdown(f"<h2 class='section-header'>{meses_nombres[mes_sel]} {año_sel} - Performance Overview</h2>", unsafe_allow_html=True)

# Calculate metrics
ventas_mes = df_mes['venta_pesos'].sum()
ventas_anterior = df_mes_anterior['venta_pesos'].sum()
cambio = ((ventas_mes - ventas_anterior) / ventas_anterior * 100) if ventas_anterior > 0 else 0

dias_operacion = df_mes['fecha'].nunique()
ventas_prom_dia = df_mes.groupby('fecha')['venta_pesos'].sum().mean()

# Best performing day
dias_semana_map = {
    'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed',
    'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'
}
mejor_dia_sem = df_mes.groupby('dia_semana')['venta_pesos'].sum().idxmax()
mejor_dia_sem_short = di
