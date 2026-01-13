# pages/06_📊_Demand_Forecaster.py
"""
Intelligent Demand Forecaster
XGBoost-based forecasting with confidence intervals and economic impact analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import cargar_datos
from utils.feature_engineering import create_all_features, get_feature_columns
from utils.model_trainer import (
    XGBoostPredictor, 
    calculate_metrics, 
    generate_alerts, 
    estimar_costos_unitarios, 
    calcular_costo_error, 
    generar_baselines, 
    comparar_metodos, 
    calcular_ahorro_economico
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Demand Forecaster",
    page_icon="📊",
    layout="wide"
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
    .forecaster-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .forecaster-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .forecaster-subtitle {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin: 0.75rem 0 0 0;
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
    
    /* Alert Boxes */
    .alert-box {
        padding: 1rem 1.25rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
        border-left: 4px solid;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .alert-warning {
        background: #fef3c7;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-info {
        background: #dbeafe;
        border-left-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-success {
        background: #d1fae5;
        border-left-color: #10b981;
        color: #065f46;
    }
    
    .alert-critical {
        background: #fee2e2;
        border-left-color: #ef4444;
        color: #991b1b;
    }
    
    .alert-box strong {
        font-weight: 700;
    }
    
    /* Progress Info Box */
    .progress-info {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Economic Impact Card */
    .economic-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .economic-card h4 {
        color: #1e293b;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    /* Methodology Box */
    .methodology-box {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .methodology-box h4 {
        color: #1e293b;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
    }
    
    /* Table Highlight */
    .highlight-row {
        background-color: #d1fae5;
        font-weight: 600;
    }
    
    /* Interpretation Box */
    .interpretation-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .interpretation-box p {
        margin: 0;
        color: #1e40af;
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

@st.cache_data
def load_data():
    """Load consolidated data"""
    return cargar_datos()

df_all = load_data()

if df_all is None:
    st.error("⚠️ Unable to load data sources")
    st.stop()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='forecaster-header'>
    <h1 class='forecaster-title'>Intelligent Demand Forecaster</h1>
    <p class='forecaster-subtitle'>Advanced ML forecasting with XGBoost, confidence intervals, and economic impact analysis</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================

st.sidebar.header("Model Configuration")

# Restaurant selection
restaurante = st.sidebar.selectbox(
    "Location",
    options=sorted(df_all['restaurante'].unique()),
    index=0
)

# Filter products
df_restaurante = df_all[df_all['restaurante'] == restaurante].copy()
productos = sorted(df_restaurante['descripcion_producto'].dropna().unique())

# Product selection
producto_seleccionado = st.sidebar.selectbox(
    "Product",
    options=productos,
    format_func=lambda x: x[:50] + "..." if len(x) > 50 else x
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")

# Horizon
horizonte = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=60,
    value=14,
    step=7,
    help="Number of days to forecast"
)

# Optuna trials
n_trials = st.sidebar.slider(
    "Optimization Trials",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="More trials = better model but slower (recommended: 20)"
)

# Train/test split
train_val_split = st.sidebar.slider(
    "Training Data %",
    min_value=60,
    max_value=90,
    value=80,
    step=5
)

st.sidebar.markdown("---")

# TRAIN BUTTON
entrenar = st.sidebar.button(
    "Train & Forecast",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Best Practices:**
- 20 trials optimal for accuracy/speed
- 80% training split recommended
- Shorter horizons (7-14 days) more accurate
""")

# ==========================================
# PRODUCT INFORMATION
# ==========================================

# Filter product data
df_producto = df_restaurante[df_restaurante['descripcion_producto'] == producto_seleccionado].copy()
df_producto = df_producto.sort_values('fecha').reset_index(drop=True)

# Validation
if len(df_producto) == 0:
    st.warning(f"No data available for {producto_seleccionado}")
    st.stop()

if 'cantidad_vendida_diaria' not in df_producto.columns:
    st.error("Column 'cantidad_vendida_diaria' not found in dataset")
    st.info(f"Available columns: {', '.join(df_producto.columns)}")
    st.stop()

# Product metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Historical Days", len(df_producto))

with col2:
    promedio = df_producto['cantidad_vendida_diaria'].mean()
    st.metric("Daily Average", f"{promedio:.1f} units")

with col3:
    std = df_producto['cantidad_vendida_diaria'].std()
    st.metric("Std Deviation", f"{std:.1f}")

with col4:
    periodo = f"{df_producto['fecha'].min().strftime('%Y-%m')} / {df_producto['fecha'].max().strftime('%Y-%m')}"
    st.metric("Period", periodo)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# HISTORICAL CHART
# ==========================================

st.markdown("<h2 class='section-header'>Historical Sales Data</h2>", unsafe_allow_html=True)

fig_historico = go.Figure()

fig_historico.add_trace(go.Scatter(
    x=df_producto['fecha'],
    y=df_producto['cantidad_vendida_diaria'],
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='#3b82f6', width=2),
    marker=dict(size=4, color='#3b82f6'),
    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Units: %{y:.1f}<extra></extra>'
))

fig_historico.update_layout(
    xaxis_title="Date",
    yaxis_title="Units Sold",
    hovermode='x unified',
    template='plotly_white',
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=400,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
)

st.plotly_chart(fig_historico, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ==========================================
# TRAINING AND PREDICTION
# ==========================================

if entrenar:
    
    with st.spinner("Training model..."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Feature Engineering
            status_text.text("Building features...")
            progress_bar.progress(10)
            
            df_features = create_all_features(df_producto)
            progress_bar.progress(20)
            
            # 2. Data cleaning
            status_text.text("Cleaning data...")
            
            essential_cols = ['cantidad_vendida_diaria']
            if 'lag_1' in df_features.columns:
                essential_cols.append('lag_1')
            if 'lag_7' in df_features.columns:
                essential_cols.append('lag_7')
            
            df_clean = df_features.dropna(subset=essential_cols).copy()
            progress_bar.progress(25)
            
            # Fill remaining NaN
            status_text.text("Filling missing values...")
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
            progress_bar.progress(30)
            
            # Verify minimum data
            if len(df_clean) < 50:
                st.error(f"Insufficient data after cleaning: {len(df_clean)} days")
                st.stop()
            
            df_features = df_clean
            
            # 3. Prepare features and target
            status_text.text("Preparing features...")
            progress_bar.progress(35)
            
            feature_cols = get_feature_columns()
            feature_cols = [col for col in feature_cols if col in df_features.columns]
            
            X = df_features[feature_cols].fillna(0)
            y = df_features['cantidad_vendida_diaria']
            
            progress_bar.progress(40)
            
            # 4. Temporal split
            status_text.text("Splitting data...")
            
            split_idx = int(len(X) * train_val_split / 100)
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            progress_bar.progress(45)
            
            # 5. Internal validation split
            status_text.text("Creating validation set...")
            
            val_split = int(len(X_train) * 0.8)
            X_train_inner = X_train[:val_split]
            y_train_inner = y_train[:val_split]
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            
            progress_bar.progress(50)
            
            # 6. Train model (MOST TIME-CONSUMING STEP)
            status_text.text(f"Training XGBoost ({n_trials} trials)... This may take a few moments")
            
            predictor = XGBoostPredictor(n_trials=n_trials, confidence_level=0.95)
            predictor.train(X_train_inner, y_train_inner, X_val, y_val)
            
            progress_bar.progress(70)
            
            # 7. Predict on test
            status_text.text("Evaluating on test set...")
            
            y_pred_test, y_pred_test_lower, y_pred_test_upper = predictor.predict(X_test, return_intervals=True)
            
            progress_bar.progress(80)
            
            # 8. Economic analysis
            status_text.text("Calculating economic impact...")
            
            costos_unitarios = estimar_costos_unitarios(df_producto)
            
            # Generate baselines
            df_with_baselines = generar_baselines(df_features)
            df_test_baselines = df_with_baselines[split_idx:].copy()
            
            # Compare methods
            df_comparacion = comparar_metodos(df_test_baselines, y_pred_test, costos_unitarios)
            
            # Calculate savings
            ahorro_info = calcular_ahorro_economico(df_comparacion)
            
            progress_bar.progress(82)
            
            # 9. Future prediction setup
            status_text.text("Preparing future forecast...")
            
            last_date = df_producto['fecha'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizonte, freq='D')
            
            # Create future dataframe
            df_future = pd.DataFrame({
                'fecha': future_dates,
                'cantidad_vendida_diaria': df_producto['cantidad_vendida_diaria'].tail(30).mean()
            })
            
            # Apply feature engineering
            df_future = create_all_features(df_future)
            
            # Ensure all features exist
            for col in feature_cols:
                if col not in df_future.columns:
                    df_future[col] = 0
            
            # Fill NaN
            df_future = df_future.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            X_future = df_future[feature_cols].fillna(0)
            
            progress_bar.progress(85)
            
            # 10. Generate forecast
            status_text.text("Generating future predictions...")
            
            y_pred_future, y_pred_future_lower, y_pred_future_upper = predictor.predict(
                X_future[:horizonte], 
                return_intervals=True
            )
            
            progress_bar.progress(90)
            
            # 11. Calculate metrics
            status_text.text("Calculating performance metrics...")
            
            metrics = calculate_metrics(y_test.values, y_pred_test)
            
            progress_bar.progress(95)
            
            # 12. Generate alerts
            status_text.text("Generating intelligent alerts...")
            
            historical_mean = df_producto['cantidad_vendida_diaria'].mean()
            historical_std = df_producto['cantidad_vendida_diaria'].std()
            
            alerts = generate_alerts(
                y_pred_future, 
                y_pred_future_lower, 
                y_pred_future_upper, 
                historical_mean, 
                historical_std
            )
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            # 13. Save to session state
            st.session_state.update({
                'predictor': predictor,
                'metrics': metrics,
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'y_pred_test_lower': y_pred_test_lower,
                'y_pred_test_upper': y_pred_test_upper,
                'df_test': df_features[split_idx:].copy(),
                'future_dates': future_dates,
                'y_pred_future': y_pred_future,
                'y_pred_future_lower': y_pred_future_lower,
                'y_pred_future_upper': y_pred_future_upper,
                'df_producto': df_producto,
                'alerts': alerts,
                'producto_nombre': producto_seleccionado,
                'costos_unitarios': costos_unitarios,
                'df_comparacion': df_comparacion,
                'ahorro_info': ahorro_info
            })
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✓ Model trained successfully!")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error(f"⚠️ Training error: {str(e)}")
            import traceback
            with st.expander("View Technical Details"):
                st.code(traceback.format_exc())

# ==========================================
# DISPLAY RESULTS
# ==========================================

if 'predictor' in st.session_state:
    
    # Extract data
    predictor = st.session_state['predictor']
    metrics = st.session_state['metrics']
    alerts = st.session_state['alerts']
    future_dates = st.session_state['future_dates']
    y_pred_future = st.session_state['y_pred_future']
    y_pred_future_lower = st.session_state['y_pred_future_lower']
    y_pred_future_upper = st.session_state['y_pred_future_upper']
    df_producto = st.session_state['df_producto']
    producto_nombre = st.session_state.get('producto_nombre', producto_seleccionado)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ==========================================
    # INTELLIGENT ALERTS
    # ==========================================
    
    if alerts:
        st.markdown("<h2 class='section-header'>Intelligent Alerts</h2>", unsafe_allow_html=True)
        
        for alert in alerts:
            alert_class = 'alert-warning' if alert['type'] == 'warning' else \
                         'alert-success' if alert['type'] == 'success' else \
                         'alert-critical' if alert['type'] == 'critical' else 'alert-info'
            
            st.markdown(f"""
            <div class='alert-box {alert_class}'>
                <strong>{alert['title']}</strong><br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ==========================================
    # MODEL METRICS
    # ==========================================
    
    st.markdown("<h2 class='section-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.2f}", help="Mean Absolute Error")
    with col2:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Root Mean Squared Error")
    with col3:
        st.metric("MAPE", f"{metrics['MAPE']:.1f}%", help="Mean Absolute Percentage Error")
    with col4:
        precision = 100 - metrics['MAPE']
        st.metric("Accuracy", f"{precision:.1f}%", help="Overall model accuracy")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ==========================================
    # BASELINE COMPARISON
    # ==========================================
    
    st.markdown("<h2 class='section-header'>Comparison with Traditional Methods</h2>", unsafe_allow_html=True)
    
    df_comparacion = st.session_state['df_comparacion']
    costos_unitarios = st.session_state['costos_unitarios']
    
    st.markdown("""
    <div class='methodology-box'>
        <h4>Traditional Baseline Methods</h4>
        <ul style='margin: 0; padding-left: 1.5rem; color: #475569;'>
            <li><strong>Last Week:</strong> Repeat sales from 7 days ago</li>
            <li><strong>4-Week Average:</strong> Rolling average of last 28 days</li>
            <li><strong>Same Weekday:</strong> Same day of week from previous week</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparative table
    df_display = df_comparacion.copy()
    df_display['MAE'] = df_display['MAE'].round(2)
    df_display['MAPE'] = df_display['MAPE'].round(1).astype(str) + '%'
    df_display['Costo Total'] = df_display['Costo Total'].apply(lambda x: f"${x:,.0f}")
    df_display['Costo/Día'] = df_display['Costo/Día'].apply(lambda x: f"${x:,.0f}")
    
    # Rename columns
    df_display.columns = ['Method', 'MAE', 'MAPE', 'Total Cost', 'Cost/Day']
    
    # Style function
    def highlight_ml(row):
        if row['Method'] == 'ML (XGBoost)':
            return ['background-color: #d1fae5; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df_display.style.apply(highlight_ml, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ==========================================
    # ECONOMIC IMPACT
    # ==========================================
    
    st.markdown("<h2 class='section-header'>Economic Impact Analysis</h2>", unsafe_allow_html=True)
    
    ahorro_info = st.session_state['ahorro_info']
    
    # Methodology explanation
    with st.expander("Methodology: Cost Calculation"):
        st.markdown(f"""
        <div class='methodology-box'>
            <h4>Cost Structure (Parsa et al., 2005)</h4>
            
            <strong>Unit Economics:</strong>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem; color: #475569;'>
                <li><strong>Average unit price:</strong> ${costos_unitarios['precio_unitario']:,.0f} COP</li>
                <li><strong>Waste cost (cu):</strong> ${costos_unitarios['costo_unitario']:,.0f} COP ({costos_unitarios['costo_ratio']*100:.0f}%)</li>
                <li><strong>Opportunity cost (co):</strong> ${costos_unitarios['margen_unitario']:,.0f} COP ({costos_unitarios['margen_ratio']*100:.0f}%)</li>
            </ul>
            
            <strong>Cost Model (Newsvendor Problem):</strong>
            <pre style='background: #f1f5f9; padding: 1rem; border-radius: 0.375rem; margin: 0.5rem 0;'>
If forecast > actual demand:
    Cost = cu × (forecast - actual)  [waste]

If forecast < actual demand:
    Cost = co × (actual - forecast)  [lost sales]
            </pre>
            
            <strong>References:</strong>
            <ul style='margin: 0; padding-left: 1.5rem; color: #475569;'>
                <li>Parsa, H.G., et al. (2005). "Why Restaurants Fail"</li>
                <li>Nahmias, S. (2011). "Perishable Inventory Theory"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Economic KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Daily Savings",
            f"${ahorro_info['ahorro_dia']:,.0f}",
            delta=f"{ahorro_info['mejora_costo']:.1f}% vs {ahorro_info['mejor_baseline']}",
            help=f"Cost reduction: ${ahorro_info['costo_baseline']:,.0f} → ${ahorro_info['costo_ml']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Monthly Savings",
            f"${ahorro_info['ahorro_mes']:,.0f}",
            help="Estimated savings over 30 days"
        )
    
    with col3:
        st.metric(
            "Annual Projection",
            f"${ahorro_info['ahorro_año']:,.0f}",
            help="Projected annual savings (365 days)"
        )
    
    # Cost comparison chart
    st.markdown("### Cost Comparison by Method")
    
    fig_costos = go.Figure()
    
    metodos = df_comparacion['Método'].tolist()
    costos = df_comparacion['Costo/Día'].tolist()
    
    colores = ['#94a3b8' if m != 'ML (XGBoost)' else '#10b981' for m in metodos]
    
    fig_costos.add_trace(go.Bar(
        x=metodos,
        y=costos,
        marker=dict(color=colores, line=dict(color='#64748b', width=1)),
        text=[f"${c:,.0f}" for c in costos],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Cost/Day: $%{y:,.0f}<extra></extra>'
    ))
    
    fig_costos.update_layout(
        yaxis_title="Average Cost per Day (COP)",
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
    )
    
    st.plotly_chart(fig_costos, use_container_width=True)
    
    # Summary
    st.markdown("### Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Error Reduction:**
        - MAE: **↓ {ahorro_info['mejora_mae']:.1f}%**
        - MAPE: **↓ {ahorro_info['mejora_mape']:.1f}%**
        """)
    
    with col2:
        st.markdown(f"""
        **Economic Impact:**
        - Daily cost: **↓ {ahorro_info['mejora_costo']:.1f}%**
        - Monthly savings: **${ahorro_info['ahorro_mes']:,.0f}**
        """)
    
    # Interpretation
    st.markdown(f"""
    <div class='interpretation-box'>
        <p><strong>Key Insight:</strong></p>
        <p>The ML model reduces forecast error cost by <strong>{ahorro_info['mejora_costo']:.1f}%</strong> 
        compared to the best traditional method ({ahorro_info['mejor_baseline']}), resulting in estimated 
        savings of <strong>${ahorro_info['ahorro_mes']:,.0f} COP per month</strong> for this product.</p>
        <p><strong>Impact drivers:</strong></p>
        <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
            <li>Reduced waste from overstock</li>
            <li>Minimized lost sales from stockouts</li>
            <li>Optimized purchasing cycles</li>
        </ul>
        <p><strong>Annual projection:</strong> ${ahorro_info['ahorro_año']:,.0f} COP</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ==========================================
    # TABS: FORECAST, EVALUATION, DATA
    # ==========================================
    
    tab1, tab2, tab3 = st.tabs(["Forecast", "Evaluation", "Export"])
    
    with tab1:
        st.markdown("<h3 class='section-header'>Future Forecast with 95% Confidence Intervals</h3>", unsafe_allow_html=True)
        
        # Chart
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=df_producto['fecha'],
            y=df_producto['cantidad_vendida_diaria'],
            mode='lines',
            name='Historical',
            line=dict(color='#3b82f6', width=2),
            opacity=0.6,
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Actual: %{y:.1f}<extra></extra>'
        ))
        
        # Upper confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=y_pred_future_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Lower confidence interval (with fill)
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=y_pred_future_lower,
            mode='lines',
            name='95% Confidence',
            line=dict(width=0),
            fillcolor='rgba(255, 127, 14, 0.15)',
            fill='tonexty',
            hovertemplate='<b>%{x|%b %d}</b><br>Lower: %{y:.1f}<extra></extra>'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=y_pred_future,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=8, color='#f59e0b', line=dict(width=2, color='white')),
            hovertemplate='<b>%{x|%b %d}</b><br>Forecast: %{y:.1f}<extra></extra>'
        ))
        
        # "Today" divider line
        last_historical_date = pd.Timestamp(df_producto['fecha'].max())
        
        fig.add_shape(
            type="line",
            x0=last_historical_date,
            x1=last_historical_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#ef4444", width=2, dash="dot")
        )
        
        fig.add_annotation(
            x=last_historical_date,
            y=1.05,
            yref="paper",
            text="Today",
            showarrow=False,
            font=dict(color="#ef4444", size=12, weight='bold')
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode='x unified',
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown("### Detailed Forecast")
        
        interval_width = y_pred_future_upper - y_pred_future_lower
        
        df_table = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Weekday': future_dates.strftime('%A'),
            'Lower Bound': y_pred_future_lower.round(1),
            'Forecast': y_pred_future.round(1),
            'Upper Bound': y_pred_future_upper.round(1),
            'Uncertainty': interval_width.round(1)
        })
        
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        # Interpretation guide
        with st.expander("How to Interpret Confidence Intervals"):
            st.markdown("""
            <div class='methodology-box'>
                <h4>95% Confidence Interval Interpretation</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #475569;'>
                    <li>There is a 95% probability that actual demand will fall between the lower and upper bounds</li>
                    <li><strong>Lower Bound (5%):</strong> Conservative estimate - minimum expected demand</li>
                    <li><strong>Forecast:</strong> Most likely outcome based on model prediction</li>
                    <li><strong>Upper Bound (95%):</strong> Optimistic estimate - maximum expected demand</li>
                </ul>
                
                <h4 style='margin-top: 1rem;'>Purchasing Recommendations</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #475569;'>
                    <li>If uncertainty > 50% of forecast: Purchase conservatively (closer to lower bound)</li>
                    <li>If uncertainty < 30% of forecast: Purchase near forecast value</li>
                    <li>Always consider shelf life and storage constraints</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3 class='section-header'>Test Set Evaluation</h3>", unsafe_allow_html=True)
        
        y_test = st.session_state['y_test']
        y_pred_test = st.session_state['y_pred_test']
        df_test = st.session_state['df_test']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Test performance chart
            fig_test = go.Figure()
            
            fig_test.add_trace(go.Scatter(
                x=df_test['fecha'],
                y=y_test,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x|%b %d}</b><br>Actual: %{y:.1f}<extra></extra>'
            ))
            
            fig_test.add_trace(go.Scatter(
                x=df_test['fecha'],
                y=y_pred_test,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#f59e0b', width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x|%b %d}</b><br>Predicted: %{y:.1f}<extra></extra>'
            ))
            
            fig_test.update_layout(
                xaxis_title="Date",
                yaxis_title="Units",
                hovermode='x unified',
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                title="Actual vs Predicted"
            )
            
            st.plotly_chart(fig_test, use_container_width=True)
        
        with col2:
            # Scatter plot
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=y_test,
                y=y_pred_test,
                mode='markers',
                marker=dict(size=8, opacity=0.6, color='#3b82f6'),
                hovertemplate='<b>Actual:</b> %{x:.1f}<br><b>Predicted:</b> %{y:.1f}<extra></extra>'
            ))
            
            # Perfect prediction line
            max_val = max(y_test.max(), y_pred_test.max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='#ef4444', dash='dash', width=2),
                name='Perfect Prediction'
            ))
            
            fig_scatter.update_layout(
                xaxis_title="Actual",
                yaxis_title="Predicted",
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                title="Prediction Correlation",
                xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Feature importance
        st.markdown("### Top 10 Most Important Features")
        
        importance_df = pd.DataFrame({
            'Feature': predictor.feature_names,
            'Importance': predictor.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig_imp = go.Figure()
        
        fig_imp.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Importance'],
            orientation='h',
            marker=dict(color='#3b82f6', line=dict(color='#1e40af', width=1)),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig_imp.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=False, categoryorder='total ascending')
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab3:
        st.markdown("<h3 class='section-header'>Export Predictions</h3>", unsafe_allow_html=True)
        
        # CSV download
        df_download = pd.DataFrame({
            'date': future_dates.strftime('%Y-%m-%d'),
            'weekday': future_dates.strftime('%A'),
            'product': producto_nombre,
            'forecast': y_pred_future.round(2),
            'lower_bound_95': y_pred_future_lower.round(2),
            'upper_bound_95': y_pred_future_upper.round(2),
            'uncertainty': (y_pred_future_upper - y_pred_future_lower).round(2)
        })
        
        csv = df_download.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "Download CSV",
            csv,
            f"forecast_{producto_nombre[:20].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
        
        st.markdown("### Data Preview")
        st.dataframe(df_download, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class='interpretation-box'>
        <p style='text-align: center; font-size: 1.1rem;'>
            Configure model parameters in the sidebar and click <strong>"Train & Forecast"</strong> to begin
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 0.875rem;'>", unsafe_allow_html=True)
st.markdown("Powered by XGBoost with Optuna hyperparameter optimization • 95% confidence intervals using quantile regression • Economic analysis based on Newsvendor model")
st.markdown("</div>", unsafe_allow_html=True)
