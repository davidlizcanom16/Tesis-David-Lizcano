"""
Entrenador de modelos con Optuna + Intervalos de Confianza
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBoostPredictor:
    """Predictor con XGBoost, tuning automático e intervalos de confianza"""
    
    def __init__(self, n_trials=20, random_state=42, confidence_level=0.95):
        self.n_trials = n_trials
        self.random_state = random_state
        self.confidence_level = confidence_level
        self.model = None
        self.model_lower = None  # Cuantil inferior
        self.model_upper = None  # Cuantil superior
        self.best_params = None
        self.feature_names = None
        
        # Calcular quantiles para intervalos
        alpha = 1 - confidence_level
        self.quantile_lower = alpha / 2
        self.quantile_upper = 1 - (alpha / 2)
        
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimizar hiperparámetros con Optuna"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            
            return mae
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        return study.best_params
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Entrenar modelo + intervalos de confianza"""
        
        # Si hay validación, hacer tuning
        if X_val is not None and y_val is not None:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            
            # Concatenar train + val para entrenamiento final
            X_full = pd.concat([X_train, X_val])
            y_full = pd.concat([y_train, y_val])
        else:
            # Usar parámetros por defecto
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            X_full = X_train
            y_full = y_train
        
        # 1. Entrenar modelo principal (predicción puntual)
        self.model = xgb.XGBRegressor(**self.best_params, verbosity=0)
        self.model.fit(X_full, y_full)
        
        # 2. Entrenar modelos para intervalos (quantile regression)
        params_quantile = self.best_params.copy()
        params_quantile['n_estimators'] = max(50, params_quantile['n_estimators'] // 2)  # Más rápido
        
        # Cuantil inferior
        self.model_lower = xgb.XGBRegressor(
            **params_quantile,
            objective='reg:quantileerror',
            quantile_alpha=self.quantile_lower,
            verbosity=0
        )
        self.model_lower.fit(X_full, y_full)
        
        # Cuantil superior
        self.model_upper = xgb.XGBRegressor(
            **params_quantile,
            objective='reg:quantileerror',
            quantile_alpha=self.quantile_upper,
            verbosity=0
        )
        self.model_upper.fit(X_full, y_full)
        
        self.feature_names = X_full.columns.tolist()
        
        return self.model
    
    def predict(self, X, return_intervals=False):
        """Hacer predicción con o sin intervalos"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Predicción puntual
        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)
        
        if not return_intervals:
            return predictions
        
        # Intervalos de confianza
        pred_lower = self.model_lower.predict(X)
        pred_upper = self.model_upper.predict(X)
        
        # Asegurar que lower <= pred <= upper
        pred_lower = np.maximum(pred_lower, 0)
        pred_upper = np.maximum(pred_upper, predictions)
        pred_lower = np.minimum(pred_lower, predictions)
        
        return predictions, pred_lower, pred_upper
    
    def save(self, filepath):
        """Guardar modelo"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_lower': self.model_lower,
            'model_upper': self.model_upper,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'confidence_level': self.confidence_level
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """Cargar modelo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.model_lower = model_data.get('model_lower')
        predictor.model_upper = model_data.get('model_upper')
        predictor.best_params = model_data['best_params']
        predictor.feature_names = model_data['feature_names']
        predictor.random_state = model_data['random_state']
        predictor.confidence_level = model_data.get('confidence_level', 0.95)
        
        return predictor

def calculate_metrics(y_true, y_pred):
    """Calcular métricas de evaluación"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def generate_alerts(predictions, pred_lower, pred_upper, historical_mean, historical_std):
    """Generar alertas inteligentes"""
    alerts = []
    
    # 1. Demanda inusualmente alta
    high_threshold = historical_mean + 2 * historical_std
    high_demand_days = predictions > high_threshold
    
    if high_demand_days.any():
        n_days = high_demand_days.sum()
        max_pred = predictions[high_demand_days].max()
        alerts.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': 'Demanda Alta Esperada',
            'message': f'{n_days} día(s) con demanda inusualmente alta (hasta {max_pred:.0f} unidades, +{((max_pred/historical_mean - 1)*100):.0f}% vs promedio)',
            'days': np.where(high_demand_days)[0] + 1
        })
    
    # 2. Demanda inusualmente baja
    low_threshold = max(0, historical_mean - 2 * historical_std)
    low_demand_days = predictions < low_threshold
    
    if low_demand_days.any():
        n_days = low_demand_days.sum()
        min_pred = predictions[low_demand_days].min()
        alerts.append({
            'type': 'info',
            'icon': '📉',
            'title': 'Demanda Baja Esperada',
            'message': f'{n_days} día(s) con demanda baja (mínimo {min_pred:.0f} unidades, -{((1 - min_pred/historical_mean)*100):.0f}% vs promedio)',
            'days': np.where(low_demand_days)[0] + 1
        })
    
    # 3. Alta incertidumbre
    uncertainty = pred_upper - pred_lower
    uncertainty_ratio = uncertainty / predictions
    high_uncertainty_days = uncertainty_ratio > 0.5  # Más de 50% de incertidumbre
    
    if high_uncertainty_days.any():
        n_days = high_uncertainty_days.sum()
        alerts.append({
            'type': 'info',
            'icon': '🔮',
            'title': 'Alta Incertidumbre',
            'message': f'{n_days} día(s) con alta incertidumbre en la predicción. Considera comprar de manera conservadora.',
            'days': np.where(high_uncertainty_days)[0] + 1
        })
    
    # 4. Tendencia creciente
    if len(predictions) >= 7:
        first_half_mean = predictions[:len(predictions)//2].mean()
        second_half_mean = predictions[len(predictions)//2:].mean()
        
        if second_half_mean > first_half_mean * 1.2:  # 20% más
            alerts.append({
                'type': 'success',
                'icon': '📈',
                'title': 'Tendencia Creciente',
                'message': f'La demanda muestra tendencia creciente (+{((second_half_mean/first_half_mean - 1)*100):.0f}% en segunda mitad)',
                'days': None
            })
        elif second_half_mean < first_half_mean * 0.8:  # 20% menos
            alerts.append({
                'type': 'info',
                'icon': '📉',
                'title': 'Tendencia Decreciente',
                'message': f'La demanda muestra tendencia decreciente (-{((1 - second_half_mean/first_half_mean)*100):.0f}% en segunda mitad)',
                'days': None
            })
    
    # 5. Fin de semana vs días laborales
    # Asumiendo que tenemos información de días de la semana
    # (Esto se puede mejorar con features temporales)
    
    return alerts
# ==========================================
# ANÁLISIS ECONÓMICO
# ==========================================

def estimar_costos_unitarios(df, precio_col='valor_total_diario', cantidad_col='cantidad_vendida_diaria'):
    """
    Estimar precio unitario promedio y estructura de costos
    
    Args:
        df: DataFrame con datos del producto
        precio_col: Columna con valor total diario
        cantidad_col: Columna con cantidades vendidas
    
    Returns:
        dict con precio_unitario, costo_unitario, margen_unitario
    """
    # Calcular precio unitario promedio
    precio_unitario = df[precio_col].sum() / df[cantidad_col].sum()
    
    # Aplicar estructura de costos de literatura (Parsa et al., 2005)
    # Restaurantes premium: 35% costo alimentos, 65% margen bruto
    costo_ratio = 0.35
    margen_ratio = 0.65
    
    costo_unitario = precio_unitario * costo_ratio
    margen_unitario = precio_unitario * margen_ratio
    
    return {
        'precio_unitario': precio_unitario,
        'costo_unitario': costo_unitario,  # cu: costo de desperdicio
        'margen_unitario': margen_unitario,  # co: costo de oportunidad
        'costo_ratio': costo_ratio,
        'margen_ratio': margen_ratio
    }


def calcular_costo_error(y_true, y_pred, cu, co):
    """
    Calcular costo total de error de predicción (Newsvendor Problem)
    
    Args:
        y_true: Demanda real (array)
        y_pred: Predicción (array)
        cu: Costo unitario de desperdicio (overage cost)
        co: Costo de oportunidad (underage cost)
    
    Returns:
        dict con costo_total, costo_promedio_dia, breakdown
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Costo por día
    costos_diarios = []
    excesos = []
    faltantes = []
    
    for real, pred in zip(y_true, y_pred):
        if pred > real:
            # Exceso: desperdicio
            exceso = pred - real
            costo = cu * exceso
            excesos.append(exceso)
            faltantes.append(0)
        else:
            # Faltante: pérdida de venta
            faltante = real - pred
            costo = co * faltante
            excesos.append(0)
            faltantes.append(faltante)
        
        costos_diarios.append(costo)
    
    costo_total = sum(costos_diarios)
    costo_promedio = np.mean(costos_diarios)
    
    return {
        'costo_total': costo_total,
        'costo_promedio_dia': costo_promedio,
        'dias_exceso': sum(1 for e in excesos if e > 0),
        'dias_faltante': sum(1 for f in faltantes if f > 0),
        'exceso_total': sum(excesos),
        'faltante_total': sum(faltantes)
    }


def generar_baselines(df, target_col='cantidad_vendida_diaria'):
    """
    Generar predicciones usando métodos baseline tradicionales
    
    Args:
        df: DataFrame ordenado por fecha
        target_col: Columna con cantidad vendida
    
    Returns:
        DataFrame con predicciones de cada baseline
    """
    df = df.copy()
    
    # Baseline 1: Naive (semana anterior)
    df['pred_naive'] = df[target_col].shift(7)
    
    # Baseline 2: Moving Average (4 semanas)
    df['pred_ma'] = df[target_col].shift(1).rolling(window=28, min_periods=7).mean()
    
    # Baseline 3: Seasonal Naive (mismo día de la semana anterior)
    df['dia_semana'] = pd.to_datetime(df['fecha']).dt.dayofweek
    df['pred_seasonal'] = df.groupby('dia_semana')[target_col].shift(1)
    
    return df


def comparar_metodos(df_test, y_pred_ml, costos_unitarios):
    """
    Comparar performance económico de ML vs baselines
    
    Args:
        df_test: DataFrame de test con baselines generados
        y_pred_ml: Predicciones del modelo ML
        costos_unitarios: Dict con cu y co
    
    Returns:
        DataFrame con comparación de métodos
    """
    cu = costos_unitarios['costo_unitario']
    co = costos_unitarios['margen_unitario']
    
    y_true = df_test['cantidad_vendida_diaria'].values
    
    resultados = []
    
    # Método 1: Naive
    mask_naive = ~df_test['pred_naive'].isna()
    if mask_naive.sum() > 0:
        mae_naive = mean_absolute_error(y_true[mask_naive], df_test['pred_naive'][mask_naive])
        mape_naive = np.mean(np.abs((y_true[mask_naive] - df_test['pred_naive'][mask_naive]) / y_true[mask_naive])) * 100
        costo_naive = calcular_costo_error(y_true[mask_naive], df_test['pred_naive'][mask_naive], cu, co)
        
        resultados.append({
            'Método': 'Semana Anterior',
            'MAE': mae_naive,
            'MAPE': mape_naive,
            'Costo Total': costo_naive['costo_total'],
            'Costo/Día': costo_naive['costo_promedio_dia']
        })
    
    # Método 2: Moving Average
    mask_ma = ~df_test['pred_ma'].isna()
    if mask_ma.sum() > 0:
        mae_ma = mean_absolute_error(y_true[mask_ma], df_test['pred_ma'][mask_ma])
        mape_ma = np.mean(np.abs((y_true[mask_ma] - df_test['pred_ma'][mask_ma]) / y_true[mask_ma])) * 100
        costo_ma = calcular_costo_error(y_true[mask_ma], df_test['pred_ma'][mask_ma], cu, co)
        
        resultados.append({
            'Método': 'Promedio 4 Semanas',
            'MAE': mae_ma,
            'MAPE': mape_ma,
            'Costo Total': costo_ma['costo_total'],
            'Costo/Día': costo_ma['costo_promedio_dia']
        })
    
    # Método 3: Seasonal Naive
    mask_seasonal = ~df_test['pred_seasonal'].isna()
    if mask_seasonal.sum() > 0:
        mae_seasonal = mean_absolute_error(y_true[mask_seasonal], df_test['pred_seasonal'][mask_seasonal])
        mape_seasonal = np.mean(np.abs((y_true[mask_seasonal] - df_test['pred_seasonal'][mask_seasonal]) / y_true[mask_seasonal])) * 100
        costo_seasonal = calcular_costo_error(y_true[mask_seasonal], df_test['pred_seasonal'][mask_seasonal], cu, co)
        
        resultados.append({
            'Método': 'Mismo Día Sem. Ant.',
            'MAE': mae_seasonal,
            'MAPE': mape_seasonal,
            'Costo Total': costo_seasonal['costo_total'],
            'Costo/Día': costo_seasonal['costo_promedio_dia']
        })
    
    # Método 4: ML (XGBoost)
    mae_ml = mean_absolute_error(y_true, y_pred_ml)
    mape_ml = np.mean(np.abs((y_true - y_pred_ml) / y_true)) * 100
    costo_ml = calcular_costo_error(y_true, y_pred_ml, cu, co)
    
    resultados.append({
        'Método': 'ML (XGBoost)',
        'MAE': mae_ml,
        'MAPE': mape_ml,
        'Costo Total': costo_ml['costo_total'],
        'Costo/Día': costo_ml['costo_promedio_dia']
    })
    
    df_comparacion = pd.DataFrame(resultados)
    
    return df_comparacion


def calcular_ahorro_economico(df_comparacion, dias_mes=30, dias_año=365):
    """
    Calcular ahorro económico del ML vs mejor baseline
    
    Args:
        df_comparacion: DataFrame con comparación de métodos
        dias_mes: Días promedio por mes (default: 30)
        dias_año: Días por año (default: 365)
    
    Returns:
        dict con ahorros y mejoras
    """
    # Encontrar mejor baseline (menor costo)
    baselines = df_comparacion[df_comparacion['Método'] != 'ML (XGBoost)']
    mejor_baseline = baselines.loc[baselines['Costo/Día'].idxmin()]
    
    ml_row = df_comparacion[df_comparacion['Método'] == 'ML (XGBoost)'].iloc[0]
    
    # Calcular ahorros
    ahorro_dia = mejor_baseline['Costo/Día'] - ml_row['Costo/Día']
    ahorro_mes = ahorro_dia * dias_mes
    ahorro_año = ahorro_dia * dias_año
    
    # Calcular mejoras porcentuales
    mejora_mae = ((mejor_baseline['MAE'] - ml_row['MAE']) / mejor_baseline['MAE']) * 100
    mejora_mape = ((mejor_baseline['MAPE'] - ml_row['MAPE']) / mejor_baseline['MAPE']) * 100
    mejora_costo = ((mejor_baseline['Costo/Día'] - ml_row['Costo/Día']) / mejor_baseline['Costo/Día']) * 100
    
    return {
        'mejor_baseline': mejor_baseline['Método'],
        'ahorro_dia': ahorro_dia,
        'ahorro_mes': ahorro_mes,
        'ahorro_año': ahorro_año,
        'mejora_mae': mejora_mae,
        'mejora_mape': mejora_mape,
        'mejora_costo': mejora_costo,
        'costo_baseline': mejor_baseline['Costo/Día'],
        'costo_ml': ml_row['Costo/Día']
    }
