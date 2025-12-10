import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas - Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    h1, h2, h3 {
        color: white;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    """Carga el dataframe de inferencia"""
    try:
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return None

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo = joblib.load('models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        return None

def calcular_precio_competencia(amazon, decathlon, deporvillage, escenario):
    """Calcula el precio promedio de competencia según el escenario"""
    factor = 1.0
    if escenario == "Competencia -5%":
        factor = 0.95
    elif escenario == "Competencia +5%":
        factor = 1.05
    
    return ((amazon + decathlon + deporvillage) / 3) * factor

def aplicar_escenario(df_producto, ajuste_descuento, escenario_competencia):
    """Aplica el ajuste de descuento y escenario de competencia al dataframe"""
    df = df_producto.copy()
    
    # Aplicar ajuste de descuento al precio de venta
    factor_descuento = 1 + (ajuste_descuento / 100)
    df['precio_venta'] = df['precio_base'] * factor_descuento
    
    # Recalcular precio de competencia según escenario
    df['precio_competencia'] = df.apply(
        lambda row: calcular_precio_competencia(
            row['Amazon'], row['Decathlon'], row['Deporvillage'], escenario_competencia
        ), axis=1
    )
    
    # Calcular descuento porcentaje y ratio precio
    df['descuento_porcentaje'] = ((df['precio_base'] - df['precio_venta']) / df['precio_base'] * 100).clip(lower=0)
    df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']
    
    return df

def predecir_recursivo(modelo, df_producto):
    """Realiza predicciones recursivas día por día actualizando lags"""
    df = df_producto.copy().sort_values('fecha').reset_index(drop=True)
    predicciones = []
    
    # Obtener las columnas que el modelo espera
    feature_cols = [col for col in modelo.feature_names_in_ if col in df.columns]
    
    for i in range(len(df)):
        # Preparar datos para predicción
        X = df.iloc[[i]][feature_cols]
        
        # Realizar predicción
        pred = modelo.predict(X)[0]
        pred = max(0, pred)  # Asegurar que no sea negativo
        predicciones.append(pred)
        
        # Actualizar lags para la siguiente iteración (si no es el último día)
        if i < len(df) - 1:
            # Crear variables de lag si existen en el dataframe
            if 'unidades_vendidas_lag_1' in df.columns:
                # Desplazar lags: lag_7 <- lag_6, ..., lag_2 <- lag_1
                for lag in range(7, 1, -1):
                    if f'unidades_vendidas_lag_{lag}' in df.columns:
                        df.at[i+1, f'unidades_vendidas_lag_{lag}'] = df.at[i, f'unidades_vendidas_lag_{lag-1}']
                
                # Actualizar lag_1 con la predicción actual
                df.at[i+1, 'unidades_vendidas_lag_1'] = pred
            
            # Actualizar media móvil de 7 días
            if 'unidades_vendidas_ma7' in df.columns:
                # Obtener las últimas 7 predicciones (o menos si no hay suficientes)
                ultimas_preds = predicciones[-7:] if len(predicciones) >= 7 else predicciones
                df.at[i+1, 'unidades_vendidas_ma7'] = np.mean(ultimas_preds)
    
    df['unidades_predichas'] = predicciones
    df['ingresos_proyectados'] = df['unidades_predichas'] * df['precio_venta']
    
    return df

def crear_grafico_prediccion(df_resultado, producto_nombre):
    """Crea el gráfico de predicción diaria con Black Friday destacado"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Extraer día del mes para el eje X
    dias = df_resultado['fecha'].dt.day.values
    unidades = df_resultado['unidades_predichas'].values
    
    # Crear línea principal
    ax.plot(dias, unidades, linewidth=2.5, color='#667eea', marker='o', markersize=6, label='Predicción de Ventas')
    
    # Marcar el Black Friday (día 28)
    bf_idx = df_resultado[df_resultado['fecha'].dt.day == 28].index
    if len(bf_idx) > 0:
        bf_idx = bf_idx[0]
        bf_unidades = df_resultado.loc[bf_idx, 'unidades_predichas']
        
        # Línea vertical
        ax.axvline(x=28, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Black Friday')
        
        # Punto destacado
        ax.plot(28, bf_unidades, 'ro', markersize=12, zorder=5)
        
        # Anotación
        ax.annotate('🔥 Black Friday', 
                   xy=(28, bf_unidades), 
                   xytext=(28, bf_unidades * 1.15),
                   ha='center',
                   fontsize=12,
                   fontweight='bold',
                   color='red',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Configurar ejes
    ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicción de Ventas Diarias - {producto_nombre}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(1, 31))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Mejorar apariencia
    plt.tight_layout()
    
    return fig

def crear_tabla_detallada(df_resultado):
    """Crea la tabla detallada con formato mejorado"""
    # Mapeo de nombres de días
    dias_semana = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    
    tabla = df_resultado[['fecha', 'precio_venta', 'precio_competencia', 
                          'descuento_porcentaje', 'unidades_predichas', 'ingresos_proyectados']].copy()
    
    # Formatear fecha y día de la semana
    tabla['Fecha'] = tabla['fecha'].dt.strftime('%d/%m/%Y')
    tabla['Día'] = tabla['fecha'].dt.day_name().map(dias_semana)
    
    # Identificar Black Friday
    tabla['BF'] = tabla['fecha'].dt.day.apply(lambda x: '🔥' if x == 28 else '')
    
    # Formatear valores
    tabla['Precio Venta'] = tabla['precio_venta'].apply(lambda x: f"{x:.2f}€")
    tabla['Precio Competencia'] = tabla['precio_competencia'].apply(lambda x: f"{x:.2f}€")
    tabla['Descuento'] = tabla['descuento_porcentaje'].apply(lambda x: f"{x:.1f}%")
    tabla['Unidades'] = tabla['unidades_predichas'].apply(lambda x: f"{int(x):,}")
    tabla['Ingresos'] = tabla['ingresos_proyectados'].apply(lambda x: f"{x:,.2f}€")
    
    # Seleccionar columnas finales
    tabla_final = tabla[['BF', 'Fecha', 'Día', 'Precio Venta', 'Precio Competencia', 
                         'Descuento', 'Unidades', 'Ingresos']]
    
    return tabla_final

# ==========================
# INTERFAZ PRINCIPAL
# ==========================

# Título principal
st.markdown("<h1 style='text-align: center;'>📊 Simulador de Ventas - Noviembre 2025</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Predicción inteligente con Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Cargar datos y modelo
df_inferencia = cargar_datos()
modelo = cargar_modelo()

if df_inferencia is None or modelo is None:
    st.stop()

# ==========================
# SIDEBAR - CONTROLES
# ==========================

st.sidebar.markdown("## 🎛️ Controles de Simulación")
st.sidebar.markdown("---")

# Obtener lista de productos únicos
productos = sorted(df_inferencia['nombre'].unique())

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "🛍️ Selecciona un Producto",
    productos,
    index=0
)

# Slider de descuento
ajuste_descuento = st.sidebar.slider(
    "💰 Ajuste de Descuento",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    help="Ajusta el descuento sobre el precio base"
)

# Selector de escenario de competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia")
escenario_competencia = st.sidebar.radio(
    "",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    help="Simula cambios en los precios de la competencia"
)

st.sidebar.markdown("---")

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.info("""
**Instrucciones:**
1. Selecciona un producto
2. Ajusta el descuento deseado
3. Elige el escenario de competencia
4. Haz clic en 'Simular Ventas'

**Nota:** Las predicciones se actualizan recursivamente día a día.
""")

# ==========================
# ZONA PRINCIPAL
# ==========================

if simular:
    with st.spinner('⏳ Procesando predicciones recursivas...'):
        # Filtrar datos del producto seleccionado
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        if len(df_producto) == 0:
            st.error("❌ No hay datos disponibles para este producto")
            st.stop()
        
        # Aplicar escenario
        df_escenario = aplicar_escenario(df_producto, ajuste_descuento, escenario_competencia)
        
        # Realizar predicción recursiva
        df_resultado = predecir_recursivo(modelo, df_escenario)
        
        # ==========================
        # HEADER
        # ==========================
        st.markdown(f"<h2 style='text-align: center;'>📦 {producto_seleccionado}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: white;'>Proyección para Noviembre 2025 | Descuento: {ajuste_descuento}% | Escenario: {escenario_competencia}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # ==========================
        # KPIs DESTACADOS
        # ==========================
        col1, col2, col3, col4 = st.columns(4)
        
        unidades_totales = int(df_resultado['unidades_predichas'].sum())
        ingresos_totales = df_resultado['ingresos_proyectados'].sum()
        precio_promedio = df_resultado['precio_venta'].mean()
        descuento_promedio = df_resultado['descuento_porcentaje'].mean()
        
        with col1:
            st.metric(
                label="📦 Unidades Totales",
                value=f"{unidades_totales:,}",
                delta="Proyectadas"
            )
        
        with col2:
            st.metric(
                label="💰 Ingresos Proyectados",
                value=f"{ingresos_totales:,.2f}€",
                delta="Total mes"
            )
        
        with col3:
            st.metric(
                label="🏷️ Precio Promedio",
                value=f"{precio_promedio:.2f}€",
                delta="Por unidad"
            )
        
        with col4:
            st.metric(
                label="🎯 Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta="Aplicado"
            )
        
        st.markdown("---")
        
        # ==========================
        # GRÁFICO DE PREDICCIÓN
        # ==========================
        st.markdown("### 📈 Predicción de Ventas Diarias")
        fig = crear_grafico_prediccion(df_resultado, producto_seleccionado)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # ==========================
        # TABLA DETALLADA
        # ==========================
        st.markdown("### 📋 Detalle Diario de Predicciones")
        tabla_detallada = crear_tabla_detallada(df_resultado)
        st.dataframe(tabla_detallada, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ==========================
        # COMPARATIVA DE ESCENARIOS
        # ==========================
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")
        st.markdown("*Comparación manteniendo el mismo descuento seleccionado*")
        
        col1, col2, col3 = st.columns(3)
        
        escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        resultados_escenarios = {}
        
        for esc in escenarios:
            df_esc = aplicar_escenario(df_producto, ajuste_descuento, esc)
            df_pred_esc = predecir_recursivo(modelo, df_esc)
            resultados_escenarios[esc] = {
                'unidades': int(df_pred_esc['unidades_predichas'].sum()),
                'ingresos': df_pred_esc['ingresos_proyectados'].sum()
            }
        
        with col1:
            st.markdown("#### 🔵 Sin Cambios")
            st.metric(
                "Unidades",
                f"{resultados_escenarios['Actual (0%)']['unidades']:,}"
            )
            st.metric(
                "Ingresos",
                f"{resultados_escenarios['Actual (0%)']['ingresos']:,.2f}€"
            )
        
        with col2:
            st.markdown("#### 🟢 Competencia -5%")
            delta_unidades = resultados_escenarios['Competencia -5%']['unidades'] - resultados_escenarios['Actual (0%)']['unidades']
            delta_ingresos = resultados_escenarios['Competencia -5%']['ingresos'] - resultados_escenarios['Actual (0%)']['ingresos']
            
            st.metric(
                "Unidades",
                f"{resultados_escenarios['Competencia -5%']['unidades']:,}",
                delta=f"{delta_unidades:+,}"
            )
            st.metric(
                "Ingresos",
                f"{resultados_escenarios['Competencia -5%']['ingresos']:,.2f}€",
                delta=f"{delta_ingresos:+,.2f}€"
            )
        
        with col3:
            st.markdown("#### 🔴 Competencia +5%")
            delta_unidades = resultados_escenarios['Competencia +5%']['unidades'] - resultados_escenarios['Actual (0%)']['unidades']
            delta_ingresos = resultados_escenarios['Competencia +5%']['ingresos'] - resultados_escenarios['Actual (0%)']['ingresos']
            
            st.metric(
                "Unidades",
                f"{resultados_escenarios['Competencia +5%']['unidades']:,}",
                delta=f"{delta_unidades:+,}"
            )
            st.metric(
                "Ingresos",
                f"{resultados_escenarios['Competencia +5%']['ingresos']:,.2f}€",
                delta=f"{delta_ingresos:+,.2f}€"
            )
        
        st.markdown("---")
        st.success("✅ Simulación completada exitosamente")

else:
    # Mensaje inicial
    st.markdown("<div style='text-align: center; padding: 50px;'>", unsafe_allow_html=True)
    st.markdown("### 👈 Configura los parámetros en el panel lateral")
    st.markdown("### 🚀 Haz clic en 'Simular Ventas' para comenzar")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Mostrar información general
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **📦 Productos Disponibles**
        
        Total: {len(productos)} productos
        """)
    
    with col2:
        st.info(f"""
        **📅 Período de Análisis**
        
        Noviembre 2025 (30 días)
        """)
    
    with col3:
        st.info(f"""
        **🤖 Modelo ML**
        
        HistGradientBoostingRegressor
        """)
