"""
Dashboard Completo - Análisis de Compras Públicas Ecuador
Endpoint API: https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis
Ejecutar con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ==================== CONFIGURACIÓN DE PÁGINA ====================
st.set_page_config(
    page_title="Compras Públicas Ecuador - EDA Completo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ESTILOS CSS ====================


# ==================== CONSTANTES ====================
API_BASE_URL = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis"
MAX_RETRIES = 3
RETRY_DELAY = 5  # segundos entre reintentos

# ==================== FUNCIONES DE EXTRACCIÓN ====================

@st.cache_data(ttl=3600)
def fetch_data_from_api(year=None, region=None, internal_type=None):
    """
    Obtiene datos desde la API de Compras Públicas con reintentos automáticos.
    
    Parámetros:
        year (str): Año específico o 'all' para todos
        region (str): Provincia/región específica
        internal_type (str): Tipo de contratación
    """
    params = {}
    if year and year != 'Todos':
        params['year'] = year
    if region and region != 'Todas':
        params['region'] = region
    if internal_type and internal_type != 'Todos':
        params['type'] = internal_type
    
    # Intentar con reintentos
    for intento in range(MAX_RETRIES):
        try:
            with st.spinner(f'Consultando API... (Intento {intento + 1}/{MAX_RETRIES})'):
                response = requests.get(API_BASE_URL, params=params, timeout=60)
                
                # Si es 429, esperar y reintentar
                if response.status_code == 429:
                    if intento < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (intento + 1)  # Backoff exponencial
                        st.warning(f"⏳ Límite de tasa alcanzado. Esperando {wait_time} segundos antes de reintentar...")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.error("❌ Se alcanzó el límite máximo de reintentos. La API está temporalmente no disponible.")
                        st.info("💡 **Sugerencias:**\n- Espera unos minutos antes de volver a intentar\n- Usa la opción de carga desde CSV")
                        return None
                
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    return data
                    
        except requests.exceptions.Timeout:
            st.error(f"⏱️ Tiempo de espera agotado en el intento {intento + 1}")
            if intento < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            if intento < MAX_RETRIES - 1:
                st.warning(f"⚠️ Error en intento {intento + 1}: {e}. Reintentando...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                st.error(f"❌ Error en petición a la API después de {MAX_RETRIES} intentos: {e}")
                st.info("💡 **Alternativa:** Usa la opción de carga desde archivo CSV")
                return None
    
    return None

# ==================== FUNCIONES DE LIMPIEZA ====================

@st.cache_data
def clean_dataframe(df):
    """
    Limpia y normaliza el DataFrame según los requisitos del proyecto.
    """
    df_clean = df.copy()
    
    # 1. ESTANDARIZACIÓN DE NOMBRES DE COLUMNAS
    column_mapping = {
        'provincia': 'region',
        'tipo_contratacion': 'internal_type',
        'monto_total': 'total',
        'numero_contratos': 'contracts'
    }
    df_clean.rename(columns=column_mapping, inplace=True)
    
    # 2. VERIFICAR Y CONVERTIR TIPOS DE DATOS
    # Convertir fecha
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['month_name'] = df_clean['date'].dt.strftime('%B')
    
    # Convertir tipos numéricos
    numeric_columns = ['total', 'contracts']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 3. TRATAR VALORES NULOS EN CAMPOS CRÍTICOS
    critical_fields = ['total', 'internal_type']
    for field in critical_fields:
        if field in df_clean.columns:
            # Eliminar filas con nulos en campos críticos
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[field])
            removed_nulls = initial_count - len(df_clean)
            if removed_nulls > 0:
                st.info(f"🔧 Se eliminaron {removed_nulls} registros con nulos en '{field}'")
    
    # 4. ELIMINAR DUPLICADOS
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    
    if removed_duplicates > 0:
        st.info(f"🔧 Se eliminaron {removed_duplicates} registros duplicados")
    
    # 5. NORMALIZAR TEXTO
    text_columns = ['region', 'internal_type']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
            # Reemplazar valores NaN string por algo más descriptivo
            df_clean[col] = df_clean[col].replace('NAN', 'NO ESPECIFICADO')
    
    return df_clean

# ==================== FUNCIONES DE ANÁLISIS ====================

def calcular_kpis(df):
    """Calcula indicadores clave de desempeño."""
    total_registros = len(df)
    
    if 'total' in df.columns:
        monto_total = df['total'].sum()
        promedio_por_registro = df['total'].mean()
        monto_maximo = df['total'].max()
        monto_minimo = df['total'].min()
    else:
        monto_total = promedio_por_registro = monto_maximo = monto_minimo = 0
    
    if 'contracts' in df.columns:
        total_contratos = df['contracts'].sum()
    else:
        total_contratos = total_registros
    
    return {
        'total_registros': total_registros,
        'monto_total': monto_total,
        'promedio_por_registro': promedio_por_registro,
        'monto_maximo': monto_maximo,
        'monto_minimo': monto_minimo,
        'total_contratos': total_contratos
    }

def mostrar_kpis(kpis):
    """Muestra KPIs en formato de métricas."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total de Registros", f"{kpis['total_registros']:,}")
        st.metric("💰 Monto Total", f"${kpis['monto_total']:,.2f}")
    
    with col2:
        st.metric("📈 Promedio por Registro", f"${kpis['promedio_por_registro']:,.2f}")
        st.metric("🔝 Monto Máximo", f"${kpis['monto_maximo']:,.2f}")
    
    with col3:
        st.metric("🔽 Monto Mínimo", f"${kpis['monto_minimo']:,.2f}")
        st.metric("📋 Total Contratos", f"{kpis['total_contratos']:,}")

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def viz_barras_tipo(df):
    """a) Total de Montos por Tipo de Contratación"""
    if 'internal_type' not in df.columns or 'total' not in df.columns:
        st.warning("⚠️ Columnas necesarias no disponibles")
        return
    
    datos_tipo = df.groupby('internal_type')['total'].sum().sort_values(ascending=False)
    
    fig = px.bar(
        x=datos_tipo.values,
        y=datos_tipo.index,
        orientation='h',
        title='a) Total de Montos por Tipo de Contratación',
        labels={'x': 'Monto Total ($)', 'y': 'Tipo de Contratación'},
        color=datos_tipo.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación con validación
    if len(datos_tipo) >= 2:
        st.markdown(f"""
        <div class="interpretation">
        <strong>📊 Interpretación:</strong> El tipo de contratación <strong>{datos_tipo.index[0]}</strong> 
        representa el mayor monto total con <strong>${datos_tipo.values[0]:,.2f}</strong>, 
        seguido por <strong>{datos_tipo.index[1]}</strong> con <strong>${datos_tipo.values[1]:,.2f}</strong>.
        Esto indica que estas modalidades son las más utilizadas en términos de inversión pública.
        </div>
        """, unsafe_allow_html=True)
    elif len(datos_tipo) == 1:
        st.markdown(f"""
        <div class="interpretation">
        <strong>📊 Interpretación:</strong> El tipo de contratación <strong>{datos_tipo.index[0]}</strong> 
        representa el total con <strong>${datos_tipo.values[0]:,.2f}</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No hay datos suficientes para interpretar")

def viz_linea_mensual(df):
    """b) Evolución Mensual de Montos Totales"""
    if 'date' not in df.columns or 'total' not in df.columns:
        st.warning("⚠️ Columnas necesarias no disponibles")
        return
    
    # Agrupar por mes
    df_mensual = df.set_index('date').resample('M')['total'].sum().reset_index()
    
    fig = px.line(
        df_mensual,
        x='date',
        y='total',
        title='b) Evolución Mensual de Montos Totales',
        labels={'date': 'Fecha', 'total': 'Monto Total ($)'},
        markers=True
    )
    fig.update_traces(line_color='#0068c9', line_width=3)
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación
    max_mes = df_mensual.loc[df_mensual['total'].idxmax(), 'date'].strftime('%B %Y')
    max_monto = df_mensual['total'].max()
    
    st.markdown(f"""
    <div class="interpretation">
    <strong>📈 Interpretación:</strong> Se observa una tendencia en la contratación pública 
    a lo largo del tiempo. El pico máximo ocurrió en <strong>{max_mes}</strong> con un monto 
    de <strong>${max_monto:,.2f}</strong>. Las variaciones mensuales pueden estar relacionadas 
    con ciclos presupuestarios y planificación gubernamental.
    </div>
    """, unsafe_allow_html=True)

def viz_barras_apiladas(df):
    """c) Total de Montos por Tipo de Contratación por Mes"""
    if 'month' not in df.columns or 'internal_type' not in df.columns or 'total' not in df.columns:
        st.warning("⚠️ Columnas necesarias no disponibles")
        return
    
    datos_mes_tipo = df.groupby(['month', 'internal_type'])['total'].sum().reset_index()
    
    fig = px.bar(
        datos_mes_tipo,
        x='month',
        y='total',
        color='internal_type',
        title='c) Total de Montos por Tipo de Contratación por Mes',
        labels={'month': 'Mes', 'total': 'Monto Total ($)', 'internal_type': 'Tipo'},
        barmode='stack'
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación
    st.markdown("""
    <div class="interpretation">
    <strong>📊 Interpretación:</strong> Este gráfico permite identificar la distribución 
    de tipos de contratación mes a mes. Se pueden observar patrones estacionales y 
    variaciones en las preferencias de modalidades de contratación a lo largo del año.
    </div>
    """, unsafe_allow_html=True)

def viz_pastel_proporcion(df):
    """d) Proporción de Contratos por Tipo de Contratación"""
    if 'internal_type' not in df.columns or 'contracts' not in df.columns:
        st.warning("⚠️ Columnas necesarias no disponibles")
        return
    
    datos_proporcion = df.groupby('internal_type')['contracts'].sum()
    
    fig = px.pie(
        values=datos_proporcion.values,
        names=datos_proporcion.index,
        title='d) Proporción de Contratos por Tipo de Contratación',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación con validación
    total = datos_proporcion.sum()
    if total > 0 and len(datos_proporcion) > 0:
        tipo_principal = datos_proporcion.idxmax()
        porcentaje = (datos_proporcion.max() / total * 100)
        
        st.markdown(f"""
        <div class="interpretation">
        <strong>📊 Interpretación:</strong> <strong>{tipo_principal}</strong> representa 
        el <strong>{porcentaje:.1f}%</strong> del total de contratos, siendo la modalidad 
        más utilizada en términos de cantidad de procesos de contratación.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No hay datos suficientes para interpretar")

def viz_dispersion_monto_contratos(df):
    """6. Relación entre Monto Total y Cantidad de Contratos"""
    if 'contracts' not in df.columns or 'total' not in df.columns or 'internal_type' not in df.columns:
        st.warning("⚠️ Columnas necesarias no disponibles")
        return
    
    fig = px.scatter(
        df,
        x='contracts',
        y='total',
        color='internal_type',
        title='Dispersión: Monto Total vs. Cantidad de Contratos',
        labels={'contracts': 'Cantidad de Contratos', 'total': 'Monto Total ($)', 'internal_type': 'Tipo'},
        hover_data=['region'] if 'region' in df.columns else None,
        trendline='ols'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calcular correlación
    correlacion = df[['contracts', 'total']].corr().iloc[0, 1]
    
    # Interpretación
    if correlacion > 0.7:
        tipo_corr = "positiva fuerte"
    elif correlacion > 0.3:
        tipo_corr = "positiva moderada"
    elif correlacion > -0.3:
        tipo_corr = "débil"
    elif correlacion > -0.7:
        tipo_corr = "negativa moderada"
    else:
        tipo_corr = "negativa fuerte"
    
    st.markdown(f"""
    <div class="interpretation">
    <strong>🔍 Interpretación:</strong> La correlación entre cantidad de contratos y monto total 
    es <strong>{tipo_corr}</strong> (r={correlacion:.3f}). {'Esto indica que a mayor número de contratos, tiende a haber un mayor monto total invertido.' if correlacion > 0.3 else 'La relación entre estas variables no es claramente lineal, sugiriendo que otros factores influyen en los montos.'}
    </div>
    """, unsafe_allow_html=True)

def viz_comparativa_tipos_mes(df):
    """7. Comparativa de Tipos de Contratación a lo largo del Año"""
    if 'month' not in df.columns or 'internal_type' not in df.columns or 'total' not in df.columns:
        st.warning("⚠️ Columnas necesarias no disponibles")
        return
    
    datos_serie = df.groupby(['month', 'internal_type'])['total'].sum().reset_index()
    
    fig = px.line(
        datos_serie,
        x='month',
        y='total',
        color='internal_type',
        title='Comparativa de Tipos de Contratación por Mes',
        labels={'month': 'Mes', 'total': 'Monto Total ($)', 'internal_type': 'Tipo'},
        markers=True
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.update_layout(hovermode='x unified', height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación
    st.markdown("""
    <div class="interpretation">
    <strong>📈 Interpretación:</strong> Este gráfico permite comparar el comportamiento 
    de cada tipo de contratación a lo largo del año. Se pueden identificar patrones 
    estacionales, períodos de mayor actividad y tendencias específicas para cada modalidad.
    </div>
    """, unsafe_allow_html=True)

def analisis_por_anios(df):
    """8. Análisis comparativo por años"""
    if 'year' not in df.columns:
        st.warning("⚠️ Columna 'year' no disponible")
        return
    
    st.subheader("📅 Análisis Comparativo por Años")
    
    # KPIs por año
    st.markdown("#### KPIs por Año")
    kpis_anuales = df.groupby('year').agg({
        'total': ['sum', 'mean', 'count'],
        'contracts': 'sum' if 'contracts' in df.columns else 'count'
    }).round(2)
    st.dataframe(kpis_anuales, use_container_width=True)
    
    # Gráfico: Tipo x Año (Barras apiladas)
    st.markdown("#### Montos por Tipo y Año")
    if 'internal_type' in df.columns:
        datos_tipo_anio = df.groupby(['year', 'internal_type'])['total'].sum().reset_index()
        
        fig1 = px.bar(
            datos_tipo_anio,
            x='year',
            y='total',
            color='internal_type',
            title='Montos por Tipo de Contratación y Año',
            labels={'year': 'Año', 'total': 'Monto Total ($)', 'internal_type': 'Tipo'},
            barmode='stack'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Evolución mensual comparada por año
    st.markdown("#### Evolución Mensual Comparada")
    if 'month' in df.columns:
        datos_mes_anio = df.groupby(['year', 'month'])['total'].sum().reset_index()
        
        fig2 = px.line(
            datos_mes_anio,
            x='month',
            y='total',
            color='year',
            title='Evolución Mensual por Año',
            labels={'month': 'Mes', 'total': 'Monto Total ($)', 'year': 'Año'},
            markers=True
        )
        fig2.update_xaxes(tickmode='linear', tick0=1, dtick=1)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap año x mes
    st.markdown("#### Mapa de Calor: Año × Mes")
    if 'month' in df.columns:
        pivot_data = df.pivot_table(values='total', index='year', columns='month', aggfunc='sum', fill_value=0)
        
        fig3 = px.imshow(
            pivot_data,
            labels=dict(x="Mes", y="Año", color="Monto Total ($)"),
            title="Mapa de Calor: Distribución de Montos por Año y Mes",
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Interpretación
    years = df['year'].unique()
    if len(years) >= 2:
        max_year = df.groupby('year')['total'].sum().idxmax()
        min_year = df.groupby('year')['total'].sum().idxmin()
        
        st.markdown(f"""
        <div class="interpretation">
        <strong>📊 Interpretación de Tendencias Anuales:</strong><br>
        • <strong>Año con mayor inversión:</strong> {max_year}<br>
        • <strong>Año con menor inversión:</strong> {min_year}<br>
        • Se pueden observar variaciones interanuales que pueden estar relacionadas con 
        cambios en políticas públicas, prioridades gubernamentales, ciclos económicos o 
        eventos específicos (ej: emergencias, elecciones, reformas administrativas).
        </div>
        """, unsafe_allow_html=True)

# ==================== INTERFAZ PRINCIPAL ====================

def main():
    regiones_seleccionadas = []
    # HEADER
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("### Compras Públicas Ecuador - Sistema SERCOP")
    st.markdown("**Endpoint API:** `get_analysis` | **Análisis Completo con Visualizaciones Interactivas**")
    st.markdown("---")
    
    # ==================== SIDEBAR: FILTROS ====================
    with st.sidebar:
        st.header("⚙️ Filtros de Análisis")
        st.markdown("Configure los parámetros para la consulta:")
        
        # Filtro de año
        year_option = st.selectbox(
            "📅 Seleccionar Año",
            options=['Todos', '2024', '2023', '2022', '2021', '2020'],
            index=0
        )
        
        # Filtro de provincia
        region_option = st.selectbox(
            "🗺️ Seleccionar Provincia",
            options=['Todas', 'PICHINCHA', 'GUAYAS', 'AZUAY', 'MANABI', 'LOS RIOS'],
            index=0
        )
        
        # Filtro de tipo
        type_option = st.selectbox(
            "📋 Tipo de Contratación",
            options=['Todos', 'Licitación', 'Menor Cuantía', 'Subasta Inversa', 'Contratación Directa'],
            index=0
        )
        
        st.markdown("---")
        
        # Opción de carga de datos
        data_source = st.radio(
            "📂 Fuente de Datos",
            options=['API', 'Archivo CSV'],
            index=0,
            help="API: Consulta en tiempo real | CSV: Carga archivo local sin límites de tasa"
        )
        
        if data_source == 'Archivo CSV':
            uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
            st.info("ℹ️ **Ventaja del CSV:** No hay límites de tasa de la API")
        else:
            st.warning("⚠️ **Nota:** La API tiene límites de tasa. Si falla, usa CSV o espera unos minutos.")
        
        st.markdown("---")
        cargar_btn = st.button("🔄 Cargar/Actualizar Datos", type="primary")
        
        # Botón para limpiar caché
        if st.button("🗑️ Limpiar Caché", help="Útil si hay errores de caché o datos antiguos"):
            st.cache_data.clear()
            st.success("✅ Caché limpiado exitosamente")
            st.rerun()
    
    # ==================== CARGA DE DATOS ====================
    
    if cargar_btn:
        if data_source == 'API':
            # Cargar desde API
            datos_raw = fetch_data_from_api(
                year=year_option if year_option != 'Todos' else None,
                region=region_option if region_option != 'Todas' else None,
                internal_type=type_option if type_option != 'Todos' else None
            )
            
            if datos_raw:
                df_raw = pd.DataFrame(datos_raw)
            else:
                st.error("❌ No se pudieron obtener datos de la API")
                return
        
        else:
            # Cargar desde CSV
            if uploaded_file is not None:
                df_raw = pd.read_csv(uploaded_file)
            else:
                st.warning("⚠️ Por favor, carga un archivo CSV")
                return
        
        # Validar estructura
        st.success(f"✅ Datos cargados exitosamente")
        
        with st.expander("🔍 Validación de Estructura de Datos"):
            st.markdown("**Primeras filas del dataset:**")
            st.dataframe(df_raw.head(), use_container_width=True)
            
            st.markdown("**Información del DataFrame:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filas", df_raw.shape[0])
                st.metric("Columnas", df_raw.shape[1])
            with col2:
                st.write("**Columnas disponibles:**")
                st.write(df_raw.columns.tolist())
            
            st.markdown("**Tipos de datos:**")
            df_types = pd.DataFrame({
                'Columna': df_raw.dtypes.index,
                'Tipo': [str(dtype) for dtype in df_raw.dtypes.values]
            })
            st.dataframe(df_types, use_container_width=True)
        
        # Limpieza de datos
        with st.spinner("🔧 Limpiando y preparando datos..."):
            df_clean = clean_dataframe(df_raw)
        
        st.success(f"✅ Datos procesados: {len(df_clean)} registros válidos")
        
        # Guardar en session_state
        st.session_state['df'] = df_clean
        st.session_state['filters'] = {
            'year': year_option,
            'region': region_option,
            'type': type_option
        }
    
    # ==================== ANÁLISIS Y VISUALIZACIÓN ====================
    
    if 'df' not in st.session_state:
        st.info("👈 **Instrucciones:** Configure los filtros en el panel lateral y presione 'Cargar/Actualizar Datos' para comenzar el análisis")
        
        with st.expander("📖 Guía de Uso"):
            st.markdown("""
            ### Cómo usar este dashboard:
            
            1. **Seleccione los filtros** en el panel lateral:
               - Año de análisis (o "Todos" para análisis multi-anual)
               - Provincia específica (o "Todas")
               - Tipo de contratación (o "Todos")
            
            2. **Elija la fuente de datos**:
               - **API**: Consulta en tiempo real al sistema SERCOP
               - **CSV**: Carga un archivo local sin límites de tasa
            
            3. **Presione "Cargar/Actualizar Datos"** para iniciar el análisis
            
            4. **Explore las visualizaciones** organizadas en pestañas:
               - Análisis Descriptivo
               - Visualizaciones Principales
               - Análisis de Correlaciones
               - Comparativa Temporal
               - Análisis por Años
            """)
        
        with st.expander("⚠️ Solución al Error 429 (Too Many Requests)"):
            st.markdown("""
            ### ¿Qué es el error 429?
            
            La API de Compras Públicas tiene límites de tasa para evitar sobrecarga. 
            Esto significa que solo permite un número limitado de peticiones por minuto.
            
            ### Soluciones:
            
            **Opción 1: Esperar y Reintentar (Automático)**
            - El sistema reintenta automáticamente con esperas progresivas
            - Espera 5, 10, 15 segundos entre intentos
            
            **Opción 2: Usar Archivo CSV (Recomendado)**
            1. Descarga datos desde el portal oficial: [Portal SERCOP](https://datosabiertos.compraspublicas.gob.ec/)
            2. Selecciona "Archivo CSV" en Fuente de Datos
            3. Carga el archivo descargado
            4. ¡Sin límites de tasa!
            
            **Opción 3: Filtros Más Específicos**
            - En lugar de "Todos", selecciona año y provincia específicos
            - Reduce la cantidad de datos solicitados
            
            **Opción 4: Esperar Unos Minutos**
            - Los límites se resetean después de un tiempo
            - Intenta de nuevo en 2-3 minutos
            """)
        
        with st.expander("📥 Generar CSV de Ejemplo"):
            st.markdown("""
            ### Formato esperado del CSV:
            
            El archivo CSV debe tener las siguientes columnas:
            - `date`: Fecha (formato: YYYY-MM-DD)
            - `total`: Monto total (numérico)
            - `contracts`: Cantidad de contratos (numérico)
            - `internal_type`: Tipo de contratación (texto)
            - `region`: Provincia/región (texto)
            - `year`: Año (numérico)
            - `month`: Mes (numérico 1-12)
            
            **Ejemplo de datos:**
            ```csv
            date,total,contracts,internal_type,region,year,month
            2023-01-15,150000.50,5,Licitación,PICHINCHA,2023,1
            2023-02-20,250000.75,8,Subasta Inversa,GUAYAS,2023,2
            2023-03-10,180000.00,3,Menor Cuantía,AZUAY,2023,3
            ```
            """)
        
        return
    
    df = st.session_state['df']
    filters = st.session_state['filters']
    
    # Mostrar filtros aplicados
    st.subheader("🔎 Filtros Aplicados")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Año:** {filters['year']}")
    with col2:
        st.info(f"**Provincia:** {filters['region']}")
    with col3:
        st.info(f"**Tipo:** {filters['type']}")
    
    st.markdown("---")
    
    # ==================== TABS PRINCIPALES ====================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Análisis Descriptivo",
        "📈 Visualizaciones Principales",
        "🔗 Correlaciones",
        "⏱️ Comparativa Temporal",
        "📅 Análisis por Años",
        "💾 Exportar Datos"
    ])
    
    # TAB 1: ANÁLISIS DESCRIPTIVO
    with tab1:
        st.header("4. Análisis Descriptivo")
        
        # KPIs
        kpis = calcular_kpis(df)
        mostrar_kpis(kpis)
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        st.subheader("📊 Estadísticas Descriptivas")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Interpretación
            st.markdown("""
            <div class="interpretation">
            <strong>📋 Hallazgos de las Estadísticas:</strong><br>
            • Revisar la <strong>desviación estándar</strong> indica la dispersión de los datos<br>
            • Los valores de <strong>máximo y mínimo</strong> muestran el rango de variación<br>
            • La <strong>mediana (50%)</strong> vs <strong>media</strong> indica sesgo en la distribución
            </div>
            """, unsafe_allow_html=True)
        
        # Conteos por categoría
        st.markdown("---")
        st.subheader("📋 Conteos por Categoría")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'internal_type' in df.columns:
                st.markdown("**Por Tipo de Contratación:**")
                type_counts = df['internal_type'].value_counts()
                st.dataframe(type_counts, use_container_width=True)
        
        with col2:
            if 'region' in df.columns:
                st.markdown("**Por Región/Provincia:**")
                region_counts = df['region'].value_counts().head(10)
                st.dataframe(region_counts, use_container_width=True)
    
    # TAB 2: VISUALIZACIONES PRINCIPALES
    with tab2:
        st.header("5. Visualización de Datos")
        
        viz_barras_tipo(df)
        
        st.markdown("---")
        viz_linea_mensual(df)
        
        st.markdown("---")
        viz_barras_apiladas(df)
        
        st.markdown("---")
        viz_pastel_proporcion(df)
    
    # TAB 3: CORRELACIONES
    with tab3:
        st.header("6. Relación entre Monto Total y Cantidad de Contratos")
        
        viz_dispersion_monto_contratos(df)
        
        # Análisis adicional de correlación
        st.markdown("---")
        st.subheader("📊 Matriz de Correlación")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title='Matriz de Correlación entre Variables Numéricas',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                labels=dict(color="Correlación")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="interpretation">
            <strong>🔍 Interpretación de la Matriz:</strong><br>
            • Valores cercanos a <strong>+1</strong>: correlación positiva fuerte<br>
            • Valores cercanos a <strong>-1</strong>: correlación negativa fuerte<br>
            • Valores cercanos a <strong>0</strong>: sin correlación lineal
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 4: COMPARATIVA TEMPORAL
    with tab4:
        st.header("7. Comparativa de Tipos de Contratación a lo largo del Año")
        
        viz_comparativa_tipos_mes(df)
        
        # Análisis de tendencias
        st.markdown("---")
        st.subheader("📈 Análisis de Tendencias Mensuales")
        
        if 'month' in df.columns and 'total' in df.columns:
            tendencia_mensual = df.groupby('month')['total'].agg(['sum', 'mean', 'count']).reset_index()
            tendencia_mensual.columns = ['Mes', 'Total', 'Promedio', 'Cantidad']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Tabla de Tendencias:**")
                st.dataframe(tendencia_mensual, use_container_width=True)
            
            with col2:
                # Identificar picos y valles
                mes_pico = tendencia_mensual.loc[tendencia_mensual['Total'].idxmax(), 'Mes']
                monto_pico = tendencia_mensual['Total'].max()
                mes_valle = tendencia_mensual.loc[tendencia_mensual['Total'].idxmin(), 'Mes']
                monto_valle = tendencia_mensual['Total'].min()
                
                st.markdown("**Hallazgos Clave:**")
                st.metric("Mes con Mayor Actividad", f"Mes {int(mes_pico)}", f"${monto_pico:,.2f}")
                st.metric("Mes con Menor Actividad", f"Mes {int(mes_valle)}", f"${monto_valle:,.2f}")
                
                variacion = ((monto_pico - monto_valle) / monto_valle * 100) if monto_valle > 0 else 0
                st.metric("Variación Pico-Valle", f"{variacion:.1f}%")
    
    # TAB 5: ANÁLISIS POR AÑOS
    with tab5:
        st.header("8. Análisis por Años")
        
        if 'year' in df.columns and df['year'].nunique() > 1:
            analisis_por_anios(df)
        else:
            st.info("ℹ️ Este análisis requiere datos de múltiples años. Seleccione 'Todos' en el filtro de año para ver comparaciones temporales.")
            
            # Mostrar análisis del año único
            if 'year' in df.columns:
                anio_actual = df['year'].iloc[0]
                st.markdown(f"### Análisis del Año {anio_actual}")
                
                kpis_anio = calcular_kpis(df)
                mostrar_kpis(kpis_anio)
    
    # TAB 6: EXPORTAR DATOS
    with tab6:
        st.header("9. Exportación de Resultados")
        
        st.markdown("""
        Descargue los datos procesados y listos para análisis adicional:
        """)
        
        # Opción 1: Datos completos
        st.subheader("📥 Datos Procesados Completos")
        csv_completo = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV Completo",
            data=csv_completo,
            file_name=f"compras_publicas_procesado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Opción 2: Resumen estadístico
        st.subheader("📊 Resumen Estadístico")
        if 'internal_type' in df.columns and 'total' in df.columns:
            resumen = df.groupby('internal_type').agg({
                'total': ['sum', 'mean', 'count', 'std']
            }).round(2)
            resumen.columns = ['Total', 'Promedio', 'Cantidad', 'Desv. Estándar']
            
            csv_resumen = resumen.to_csv().encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Resumen por Tipo",
                data=csv_resumen,
                file_name=f"resumen_tipos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.dataframe(resumen, use_container_width=True)
        
        # Opción 3: Datos filtrados personalizados
        st.subheader("🔍 Exportar Datos Filtrados")

        tipos_seleccionados = []
        regiones_seleccionadas = []
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'internal_type' in df.columns:
                tipos_seleccionados = st.multiselect(
                    "Seleccionar Tipos",
                    options=df['internal_type'].unique(),
                    default=[]
                )
        
        with col2:
            if 'region' in df.columns:
                regiones_seleccionadas = st.multiselect(
                    "Seleccionar Regiones",
                    options=df['region'].unique(),
                    default=[]
                )
        
        # Aplicar filtros
        df_filtrado = df.copy()
        
        if tipos_seleccionados and 'internal_type' in df.columns:
            df_filtrado = df_filtrado[df_filtrado['internal_type'].isin(tipos_seleccionados)]
        
        if regiones_seleccionadas and 'region' in df.columns:
            df_filtrado = df_filtrado[df_filtrado['region'].isin(regiones_seleccionadas)]
        
        st.info(f"📊 Registros seleccionados: {len(df_filtrado)} de {len(df)}")
        
        if len(df_filtrado) > 0:
            csv_filtrado = df_filtrado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Datos Filtrados",
                data=csv_filtrado,
                file_name=f"compras_publicas_filtrado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Vista previa
        with st.expander("👁️ Vista Previa de Datos a Exportar"):
            st.dataframe(df_filtrado.head(20), use_container_width=True)
    
    # ==================== FOOTER ====================
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Dashboard de Análisis de Compras Públicas Ecuador</strong></p>
        <p>Fuente de datos: Portal de Datos Abiertos SERCOP | Desarrollado con Streamlit + Plotly</p>
        <p>© 2024 - Análisis Exploratorio de Datos</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== EJECUTAR APLICACIÓN ====================

if __name__ == "__main__":
    main()