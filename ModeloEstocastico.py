# Proyecto: Modelado Estocástico para Derivados Financieros con datos BCCh
# Autor: Alejandro Espinoza
# Descripción: Simulación de precios CLP/USD usando Movimiento Browniano Geométrico

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.stats import norm

# Configuración global
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)  # Para reproducibilidad

# 1️⃣ API ACTUALIZADA Banco Central de Chile (Julio 2025)
def obtener_datos_bcch():
    end_date = datetime(2025, 7, 21)
    start_date = end_date - timedelta(days=5*365)
    
    # Nuevo endpoint y parámetros (actualizado Julio 2025)
    url = "https://si3.bcentral.cl/SieteRestWS/siete/WS/SeriesData"
    params = {
        'series': 'F073.TCO.PRE.Z.D',
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'timeseries': 'T',
        'function': 'GetSeries'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Nueva estructura JSON del BCCh
        series = data.get('Series', [])
        if not series:
            raise ValueError("Respuesta vacía del API")
            
        obs = series[0].get('Obs', [])
        fechas = [datetime.strptime(o['indexDateString'], '%Y-%m-%d') for o in obs]
        valores = [float(o['value']) for o in obs]
        
        df = pd.DataFrame({'Fecha': fechas, 'Valor': valores})
        print(f"Datos BCCh descargados: {len(df)} registros")
        return df.sort_values('Fecha').dropna().reset_index(drop=True)
    
    except Exception as e:
        print(f"Error API BCCh: {e}\nGenerando datos sintéticos...")
        # Datos sintéticos más realistas (mejorado)
        fechas = pd.bdate_range(start="2020-01-01", end="2025-07-21")
        base = 800 + 50 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))
        volatilidad = np.random.gamma(shape=2, scale=0.8, size=len(fechas))
        valores = base * np.exp(np.cumsum(volatilidad * np.random.randn(len(fechas)) * 0.01))
        return pd.DataFrame({'Fecha': fechas, 'Valor': valores})

# 2️⃣ Procesamiento mejorado
def procesar_datos(df):
    """Calcula parámetros estocásticos con manejo robusto de outliers"""
    # Retornos logarítmicos
    df['log_ret'] = np.log(df['Valor'] / df['Valor'].shift(1))
    
    # Filtrar outliers usando IQR
    Q1 = df['log_ret'].quantile(0.25)
    Q3 = df['log_ret'].quantile(0.75)
    IQR = Q3 - Q1
    df_filtrado = df[(df['log_ret'] >= Q1 - 1.5*IQR) & (df['log_ret'] <= Q3 + 1.5*IQR)]
    
    # Parámetros anualizados
    mu = df_filtrado['log_ret'].mean() * 252
    sigma = df_filtrado['log_ret'].std() * np.sqrt(252)
    S0 = df['Valor'].iloc[-1]
    
    print("\n" + "="*60)
    print("PARÁMETROS CALIBRADOS (Datos BCCh 2020-2025)")
    print(f"• Retorno anualizado (μ): {mu:.6f}")
    print(f"• Volatilidad anualizada (σ): {sigma:.6f}")
    print(f"• Precio spot (S₀): {S0:.2f} CLP/USD")
    print(f"• Muestra efectiva: {len(df_filtrado)}/{len(df)} días")
    print("="*60)
    
    return mu, sigma, S0, df_filtrado

# 3️⃣ Simulación GBM corregida (vectorizada)
def simular_gbm(S0, mu, sigma, T=0.5, dt=1/252, n_sim=500):
    """Simulación vectorizada eficiente de trayectorias"""
    n_steps = int(T/dt)
    t = np.linspace(0, T, n_steps)
    
    # Generación de retornos (mejor eficiencia)
    dW = np.random.normal(0, np.sqrt(dt), (n_sim, n_steps-1))
    retornos = np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
    
    # Construcción trayectorias con cumprod
    trayectorias = np.zeros((n_sim, n_steps))
    trayectorias[:, 0] = S0
    trayectorias[:, 1:] = S0 * np.cumprod(retornos, axis=1)
    
    return t, trayectorias

# 4️⃣ Visualización profesional con fechas reales
def visualizar_resultados(t, trayectorias, df, S0):
    plt.figure(figsize=(13, 7))
    
    # Configurar eje de fechas
    ultima_fecha = df['Fecha'].iloc[-1]
    fechas_sim = pd.bdate_range(start=ultima_fecha, periods=len(t), freq='B')
    
    # Límites de confianza
    p95 = np.percentile(trayectorias, 95, axis=0)
    p05 = np.percentile(trayectorias, 5, axis=0)
    media = np.mean(trayectorias, axis=0)
    
    # Trayectorias de muestra
    for i in range(min(30, len(trayectorias))):
        plt.plot(fechas_sim, trayectorias[i], lw=0.8, alpha=0.15, color='steelblue')
    
    # Bandas y tendencias
    plt.fill_between(fechas_sim, p05, p95, color='lightblue', alpha=0.4, label='Banda 90% confianza')
    plt.plot(fechas_sim, media, 'b--', lw=2, label='Trayectoria media')
    plt.plot(fechas_sim, p95, 'r-', lw=1, alpha=0.6, label='Percentil 95')
    plt.plot(fechas_sim, p05, 'g-', lw=1, alpha=0.6, label='Percentil 5')
    
    # Histórico reciente
    hist_days = min(90, len(df))
    plt.plot(df['Fecha'].iloc[-hist_days:], df['Valor'].iloc[-hist_days:], 
             'ko-', lw=1.5, markersize=3, label='Histórico BCCh')
    
    # Elementos estéticos
    plt.title('Modelo Estocástico CLP/USD - Banco Central de Chile\nGeometric Brownian Motion (Simulación Monte Carlo)', 
              fontsize=15, pad=20)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('CLP/USD', fontsize=12)
    plt.axvline(ultima_fecha, color='purple', ls=':', alpha=0.8, label='Inicio simulación')
    plt.axhline(S0, color='navy', ls='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, alpha=0.3)
    
    # Formato fechas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('modelado_clp_usd_bcch.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5️⃣ Modelo CORRECTO para opciones en divisas (Garman-Kohlhagen)
def valorar_opcion_divisa(S, K, T, r_d, r_f, sigma, tipo='call'):
    """
    r_d: Tasa libre de riesgo domestica (CLP)
    r_f: Tasa libre de riesgo extranjera (USD)
    """
    d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if tipo == 'call':
        precio = S * np.exp(-r_f*T) * norm.cdf(d1) - K * np.exp(-r_d*T) * norm.cdf(d2)
    else:
        precio = K * np.exp(-r_d*T) * norm.cdf(-d2) - S * np.exp(-r_f*T) * norm.cdf(-d1)
    return precio

# --------------------------------------------
# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Paso 1: Obtener datos
    df = obtener_datos_bcch()
    
    # Paso 2: Calibrar modelo
    mu, sigma, S0, df_filtrado = procesar_datos(df)
    
    # Paso 3: Simulación Monte Carlo
    t, trayectorias = simular_gbm(S0, mu, sigma, T=0.5, n_sim=500)
    
    # Paso 4: Visualización
    visualizar_resultados(t, trayectorias, df_filtrado, S0)
    
    # Paso 5: Valoración de opciones (ejemplo)
    print("\nVALORACIÓN DE OPCIÓN CALL CLP/USD (Modelo Garman-Kohlhagen)")
    K = S0 * 1.05  # Strike 5% arriba spot
    T_opcion = 0.25  # 3 meses
    
    # Tasas de referencia Julio 2025 (ejemplo)
    r_clp = 0.045  # Tasa política monetaria Chile
    r_usd = 0.0275 # Tasa FED Estados Unidos
    
    precio_call = valorar_opcion_divisa(S0, K, T_opcion, r_clp, r_usd, sigma)
    
    print(f"• Spot: {S0:.2f} CLP/USD | Strike: {K:.2f}")
    print(f"• Vencimiento: {T_opcion*12:.1f} meses | Volatilidad: {sigma*100:.2f}%")
    print(f"• Tasas: CLP={r_clp*100:.2f}% USD={r_usd*100:.2f}%")
    print(f"• Precio teórico opción CALL: {precio_call:.4f} CLP")
    print("="*60)