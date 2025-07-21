# ModeloEstocastico_CLP_USD

Simulador estocástico del tipo **Geometric Brownian Motion (GBM)** con simulación **Monte Carlo** para el tipo de cambio CLP/USD, utilizando datos del **Banco Central de Chile (BCCh)**. Incluye valoración de opciones tipo **call/put** con el modelo **Garman-Kohlhagen** para opciones en divisas.

Funcionalidades
- Obtención de datos reales del BCCh (maneja errores generando datos sintéticos en contingencia).
- Calibración de retorno esperado (μ) y volatilidad (σ) anualizada con manejo de outliers.
- Simulación de trayectorias de tipo de cambio mediante GBM.
- Visualización profesional con Matplotlib.
- Valoración de opciones de divisas (call/put) con tasas locales y extranjeras.

Instalación
Clona este repositorio:
https://github.com/tu_usuario/ModeloEstocastico_CLP_USD.git

Instala las dependencias:
pip install -r requirements.txt


Uso
Ejecuta:
python ModeloEstocastico.py

Generará una gráfica `modelado_clp_usd_bcch.png` y mostrará parámetros calibrados y la valoración de una opción de ejemplo.

Licencia
Este proyecto está bajo la Licencia MIT.

✨ Autor
Desarrollado por [Alejandro Espinoza] como proyecto de práctica en **finanzas cuantitativas** y **Python**.

