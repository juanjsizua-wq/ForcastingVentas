---
title: Simulador de Ventas con ML
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app/app.py
pinned: false
license: mit
---

# 📊 Simulador de Ventas - Noviembre 2025

## 🎯 Descripción

Aplicación interactiva de **Machine Learning** para predecir ventas diarias de productos deportivos durante Noviembre 2025, incluyendo el impacto del **Black Friday**.

## ✨ Características

- 📈 **Predicciones día a día** con HistGradientBoostingRegressor
- 💰 **Simulador de descuentos** interactivo (-50% a +50%)
- 🏪 **Análisis de competencia** (Amazon, Decathlon, Deporvillage)
- 🔥 **Análisis especial Black Friday** (28 de Noviembre)
- 📊 **Visualizaciones dinámicas** con gráficos y tablas
- 🎯 **20 productos** en 3 categorías: Outdoor, Running, Wellness

## 🛠️ Stack Tecnológico

- **Python 3.10+**
- **Streamlit** - Framework de la aplicación
- **Scikit-learn** - Machine Learning (HistGradientBoosting)
- **Pandas & NumPy** - Procesamiento de datos
- **Matplotlib & Seaborn** - Visualizaciones

## 🚀 Uso

1. Selecciona un producto del catálogo
2. Ajusta el descuento deseado (-50% a +50%)
3. Elige el escenario de competencia
4. Haz clic en **"Simular Ventas"**
5. Analiza las predicciones diarias y KPIs

## 📦 Productos Disponibles

### 🏃 Running
- Zapatillas Running, Zapatillas Trail, Ropa Running

### 🏔️ Outdoor
- Mochila Trekking, Bicicleta Montaña, Ropa Montaña

### 🧘 Wellness
- Esterillas (Yoga/Fitness), Mancuernas, Pesas, Bloques y accesorios de Yoga

## 📊 KPIs Principales

- **Unidades Totales Proyectadas** por mes
- **Ingresos Totales** esperados
- **Precio Promedio** de venta
- **Descuento Promedio** aplicado

## 🤖 Modelo de Machine Learning

- **Algoritmo:** HistGradientBoostingRegressor
- **Predicción recursiva:** Actualización día a día con lags
- **Features:** Precio, competencia, categoría, día de semana, estacionalidad
- **Target:** Unidades vendidas diarias

## 📝 Autor

Desarrollado por **juanjsizua-wq**

## 📄 Licencia

MIT License
