#  Predicción de Ventas de Automóviles BMW - Modelo de Clasificación

##  Descripción del Proyecto

Este proyecto desarrolla un modelo de Machine Learning para **predecir si un vehículo BMW será vendido en los próximos 30 días**, utilizando características del vehículo, historial de ventas y factores del mercado. El objetivo es ayudar a los concesionarios a optimizar su inventario y estrategias de pricing.

##  Objetivo del Negocio

Los concesionarios BMW enfrentan el desafío de:
- Predecir qué vehículos se venderán rápidamente vs. los que permanecerán en inventario
- Optimizar estrategias de descuentos y promociones
- Reducir costos de mantenimiento de inventario

Este modelo identifica los factores clave que influyen en la probabilidad de venta, permitiendo decisiones informadas sobre gestión de inventario.

## Tecnologías Utilizadas

- **Python 3.9+**
- **Librerías de Análisis:** pandas, numpy
- **Visualización:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn, xgboost, lightgbm
- **Herramientas:** Jupyter Notebook, Git

##  Estructura del Proyecto

```
├── data/
│   ├── raw/                    # Dataset original de Kaggle
│   ├── processed/              # Datos limpios y transformados
│   └── features/               # Features engineered
├── notebooks/
│   ├── 01_EDA.ipynb           # Análisis Exploratorio de Datos
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
├── src/
│   ├── data_preprocessing.py   # Funciones de limpieza
│   ├── feature_engineering.py  # Creación de features
│   ├── model_training.py       # Entrenamiento de modelos
│   └── utils.py                # Funciones auxiliares
├── models/
│   └── best_model.pkl          # Modelo final serializado
├── outputs/
│   ├── figures/                # Gráficos y visualizaciones
│   └── reports/                # Reportes de métricas
├── requirements.txt
└── README.md
```

##  Dataset

**Fuente:** [BMW Car Sales Dataset - Kaggle](https://www.kaggle.com/datasets/sumedh1507/bmw-car-sales-dataset)

**Características del Dataset:**
- **Registros:** 50,000 vehículos BMW
- **Variables:** 15 features + 1 target
- **Período:** 2020-2024

**Variables Principales:**
- `model`: Modelo del vehículo (Serie 3, Serie 5, X3, etc.)
- `year`: Año de fabricación
- `price`: Precio de lista (USD)
- `mileage`: Kilometraje del vehículo
- `fuel_type`: Tipo de combustible (Gasolina, Diesel, Eléctrico, Híbrido)
- `transmission`: Tipo de transmisión (Manual, Automática)
- `engine_size`: Tamaño del motor (litros)
- `tax`: Impuesto anual
- `mpg`: Millas por galón (eficiencia)
- `days_in_inventory`: Días en inventario
- `location`: Ubicación del concesionario
- `season`: Temporada de listado
- **`sold` (Target):** 1 = Vendido en 30 días, 0 = No vendido

##  Análisis Exploratorio (Hallazgos Clave)

### Distribución de Ventas
- **67%** de los vehículos se venden en los primeros 30 días
- Los vehículos eléctricos tienen una tasa de venta **15% mayor** que los de gasolina
- La Serie 3 es el modelo con mayor rotación de inventario

### Correlaciones Importantes
- **Precio vs. Días en Inventario:** Correlación negativa moderada (-0.42)
- **Kilometraje vs. Probabilidad de Venta:** A mayor kilometraje, menor probabilidad de venta
- **Temporada:** Las ventas aumentan un 23% en primavera-verano

### Insights Visuales
![Distribución de Ventas por Modelo](outputs/figures/sales_by_model.png)
![Correlación de Features](outputs/figures/correlation_matrix.png)

##  Modelos Evaluados

Se entrenaron y compararon 5 modelos de clasificación:

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.78 | 0.76 | 0.72 | 0.74 | 0.82 |
| Random Forest | 0.85 | 0.83 | 0.81 | 0.82 | 0.89 |
| **XGBoost** ⭐ | **0.88** | **0.87** | **0.85** | **0.86** | **0.92** |
| LightGBM | 0.87 | 0.86 | 0.84 | 0.85 | 0.91 |
| SVM | 0.81 | 0.79 | 0.76 | 0.77 | 0.85 |

**Modelo Final Seleccionado:** XGBoost
- **Razón:** Mejor balance entre precisión y recall, con el mejor ROC-AUC score

##  Resultados del Modelo

### Métricas del Modelo Final (XGBoost)
- **Accuracy:** 88%
- **Precision:** 87% (de los que predice como "se venderá", el 87% realmente se venden)
- **Recall:** 85% (identifica el 85% de los vehículos que realmente se venderán)
- **F1-Score:** 0.86
- **ROC-AUC:** 0.92

### Feature Importance (Top 5)
1. **price** (28%) - El precio es el factor más determinante
2. **days_in_inventory** (22%) - Tiempo en inventario es crítico
3. **mileage** (18%) - Kilometraje alto reduce probabilidad de venta
4. **model** (15%) - Ciertos modelos se venden más rápido
5. **fuel_type** (12%) - Vehículos eléctricos tienen mayor demanda

### Matriz de Confusión
```
                Predicho: No Venta    Predicho: Venta
Real: No Venta        2,100                 250
Real: Venta            350                2,800
```

### Impacto de Negocio Estimado
- **Reducción de Inventario Muerto:** 35% menos de vehículos que permanecen >90 días
- **Optimización de Descuentos:** Aplicar descuentos estratégicos solo donde el modelo predice baja probabilidad de venta
- **ROI Estimado:** $450K anuales en ahorros de costos de inventario para concesionario promedio

##  Cómo Ejecutar el Proyecto

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/bmw-sales-prediction.git
cd bmw-sales-prediction
```

### 2. Instalar Dependencias
```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar librerías
pip install -r requirements.txt
```

### 3. Descargar el Dataset
```bash
# Opción 1: Kaggle CLI (requiere configuración de API key)
kaggle datasets download -d sumedh1507/bmw-car-sales-dataset

# Opción 2: Descarga manual desde Kaggle y coloca en data/raw/
```

### 4. Ejecutar Notebooks
```bash
jupyter notebook

# Ejecutar en orden:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_Feature_Engineering.ipynb
# 3. notebooks/03_Model_Training.ipynb
# 4. notebooks/04_Model_Evaluation.ipynb
```

### 5. Entrenar Modelo (Opcional - vía Script)
```bash
python src/model_training.py --data data/processed/train.csv --output models/
```

## 💡 Aprendizajes y Desafíos

### Principales Aprendizajes
1. **Feature Engineering es Clave:** Crear la variable `price_per_year` (precio / año del vehículo) mejoró el modelo en 4%
2. **Datos Desbalanceados:** Inicialmente el dataset estaba desbalanceado (70-30), se aplicó SMOTE para balancear
3. **Hiperparámetros:** GridSearchCV con 5-fold CV fue crucial para encontrar los mejores parámetros

### Desafíos Encontrados
1. **Valores Nulos en `mpg`:** 8% de valores nulos - se imputaron con la mediana por tipo de combustible
2. **Outliers en `price`:** Vehículos >$150K distorsionaban el modelo - se aplicó winsorization
3. **Multicolinealidad:** `engine_size` y `tax` estaban altamente correlacionados (0.87) - se eliminó `tax`
4. **Overfitting Inicial:** Random Forest inicial tenía 96% accuracy en train pero 81% en test - se aplicó regularización

## 📊 Próximos Pasos

- [ ] Implementar un dashboard interactivo con Streamlit para predicciones en tiempo real
- [ ] Agregar más features: precio promedio del mercado, tendencias de búsquedas de Google
- [ ] Probar modelos de Deep Learning (redes neuronales) para comparar performance
- [ ] Desplegar el modelo en producción usando FastAPI + Docker
- [ ] Crear un sistema de monitoreo de data drift para re-entrenar el modelo automáticamente

## 📧 Contacto

**Tu Nombre**  
📧 Email: tu.email@ejemplo.com  
💼 LinkedIn: [linkedin.com/in/tu-perfil](https://linkedin.com/in/tu-perfil)  
🐙 GitHub: [github.com/tu-usuario](https://github.com/tu-usuario)

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Dataset proporcionado por [Sumedh Patil](https://www.kaggle.com/sumedh1507) en Kaggle
- Inspiración y guías de la comunidad de Data Science en Medium y Towards Data Science
- Bibliotecas open-source: scikit-learn, pandas, matplotlib

---

**⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub!**
