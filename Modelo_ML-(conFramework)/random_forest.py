import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score

# Cargamos el dataset
#En este caso se usará un data set que tiene como objetivo predecir la dureza del cemento dependiendo de los diferentes parámetros que se le ingresan
df = pd.read_csv("ConcreteStrengthData.csv")

# Para optimizar el funcionamiento del modelo, dropeamos los datos nulos si es que los hay
df = df.dropna()

#En este caso, como el dataset ya incluye una categoría "Strength", lo vamos a separar para poder hacer las validaciones correctas
X = df.drop(['Strength'], axis=1)
y = df['Strength']

# Dividimos los datos en train, test y validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Creamos el modelo de Random forest con 19 arboles de decisión con 8 datos por hoja
model = RandomForestRegressor(n_estimators=19, random_state=2016, min_samples_leaf=8)

# Entrenamos el modelo con los datos del dataset
model.fit(X_train, y_train)

# Evaluamos el modelo y sacamos las métricas para los datos de validación
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_mae = mean_absolute_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)
print(f"Validacion MSE: {val_mse:.4f}")
print(f"Validacion MAE: {val_mae:.4f}")
print(f"Validacion R2 Score: {val_r2:.4f}")

# Evaluamos el modelo y sacamos las métricas para los datos de prueba
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")

# Para poder sacar las métricas como el accuracy y la matriz de confusión, convertiremos la predicción con un umbral para saber que tanta es la dureza del cemento
# En este caso, se considerará alta si la dureza es mayor a 40 y si es menor o igual se considerará baja
umbral = 40
y_test_class = np.where(y_test > umbral, 1, 0)
test_predictions_class = np.where(test_predictions > umbral, 1, 0)

# Calculamos el accuracy y la matriz de confusión para la clasificación binaria de la predicción
accuracy = accuracy_score(y_test_class, test_predictions_class)
matriz_confusion = confusion_matrix(y_test_class, test_predictions_class)

# Mostramos las métricas del accuracy y de la matriz de confusión
print(f"Accuracy: {accuracy:.4f}")
print("Matriz de confusión:")
print(matriz_confusion)

# Hacemos una predicción manual, en la que nosotros podamos ingresar diferentes parámetros del cemento y esperaremos una predicción de cuanta será la dureza y si es alta o baja
def predict_strength(CementComponent, BlastFurnaceSlag, FlyAshComponent, WaterComponent, SuperplasticizerComponent, CoarseAggregateComponent, FineAggregateComponent, AgeInDays):
    new_data = np.array([[CementComponent, BlastFurnaceSlag, FlyAshComponent, WaterComponent, SuperplasticizerComponent, CoarseAggregateComponent, FineAggregateComponent, AgeInDays]])
    prediction = model.predict(new_data)
    prediction_class = "alta" if prediction[0] > umbral else "baja"
    print(f"La fuerza del cemento es: {prediction[0]:.2f} ({prediction_class})")

# Ingresamos parametros del cemento que deseamos predecir
predict_strength(100.0, 10.0, 5.0, 300.0, 0.5, 600.0, 400.0, 3)
