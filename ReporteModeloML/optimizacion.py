import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score

# Cargamos el dataset
df = pd.read_csv("ConcreteStrengthData.csv")

# Dropeamos los datos nulos si es que los hay
df = df.dropna()

# Separamos la variable objetivo
X = df.drop(['Strength'], axis=1)
y = df['Strength']

# Dividimos los datos en train, test y validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Creamos el modelo de Random Forest
model = RandomForestRegressor(n_estimators=50, random_state=2016, min_samples_leaf=10)

# Implementamos la validación cruzada con K-Folds (k=5 en este caso)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Definimos los 'scorers' para MSE, MAE, y R2
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# Aplicamos la validación cruzada con el modelo para cada métrica
mse_scores = cross_val_score(model, X, y, cv=kf, scoring=mse_scorer)
mae_scores = cross_val_score(model, X, y, cv=kf, scoring=mae_scorer)
r2_scores = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)

# Convertimos los MSE y MAE a valores positivos para su interpretación
mse_scores = -mse_scores
mae_scores = -mae_scores

# Calculamos las métricas promedio y las imprimimos
print(f"Promedio MSE (K-Fold): {np.mean(mse_scores):.4f}")
print(f"Promedio MAE (K-Fold): {np.mean(mae_scores):.4f}")
print(f"Promedio R2 (K-Fold): {np.mean(r2_scores):.4f}")
# Entrenamos el modelo
model.fit(X_train, y_train)

# Evaluamos el modelo para los datos de validación
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_mae = mean_absolute_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

# Convertimos las predicciones y valores reales a clasificación binaria para la validación
umbral = 40
y_val_class = np.where(y_val > umbral, 1, 0)
val_predictions_class = np.where(val_predictions > umbral, 1, 0)

# Calculamos precisión, recall y F1-score para los datos de validación
precision = precision_score(y_val_class, val_predictions_class)
recall = recall_score(y_val_class, val_predictions_class)
f1 = f1_score(y_val_class, val_predictions_class)

# Imprimimos las métricas para los datos de validación
print(f"Validacion MSE: {val_mse:.4f}")
print(f"Validacion MAE: {val_mae:.4f}")
print(f"Validacion R2 Score: {val_r2:.4f}")
print(f"Validacion Precisión: {precision:.4f}")
print(f"Validacion Recall: {recall:.4f}")
print(f"Validacion F1 Score: {f1:.4f}")

# Evaluamos el modelo para los datos de prueba
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Convertimos las predicciones y valores reales a clasificación binaria para la prueba
y_test_class = np.where(y_test > umbral, 1, 0)
test_predictions_class = np.where(test_predictions > umbral, 1, 0)

# Calculamos el accuracy y la matriz de confusión para la clasificación binaria de la prueba
accuracy = accuracy_score(y_test_class, test_predictions_class)
matriz_confusion = confusion_matrix(y_test_class, test_predictions_class)

# Calculamos precisión, recall y F1-score para los datos de prueba
test_precision = precision_score(y_test_class, test_predictions_class)
test_recall = recall_score(y_test_class, test_predictions_class)
test_f1 = f1_score(y_test_class, test_predictions_class)

# Imprimimos las métricas para los datos de prueba
print("Matriz de confusión:")
print(matriz_confusion)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Test Precisión: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Hacemos una predicción manual
def predict_strength(CementComponent, BlastFurnaceSlag, FlyAshComponent, WaterComponent, SuperplasticizerComponent, CoarseAggregateComponent, FineAggregateComponent, AgeInDays):
    new_data = np.array([[CementComponent, BlastFurnaceSlag, FlyAshComponent, WaterComponent, SuperplasticizerComponent, CoarseAggregateComponent, FineAggregateComponent, AgeInDays]])
    prediction = model.predict(new_data)
    prediction_class = "alta" if prediction[0] > umbral else "baja"
    print(f"La fuerza del cemento es: {prediction[0]:.2f} ({prediction_class})")

# Ingresamos parámetros del cemento para predecir
predict_strength(100.0, 10.0, 5.0, 300.0, 0.5, 600.0, 400.0, 3)

# Información de los conjuntos de datos
print("\nDataset completo:")
print(df.head())
print("Tamaño del dataset:")
print(df.shape)

print("\nConjunto de entrenamiento:")
print(X_train.head())
print("Tamaño del conjunto de entrenamiento:")
print(X_train.shape)

print("\nConjunto de validación")
print(X_val.head())
print("Tamaño del conjunto de validación:")
print(X_val.shape)

print("\nConjunto de prueba")
print(X_test.head())
print("Tamaño del conjunto de prueba:")
print(X_test.shape)


#GRAFICAS:

# Definimos las métricas para la primera gráfica (MSE y MAE)
metrics1 = {
    'MSE': val_mse,
    'MAE': val_mae
}

# Creamos la primera gráfica
plt.figure(figsize=(10, 5))
bars1 = plt.bar(metrics1.keys(), metrics1.values(), color=['blue', 'orange'])

# Añadimos etiquetas a las barras
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.title('Comparación de MSE y MAE del Modelo en el conjunto de Validación')
plt.ylim(0, max(metrics1.values()) * 1.1)  
plt.show()

# Definimos las métricas para la segunda gráfica (R2, precisión, recall, F1-score, accuracy)
metrics2 = {
    'R2 Score': val_r2,
    'Precisión': precision,
    'Recall': recall,
    'F1 Score': f1,
    'Accuracy': accuracy
}

# Creamos la segunda gráfica
plt.figure(figsize=(12, 6))
bars2 = plt.bar(metrics2.keys(), metrics2.values(), color=['green', 'red', 'purple', 'brown', 'cyan'])

# Añadimos etiquetas a las barras
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.title('Métricas de evaluación del modelo en el conjunto de Validación')
plt.ylim(0, max(metrics2.values()) * 1.1)  
plt.show()

# Graficamos la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicción Baja', 'Predicción Alta'], 
            yticklabels=['Real Baja', 'Real Alta'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Definimos las métricas para la primera gráfica (MSE y MAE)
metrics1 = {
    'MSE': test_mse,
    'MAE': test_mae
}

# Creamos la primera gráfica
plt.figure(figsize=(10, 5))
bars1 = plt.bar(metrics1.keys(), metrics1.values(), color=['blue', 'orange'])

# Añadimos etiquetas a las barras
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.title('Comparación de MSE y MAE del Modelo en el conjunto de Pruebas')
plt.ylim(0, max(metrics1.values()) * 1.1) 


# Definimos las métricas para la segunda gráfica (R2, precisión, recall, F1-score, accuracy)
metrics2 = {
    'R2 Score': test_r2,
    'Precisión': test_precision,
    'Recall': test_recall,
    'F1 Score': test_f1,
    'Accuracy': accuracy
}

# Creamos la segunda gráfica
plt.figure(figsize=(12, 6))
bars2 = plt.bar(metrics2.keys(), metrics2.values(), color=['green', 'red', 'purple', 'brown', 'cyan'])

# Añadimos etiquetas a las barras
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.title('Métricas de evaluación del modelo en el conjunto de Pruebas')
plt.ylim(0, max(metrics2.values()) * 1.1) 
plt.show()
