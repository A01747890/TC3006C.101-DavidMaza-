import numpy as np
import pandas as pd
import re

# Leemos los datos y seleccionamos las columnas para renombrarlas como label y message
data = pd.read_csv('./spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convertimos los outputs de ham a 0 y spam a 1, esto con el objetivo de comprender más fácil la predicción
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Procesamos los mensajes convirtiendo todo el texto en minúsculas y eliminamos carcateres y espacios innecesarios
data['message'] = data['message'].str.lower().str.replace(r'\W', ' ', regex=True).str.strip()

# Creamos un vocabulario y cramos una lista con este mismo vocabulario creado
vocabulary = set()
for message in data['message']:
    for word in message.split():
        vocabulary.add(word)

vocabulary = list(vocabulary)
word_index = {word: i for i, word in enumerate(vocabulary)}

# Convertimos cada mensaje en un vector y contamos cuantas veces aparece cada palabra en el mensaje
def vectorize_text(text, word_index):
    vector = np.zeros(len(word_index))
    for word in text.split():
        if word in word_index:
            vector[word_index[word]] += 1
    return vector

#Guardamos los vectores en valores de x y y
X = np.array([vectorize_text(message, word_index) for message in data['message']])
y = data['label'].values

# Ajustamos los valores para que tengan una media de 0 y DE de 1
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Añadimos una columna de unos para el sesgo del modelo
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Dividimos los datos de entrenamiento, de prueba y de validación
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Aplicamos la función de regresión logística de sigmoide para determinar si un mensaje es spam
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calculamos la función de costo para saber que tan bien está funcionando el modelo
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -1/m * (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h)))

# Aplicamos el Gradiente Descendente para ajustar los parámetros del modelo y así minimizar el costo
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        theta -= (alpha / m) * X.T.dot(h - y)
        cost_history[i] = cost_function(X, y, theta)
    
    return theta, cost_history

# Establecemos los parámetros del modelo
theta = np.zeros(X_train.shape[1])
alpha = 0.01  
num_iters = 1500  

# Entrenamos el modelo usando los datos de entrenamiento
theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)

# Hacemos la predicción usando la función y determianamos si un mensaje es spam o no
def predict(X):
    prob = sigmoid(X.dot(theta))
    return prob >= 0.5

# Evaluamos con validación
y_valid_pred = predict(X_valid)
y_valid_true = y_valid

# y con los datos de prueba
y_test_pred = predict(X_test)
y_test_true = y_test

# Evaluamos las métricas de evaluación del modelo sacando f1, matriz compuesta, etc.
def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, (tp, tn, fp, fn)

precision_valid, recall_valid, f1_valid, cm_valid = evaluate(y_valid_true, y_valid_pred)
precision_test, recall_test, f1_test, cm_test = evaluate(y_test_true, y_test_pred)

print(f"Validation Precision: {precision_valid}")
print(f"Validation Recall: {recall_valid}")
print(f"Validation F1: {f1_valid}")
print(f"Validation Matriz de Confusión: VP={cm_valid[0]}, VN={cm_valid[1]}, FP={cm_valid[2]}, FN={cm_valid[3]}")

print(f"Test Precision: {precision_test}")
print(f"Test Recall: {recall_test}")
print(f"Test F1: {f1_test}")
print(f"Test Matriz de Confusión: VP={cm_test[0]}, VN={cm_test[1]}, FP={cm_test[2]}, FN={cm_test[3]}")

# Hacemos una predicción con un mensaje nuevo y establecido por nosotros para visualizar la efectividad del modelo
def predict_text(message):
    message = re.sub(r'\W', ' ', message.lower()).strip()
    features = vectorize_text(message, word_index)

    features = (features - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)
    features = np.hstack(([1], features))
    prob = sigmoid(features.dot(theta))
    return "Spam" if prob >= 0.5 else "Ham"

# Ceamos el ejemplo de la predicción
new_message = "Hey, just checking in to see if you’re still on for our meeting tomorrow. Let me know if you need to reschedule."
result = predict_text(new_message)
print("MENSAJE:")
print(new_message)
print(f"La predicción para el mensaje ingresado es: {result}")
