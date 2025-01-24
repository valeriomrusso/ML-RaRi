import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

def load_monk_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():  # salta righe vuote
                values = line.strip().split()
                # converte stringhe in interi e rimuove ultima colonna
                row = [int(x) for x in values[:-1]] 
                data.append(row)
    return np.array(data)

# Carica i dati
train_data = load_monk_data('monks-1.train')
test_data = load_monk_data('monks-1.test')

# Separazione features e labels
X_train = train_data[:, 1:]  # tutte le colonne tranne la prima
y_train = train_data[:, 0]   # solo prima colonna
X_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Normalizzazione delle feature
X_train = X_train / np.max(X_train, axis=0)
X_test = X_test / np.max(X_test, axis=0)

# Modello con più unità e regolarizzazione
model = keras.Sequential([
    keras.layers.Dense(4, activation='tanh', input_shape=(6,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

batch_size = len(X_train)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=batch_size,
    verbose=1
)
# Metriche
train_accuracies = history.history['accuracy']
test_accuracies = history.history['val_accuracy']
train_losses = history.history['loss']
test_losses = history.history['val_loss']

# Compute mean squared error
train_mse = [mean_squared_error(y_train, model.predict(X_train))]*len(train_accuracies)
test_mse = [mean_squared_error(y_test, model.predict(X_test))]*len(test_accuracies)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

epochs = range(1, len(train_accuracies) + 1)

# Accuratezza
ax1.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
ax1.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
ax1.set_xlabel('Epoche')
ax1.set_ylabel('Accuratezza')
ax1.set_title('Accuratezza vs Epoche')
ax1.legend()
ax1.grid(True)

# Mean Squared Error
ax2.plot(epochs, train_mse, 'b-', label='Train MSE')
ax2.plot(epochs, test_mse, 'r-', label='Test MSE')
ax2.set_xlabel('Epoche')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('MSE vs Epoche')
ax2.legend()
ax2.grid(True)

plt.suptitle('Performance Rete Neurale MONK-1')
plt.tight_layout()
plt.show()

print(f'Accuratezza Train finale: {train_accuracies[-1]:.4f}')
print(f'Accuratezza Test finale: {test_accuracies[-1]:.4f}')
print(f'MSE Train finale: {train_mse[-1]:.4f}')
print(f'MSE Test finale: {test_mse[-1]:.4f}')
