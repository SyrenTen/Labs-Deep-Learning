import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Генеруємо дані
mg = np.random.RandomState(1)
x = np.linspace(0, 6, 200)
y = np.sin(x) + np.sin(6*x) + mg.normal(0, 0.1, x.shape[0])

# Розділяємо на тренувальні та тестові набори
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.3, random_state=42)

# Підбір параметрів для MLPRegressor
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [500, 1000]
}

# Створюємо модель і налаштовуємо пошук
mlp = MLPRegressor(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(x_train, y_train)

# Отримуємо найкращі параметри
best_params = grid_search.best_params_
print(f'Найкращі параметри: {best_params}')

# Тренуємо модель з найкращими параметрами та збільшеною кількістю ітерацій
best_mlp = MLPRegressor(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=2000,  # Збільшено кількість ітерацій
    random_state=42
)

best_mlp.fit(x_train, y_train)

# Прогнози
y_train_pred = best_mlp.predict(x_train)
y_test_pred = best_mlp.predict(x_test)

# Обчислюємо метрики
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train MSE: {train_mse}, Train R2: {train_r2}')
print(f'Тест MSE: {test_mse}, Тест R2: {test_r2}')

# Візуалізація рез
plt.figure(figsize=(12, 6))

# Вих та тренувальні дані
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.plot(x_train, y_train_pred, color='red', label='Model Prediction')
plt.title('Train Data vs Prediction')
plt.legend()

# Тестові дані та прогноз
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color='green', label='Тест даны')
plt.plot(x_test, y_test_pred, color='red', label='Model Prediction')
plt.title('Test Data vs Prediction')
plt.legend()

plt.show()
