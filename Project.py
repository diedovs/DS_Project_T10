import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Завантаження набору даних
data = pd.read_csv("PET_PRI_GND_DCUS_NUS_W.csv")

# Вибір відповідних стовпців, включаючи 'Date', 'R1', 'M1', 'P1', 'D1'
selected_columns = ['Date', 'R1', 'M1', 'P1', 'D1']
df = data[selected_columns]

# Конвертація стовпця дати у формат datetime
df['Date'] = pd.to_datetime(df['Date'])

# Розділити дані на ознаки (X) та цільову змінну (y)
X = df[['R1', 'M1', 'P1']]
y = df['D1']

# Розділити дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація та тренування моделі лінійної регресії на тренувальному наборі
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозування цін на тестовому наборі
y_pred = model.predict(X_test)

# Виведення результатів для тестового набору
print("Метрики тестового набору:")
print("Середньоквадратична помилка (MSE):", mean_squared_error(y_test, y_pred))
print("Середня абсолютна помилка (MAE):", mean_absolute_error(y_test, y_pred))
print("R-квадрат (R2):", r2_score(y_test, y_pred))

# Прогнозування цін на пальне на 2022 рік
latest_data = df.tail(1)
X_2022 = pd.DataFrame({'R1': latest_data['R1'], 'M1': latest_data['M1'], 'P1': latest_data['P1']})
y_pred_2022 = model.predict(X_2022)
forecast_2022 = pd.DataFrame({'Дата': [datetime(2022, 1, 1)], 'R1': X_2022['R1'].values, 'M1': X_2022['M1'].values, 'P1': X_2022['P1'].values, 'D1': y_pred_2022})

# Виведення пояснень до стовпців
print("\nПояснення до стовпців:")
print("Дата: Дата вимірювання цін на пальне.")
print("R1: Ціна за галон для звичайного бензину у доларах США.")
print("M1: Ціна за галон для середньоформульованого бензину у доларах США.")
print("P1: Ціна за галон для преміального бензину у доларах США.")
print("D1: Ціна за галон для дизельного пального у доларах США.")

# Виведення прогнозів для 2022 року
print("\nПрогноз на 2022 рік:")
print(forecast_2022)

# Розрахунок відсоткового зростання цін
latest_price = latest_data['D1'].values[0]
forecasted_price_2022 = forecast_2022['D1'].values[0]
percentage_increase = ((forecasted_price_2022 - latest_price) / latest_price) * 100

print("\nВідсоткове зростання цін на 2022 рік:", round(percentage_increase, 2), "%")
