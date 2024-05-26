import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Sample dataset - Replace with your actual dataset
temperature = np.array([
    [3.9, 4.8, 6.5, 8.2, 12.2, 15.7, 17.6, 18.9, 18.1, 11.0, 6.8, 5.7],
    [4.5, 4.5, 5.8, 9.9, 14.1, 15.3, 18.0, 18.5, 17.7, 11.9, 6.6, 5.6],
    [4.2, 4.4, 6.3, 8.6, 13.6, 15.4, 17.7, 18.2, 17.8, 11.3, 7.0, 5.5],
    [4.0, 5.0, 7.2, 7.8, 11.0, 16.1, 17.6, 19.9, 18.2, 10.8, 6.9, 5.6],
    [4.4, 4.9, 6.9, 9.3, 12.8, 15.6, 17.4, 19.2, 17.9, 11.2, 6.7, 5.5]
])

aridity = np.array([
    [103.24, 103.24, 103.24, 103.24, 103.24, 103.24, 103.24, 103.24, 103.24, 103.24, 103.24, 103.24],
    [104.84, 104.84, 104.84, 104.84, 104.84, 104.84, 104.84, 104.84, 104.84, 104.84, 104.84, 104.84],
    [105.95, 105.95, 105.95, 105.95, 105.95, 105.95, 105.95, 105.95, 105.95, 105.95, 105.95, 105.95],
    [107.86, 107.86, 107.86, 107.86, 107.86, 107.86, 107.86, 107.86, 107.86, 107.86, 107.86, 107.86],
    [108.19, 108.19, 108.19, 108.19, 108.19, 108.19, 108.19, 108.19, 108.19, 108.19, 108.19, 108.19],
])

sunlight_hours = np.array([
    [1, 2, 3, 4, 5, 6, 7, 6, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 6, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 6, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 6, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 6, 4, 3, 2, 1],
])

humidity = np.array([
    [87, 82, 81, 76, 72, 72, 71, 73, 78, 83, 85, 84],
    [86, 83, 81, 76, 74, 72, 70, 71, 80, 84, 84, 85],
    [87, 84, 82, 77, 73, 71, 71, 72, 79, 84, 86, 85],
    [85, 82, 80, 75, 74, 72, 71, 73, 79, 85, 86, 86],
    [86, 83, 81, 76, 73, 73, 70, 72, 79, 84, 86, 85],
])

water_temperature = np.array([
    [6.5, 6.6, 7.3, 9.2, 12.3, 14.8, 17.0, 17.1, 14.6, 11.1, 8.5, 7.0],
    [6.3, 6.4, 7.3, 9.1, 12.2, 14.6, 16.9, 17.3, 14.7, 11.2, 8.6, 7.1],
    [6.3, 6.5, 7.2, 9.0, 12.1, 14.6, 16.8, 17.3, 14.8, 11.3, 8.7, 7.2],
    [6.4, 6.5, 7.2, 9.1, 12.1, 14.7, 16.8, 17.2, 14.8, 11.2, 8.8, 7.2],
    [6.5, 6.6, 7.3, 9.2, 12.2, 14.7, 16.9, 17.2, 14.7, 11.2, 8.7, 7.1],
])

soil_ph = np.array([
    [6.29, 6.29, 6.29, 6.29, 6.29, 6.29, 6.29, 6.29, 6.29, 6.29, 6.29, 6.29],
    [5.42, 5.42, 5.42, 5.42, 5.42, 5.42, 5.42, 5.42, 5.42, 5.42, 5.42, 5.42],
    [6.17, 6.17, 6.17, 6.17, 6.17, 6.17, 6.17, 6.17, 6.17, 6.17, 6.17, 6.17],
    [5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83, 5.83],
    [6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05],
])

list1 = []
list2 = []
list3 = []
list4 = []
list5 = []

for i in range(0, 12):
  list1.append((math.exp(-i^int(0.37))))
  list2.append((math.exp(-i^int(0.36))))
  list3.append((math.exp(-i^int(0.38))))
  list4.append((math.exp(-i^int(0.39))))
  list5.append((math.exp(-i^int(0.35))))

Y = np.array([list1, list2, list3, list4, list5])

# Define parameters for the logistic growth equation(set it as tobe inputed)
t = 60  # Time in months
K = 2000  # Carrying capacity
N_0 = 100  # Initial population size
N_t = 1000  # Population size at time t


# Combine the environmental variables into a feature matrix X
X = np.column_stack((temperature.flatten(), aridity.flatten(), soil_ph.flatten(), sunlight_hours.flatten(), humidity.flatten(), water_temperature.flatten()))

# Calculate the growth rate (r) using the logistic growth equation
r = (1 / t) * math.log((N_0 - K) / (N_0) * (K / (N_t - K)))


def perform_linear_regression(X, y, feature_name):
    model = LinearRegression()
    model.fit(X, y)
    beta_0 = model.intercept_
    beta = model.coef_

    # Print results
    print(f'{feature_name} Linear Regression:')
    print(f'beta_0: {beta_0}')
    print(f'beta: {beta}')

    # Plot the regression line
    sns.scatterplot(x=X[:, 0], y=y)
    plt.plot(X[:, 0], model.predict(X), color='red', linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('Impact Factor')
    plt.title(f'{feature_name} vs Impact Factor')
    plt.show()

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, Y.flatten())

# Extract the coefficients
beta_0 = model.intercept_
beta_1, beta_2, beta_3, beta_4, beta_5, beta_6 = model.coef_

# Calculate the Impact Factor using the provided beta values
impact_factor_predicted = (beta_0 + beta_1 * temperature + beta_2 * aridity + beta_3 * soil_ph + beta_4 * sunlight_hours + beta_5 * humidity + beta_6 * water_temperature) * r

# Print the results
print(f'Estimated Coefficients:')
print(f'beta_0: {beta_0}')
print(f'beta_1: {beta_1}')
print(f'beta_2: {beta_2}')
print(f'beta_3: {beta_3}')
print(f'beta_4: {beta_4}')
print(f'beta_5: {beta_5}')
print(f'beta_6: {beta_6}')
print(f'Growth Rate (r): {r}')
print(f'Impact Factor (Predicted): {impact_factor_predicted}')

# Plot the data and regression lines
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sns.regplot(x=temperature.flatten(), y=impact_factor_predicted.flatten(), ax=axes[0, 0])
sns.regplot(x=aridity.flatten(), y=impact_factor_predicted.flatten(), ax=axes[0, 1])
sns.regplot(x=soil_ph.flatten(), y=impact_factor_predicted.flatten(), ax=axes[0, 2])
sns.regplot(x=sunlight_hours.flatten(), y=impact_factor_predicted.flatten(), ax=axes[1, 0])
sns.regplot(x=humidity.flatten(), y=impact_factor_predicted.flatten(), ax=axes[1, 1])
sns.regplot(x=water_temperature.flatten(), y=impact_factor_predicted.flatten(), ax=axes[1, 2])

plt.tight_layout()
plt.show()

# Time array
time_array = np.arange(0, t + 1)

# Calculate population growth with impact factor and survival rate
def logistic_growth(t, N0, K, r):
    """
    Compute the logistic growth at time t.

    Parameters:
    - t: array of time values
    - N0: initial population
    - K: carrying capacity
    - r: growth rate

    Returns:
    - N: population at each time point
    """
    N = K  / (1 + ((K - N0) / N0 ) * survival_curve(t, 0.39))
    return N

# Calculate survival curve
def survival_curve(t, beta_survival):
    return np.exp(-np.power(t, beta_survival))

# Calculate population growth with impact factor and survival rate
population_growth_with_factors = logistic_growth(time_array, 100, K, r)

# Plot the population growth with impact factor and survival rate
plt.figure(figsize=(8, 6))
plt.plot(time_array, population_growth_with_factors, label='Population Growth with Factors')
plt.xlabel('Time (months)')
plt.ylabel('Population Size')
plt.title('Population Growth Over Time with Impact Factor and Survival Rate')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic

def survival_curve(t, N0, K, r):
    """
    Compute the survival curve using the logistic distribution CDF.

    Parameters:
    - t: array of time values
    - N0: initial population
    - K: carrying capacity
    - r: growth rate

    Returns:
    - Survival probability at each time point
    """
    survival_prob = logistic.cdf(r * (t - np.log((K - N0) / N0)))
    return 1 - survival_prob



# Example usage
t_values = np.linspace(0, 60, 120)  # Time values from 0 to 10
N0 = 100  # Initial population
K = 2000  # Carrying capacity
r = 0.06062643599543976  # Growth rate

# Compute survival probability at each time point
survival_probability = survival_curve(t_values, N0, K, r)


# Plot the survival curve
plt.plot(t_values, survival_probability, label='Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Survival Curve using Logistic Distribution')
plt.legend()
plt.show()