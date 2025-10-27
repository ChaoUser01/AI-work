import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''def train_data(x_train: list):
    y_train = []
    slope = (y_data[-1] - y_data[0])/ (x_train[-1] - x_train[0])
    for i in x_train:
        eqn = slope*i + random.random()
        y_train.append(eqn)
    return y_train
def predict(num:int, x_dat: list):      
    y_train = train_data(x_dat)
    slope = (y_train[-1] - y_train[0]) / (x_dat[-1] - x_dat[0])
    new_data = []
    output = []
    for i in range(num):
        #val = i * random.randint(1, 100)
        future_value = slope*i 
        output.append(future_value)
        new_data.append(i)
    print(new_data)
    print(output)
print(f"x dataset: {x_data}")
print(f"y dataset: {y_data}")
predict(7, x_data)  nb ''' 


'''def gen_data(seed: int, slope=3, intercept=40, noise_std=30):
    np.random.seed(seed)
    x = np.random.randint(0, 500, size=40)
    noise = np.random.normal(loc=0, scale=noise_std, size=40)
    y = slope * x + intercept + noise
    data = {'x': x, 'y': y}
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)

gen_data(42, slope=5, intercept=41.225, noise_std=100)
'''
# Version 2
'''x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([2, 4, 6, 8, 10, 12])

def linear_regression(x, y):
    n = len(x)
    m = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - m*np.sum(x)) / n
    return m, b

slope, intercept = linear_regression(x_data, y_data)

def predict(x, slope, intercept):
    return slope * x + intercept

x_new = np.array([7, 8, 9, 10])
y_pred = predict(x_new, slope, intercept)

plt.scatter(x_data, y_data, label="Original data")
plt.plot(x_data, predict(x_data, slope, intercept), color="red", label="Fit line")
plt.scatter(x_new, y_pred, color="green", label="Extrapolated")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression & Prediction")
plt.show()
'''
#Version 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_data(seed: int, slope=4.0, intercept=40.0, noise_std=80.0):
    np.random.seed(seed)
    x = np.random.randint(0, 500, size=80)
    noise = np.random.normal(0, noise_std, size=80)
    y = slope * x + intercept + noise
    return x, y

def linear_regression(x, y):
    n = len(x)
    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    b = (np.sum(y) - m * np.sum(x)) / n
    return m, b

def predict(x, slope, intercept):
    return slope * x + intercept

x_data, y_data = gen_data(seed=1000)

slope, intercept = linear_regression(x_data, y_data)

x_new = np.array([510, 520, 530, 540]) 
y_pred = predict(x_new, slope, intercept)

plt.scatter(x_data, y_data, label="Original noisy data")
plt.plot(x_data, predict(x_data, slope, intercept), color="red", label="Fit line")
plt.scatter(x_new, y_pred, color="green", label="Extrapolated predictions")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression on Noisy Data & Prediction")
plt.show()
