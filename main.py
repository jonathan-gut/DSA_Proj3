import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Functions for Linear Regression
def gradient_descent(m_now, b_now, data, L): #Gradient Descent function
    m_gradient = 0
    b_gradient = 0

    n = len(data)
    for i in range(n):
        x = data.iloc[i].Day
        y = data.iloc[i].Price

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m,b #O(n)

#Functions for Polynomial Regression

def create_polynomial_features(X, degree):
    X_poly = np.ones((len(X), 1))
    for i in range(1, degree + 1):
        X_poly = np.c_[X_poly, X ** i]
    return X_poly #O(n*d) where n is len of X column and d is degree
def normal_equation(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) #O(m^3)
def predict(X, coefficients):
    return X.dot(coefficients) #O(n*d)
def predict_Y_for_given_X(X_poly, coefficients):
    return np.dot(X_poly, coefficients) #O(m)
def create_polynomial_features_for_single_X(X, degree):
    X_poly = np.ones(1)
    for i in range(1, degree + 1):
        X_poly = np.append(X_poly, X ** i)
    return X_poly.reshape(1, -1) #O(d)

'''
For time complexities
d = # of degrees
n = # of rows in csv (aka # of days)
m = # of features (aka independent vars + poly features)
def gradient_descent(m_now, b_now, data, L): O(n)
def create_polynomial_features(X, degree): O(n*d)
def normal_equation(X, Y): O(m^3)
def predict_Y_for_given_X(X_poly, coefficients): O(m)
def create_polynomial_features_for_single_X(X, degree): O(d)
'''
run = True
while (run):
    # Get user input for which stock to predict
    stocks = {1: "AAPL",
              2: "MSFT",
              3: "AMZN",
              4: "JNJ",
              5: "GOOGL"}
    print("Select what stock to predict")
    print("1. AAPL")
    print("2. MSFT")
    print("3. AMZN")
    print("4. JNJ")
    print("5. GOOGL")
    userInput = int(input())

    # ----CLEANING UP/PREPPING DATA------#
    print("Fetching data for " + stocks[userInput] + "\n")
    data = pd.read_csv('Stonks/' + stocks[userInput] + ".csv")

    data['Price'] = (data['Open'] + data['Close'] + data['High'] + data['Low']) / 4  # Take the avg of all of these
    data.drop(['Open', 'Close', 'High', 'Low', 'Volume', 'Dividends',
               'Stock Splits'], axis=1, inplace=True)  # Drop unwanted columns

    data['Day'] = range(1,
                        len(data) + 1)  # Create a 'Day' column numbered 1-n (where n is len of csv) | This is to train

    # ------TRAINING-----

    # Linear Regression
    print("Training Models...")
    m, b, L, epochs = 0, 0, 0.000000001, 10
    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)

    print("Linear Regression Fit completed")

    # ------Polynomial Regression----------
    # Extract X and Y columns
    X = data['Day'].values
    Y = data['Price'].values

    # Polynomial degree
    degree = 3  # Change this to the desired degree of the polynomial

    # Transforming features to polynomial features
    X_poly = create_polynomial_features(X, degree)
    coefficients = normal_equation(X_poly, Y)


    # Predicting on training data
    Y_pred = predict(X_poly, coefficients)

    print("Polynomial Regression Fit completed")

    # ----Use Model to predict -----------#
    x_pred = int(input("How many days from " + data['Date'][0] + " would you like to predict the price of " + stocks[
        userInput] + "? "))
    y_pred = ((m * x_pred) + b) #basic linear formula y = mx + b

    # Transform the given X value into polynomial features
    given_X_poly = create_polynomial_features_for_single_X(x_pred, degree)

    # Predict the Y value for the given X
    predicted_Y = predict_Y_for_given_X(given_X_poly, coefficients)


    print(f"\nLinear Regression: ${y_pred} ")
    print(f"Polynomial Regression: ${predicted_Y[0]} \n")


    # ---------Displaying Graph------------------------#
    plt.scatter(data.Day, data.Price, color='black')
    plt.plot(list(range(0, len(data))), [m * x + b for x in range(0, len(data))], color='red')
    plt.plot(X, Y_pred, color='red', label='Polynomial Regression')
    plt.xlabel('# of days since ' + data['Date'][0])
    plt.ylabel('AVG Price')
    plt.title(stocks[userInput])
    plt.show()
    # note that in this graph the y-axis is the # of days since the first stock, and the x-axis is the avg price
    # also note that linear regression sucks at predicting stock prices lol

    #note that you must close the graph pop-up window in order to continue running past plt.show()
    userInput = input("Would you like to predict another stock? \n1. Yes\n2. No\n")
    if userInput == "2":
        run = False












