import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    return m,b

#Get user input for which stock to predict
stocks = {1: "AAPL",
          2: "MSFT",
          3: "AMZN",
          4: "NVDA",
          5: "GOOGL"}
print("Select what stock to predict")
print("1. AAPL")
print("2. MSFT")
print("3. AMZN")
print("4. NVDA")
print("5. GOOGL")
userInput = int(input())





#----CLEANING UP/PREPPING DATA------#
print("Fetching data for " + stocks[userInput] + "\n")
data = pd.read_csv('Stonks/' + stocks[userInput] + ".csv")




data['Price'] = (data['Open'] + data['Close'] + data['High'] + data['Low'])/4 #Take the avg of all of these
data.drop(['Open','Close','High','Low','Volume','Dividends',
           'Stock Splits'],axis = 1,inplace = True) #Drop unwanted columns

data['Day'] = range(1, len(data) + 1) #Create a 'Day' column numbered 1-n (where n is len of csv) | This is to train




#------TRAINING-----

#Linear Regression
print("Training Linear Regression Fit with 10 epochs and a Learning Rate of 0.00001")
m,b,L,epochs = 0,0,0.000000001,10
for i in range(epochs):
    print(i)
    m,b = gradient_descent(m,b,data,L)


print("Linear Regression Fit completed \n")

#------Polynomial Regression----------
# Extract X and Y columns
X = data['Day'].values.reshape(-1, 1)
Y = data['Price'].values

# Polynomial degree
degree = 2  # Change this to the desired degree of the polynomial

# Transforming features to polynomial features
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Fitting polynomial features into a linear regression model
model = LinearRegression()
model.fit(X_poly, Y)

# Predicting on training data
Y_pred = model.predict(X_poly)



#----Use Model to predict -----------#
x_pred = int(input("How many days from " + data['Date'][0] + " would you like to predict the price of " + stocks[userInput] + "? "))
y_pred =  ((m*x_pred) + b)

new_X = np.array([[x_pred]])  # Reshape the value as a 2D array to match the input format
new_X_poly = poly.transform(new_X)  # Transform the new X value into polynomial features

# Use the fitted polynomial model to predict Y for the new X value
predicted_Y = model.predict(new_X_poly)

print("\nLinear Regression: ", y_pred)
print(f"Polynomial Regression : {predicted_Y[0]}" )

#---------Displaying Graph------------------------#
plt.scatter(data.Day,data.Price,color = 'black')
plt.plot(list(range(0,len(data))), [m * x + b for x in range(0,len(data))], color='red')
plt.plot(X, Y_pred, color='red', label='Polynomial Regression')
plt.xlabel('# of days since ' + data['Date'][0])
plt.ylabel('AVG Price')
plt.show()
#note that in this graph the y-axis is the # of days since the first stock, and the x-axis is the avg price
#also note that linear regression sucks at predicting stock prices lol












