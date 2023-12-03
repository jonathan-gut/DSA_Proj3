import pandas as pd
import matplotlib.pyplot as plt
from LR_model import gradient_descent

#----CLEANING UP/PREPPING DATA------#
data = pd.read_csv('AAPL.csv')
data['Price'] = (data['Open'] + data['Close'] + data['High'] + data['Low'])/4 #Take the avg of all of these
data.drop(['Open','Close','High','Low','Volume','Dividends',
           'Stock Splits'],axis = 1,inplace = True) #Drop unwanted columns

data['Index'] = range(1, len(data) + 1) #Create a 'index' column numbered 1-n (where n is len of csv) | This is to train



#------TRAINING(aka finding the best m and b that fits the data)-----
m,b,L,epochs = 0,0,0.000000001,10
for i in range(epochs):
    print(i)
    m,b = gradient_descent(m,b,data,L)

#----Get user input
x_pred = int(input("How many days from "+ data['Date'][0] + " would you like to predict the price of AAPL?"))
y_pred =  ((m*x_pred) + b)/1 #divide by 100 since we multiplied price by 100 at the start
print("Predicted price: ", y_pred)

#---------Displaying Graph------------------------#
plt.scatter(data.Index,data.Price,color = 'black')
plt.plot(list(range(0,len(data))), [m * x + b for x in range(0,len(data))], color='red')
plt.xlabel('# of days since ' + data['Date'][0])
plt.ylabel('AVG Price')
plt.show()
#note that in this graph the y-axis is the # of days since the first stock, and the x-axis is the avg price*100
#also note that linear regression sucks at predicting stock prices lol










