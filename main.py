import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('AAPL.csv')
data['Price'] = (data['Open'] + data['Close'] + data['High'] + data['Low'])/4 #Take the avg of all of these
data['Price'] = data['Price'].apply(lambda x: x * 100) #Multiply price by 100 so linear regression algo dosent break (dont know why it wont work without this)
data.drop(['Open','Close','High','Low','Volume','Dividends',
           'Stock Splits'],axis = 1,inplace = True) #Drop unwanted columns

data['Index'] = range(1, len(data) + 1) #Create a 'index' column numbered 1-n (where n is len of csv) | This is to train
print(data)

def loss_function(m, b, data): #basic loss function to calculate Error (not used, can be deleted)
    total_error = 0
    for i in range(len(data)):
        x = data.iloc[i].Date
        y = data.iloc[i].Price
        total_error += (y- (m * x + b))**2
    total_error / float(len(data))

def gradient_descent(m_now, b_now, data, L): #Gradient Descent function
    m_gradient = 0
    b_gradient = 0

    n = len(data)
    for i in range(n):
        x = data.iloc[i].Index
        y = data.iloc[i].Price

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m,b

m,b,L,iter = 0,0,0.00000001,20

for i in range(10):
    print(i)
    m,b = gradient_descent(m,b,data,L)

print(m,b)

plt.scatter(data.Index,data.Price,color = 'black')
plt.plot(list(range(0,len(data))), [m * x + b for x in range(0,len(data))], color='red')
plt.ylim(0,4000)
plt.show()








