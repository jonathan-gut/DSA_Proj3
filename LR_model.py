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