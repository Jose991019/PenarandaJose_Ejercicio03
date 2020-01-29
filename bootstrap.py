import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

Betas = []
data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]
regresion = sklearn.linear_model.LinearRegression()
regresion.fit(X, Y)
Betas.append(regresion.coef_)
for i in range(1000):
    aleatorio = np.random.randint(len(X), size = 69)
    x = X[aleatorio]
    y = Y[aleatorio]
    regresion = sklearn.linear_model.LinearRegression()
    regresion.fit(x, y)
    Betas.append(regresion.coef_)
    
Betas = np.array(Betas)
plt.figure(figsize = (10,10))
for i in range(4):
    graficar = Betas[:,i]
    promedio = np.mean(graficar)
    std = np.std(graficar)
    plt.subplot(2,2,i+1)
    plt.title('Beta {}, promedio = {:.2f}, std = {:.2f}'.format(i,promedio,std))
    plt.hist(graficar, bins = 60)
    plt.grid(True)
plt.savefig("bootstrap.png")