import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

data = pd.read_csv(r"C:\Users\Mischa de Gooijer\OneDrive\Python\Python Projects\TIL_Kek\T5\cf_data.csv")

dv = np.linspace(-10, 10, 41)
s = np.linspace(0, 200, 21)
a = np.zeros((21, 41))

DV = data.dv.to_numpy()
S = data.s.to_numpy()
A = data.a.to_numpy()

#create algorithm that calculates all acceleration values and stores them in the grid
#at each grid point, it calculates a weighted mean of all measurements. The weights are based on the distance to the grid point
#the weight function is given by math.exp(-(abs(distance_dv)/upsilon) - (abs(distance_s)/sigma))
#the algorithm is given by the following formula: a = sum(w_i * a_i) / sum(w_i)
#where w_i is the weight of the i-th measurement and a_i is the acceleration of the i-th measurement
#the algorithm is implemented in the following way:
#1. loop over all grid points
#2. loop over all measurements
#3. calculate the weight of the measurement
#4. calculate the weighted acceleration
#5. add the weighted acceleration to the total weighted acceleration
#6. add the weight to the total weight
#7. divide the total weighted acceleration by the total weight
#8. store the weighted acceleration in the grid

for i in range(0, 21):
    for j in range(0, 41):
        total_weighted_acceleration = 0
        total_weight = 0
        for k in range(0, 50):
            distance_dv = DV[k] - dv[j]
            distance_s = S[k] - s[i]
            weight = math.exp(-(abs(distance_dv)/0.5) - (abs(distance_s)/10))
            weighted_acceleration = weight * A[k]
            total_weighted_acceleration += weighted_acceleration
            total_weight += weight
        a[i][j] = total_weighted_acceleration / total_weight

#plot the grid
#%%
X, Y = np.meshgrid(dv, s)
axs = plt.axes()
p = axs.pcolor(X, Y, a, shading='nearest')
axs.set_title('Acceleration [m/s/s]')
axs.set_xlabel('Speed difference [m/s]')
axs.set_ylabel('Headway [m]')
axs.figure.colorbar(p);
axs.figure.set_size_inches(10, 7)