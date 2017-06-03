import numpy as np
import matplotlib.pyplot as plt

def delta_x(x):
    if x == 0:
        return 1e+3
    else:
        return 0.

def solve_poisson():
    phi_x = []
    sum_old = 1e+5

    N = 128
    L = 2
    dx = L/N
    for i in range(N + 1):
        phi_x.append([-L/2. + dx * i, 0.])

    for count in range(100000):
        sum = 0
        for i in range(1,N):
            phi_x[i][1] = (phi_x[i + 1][1] + phi_x[i - 1][1])/2. - dx*dx/2. * delta_x(phi_x[i][0])
        for i in range(N + 1):
            sum += phi_x[i][1]
        if abs(sum_old - sum) < 0.001:
            break
        else:
            sum_old = sum
    x = []
    y = []
    for i in range(len(phi_x)):
        x.append(phi_x[i][0])
        y.append(phi_x[i][1])
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    solve_poisson()
