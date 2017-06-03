import numpy as np
import matplotlib.pyplot as plt

class GaussSeidelMethod:
    LENGTH = 1.0  # System length
    logN = 7
    N = 1 << logN
    DX = LENGTH / N
    EPS = 1e-5
    MAX_ITERATION = 100000


    def __init__(self):
        self.phi = np.zeros(self.N + 1)
        self.rho = np.zeros(self.N + 1)
        # self.rho[int(self.N/2)] = 1./self.DX
        self.rho[int(self.N / 2)] = 1000
        self.x = [-self.LENGTH / 2 + self.DX * i for i in range(self.N + 1)]

    def relative_err(self, a, b):
        return (sum((a - b) ** 2) / (sum(a ** 2) + sum(b ** 2) + 1e-10)) ** 0.5

    def gauss_seidel(self, phi, rho):
        n = len(phi) - 1
        dx = self.LENGTH / n
        new_phi = np.zeros_like(phi)
        for i in range(1, n):
            new_phi[i] = (phi[i + 1] + phi[i - 1]) / 2. - dx * dx * rho[i] / 2.
        return new_phi

    def solve_GS(self, phi, rho):
        for i in range(self.MAX_ITERATION):
            new_phi = self.gauss_seidel(phi, rho)
            if self.relative_err(phi, new_phi) < self.EPS:
                print('break')
                break
            else:
                phi = new_phi
        return new_phi

    def run(self):
        self.phi = self.solve_GS(self.phi, self.rho)

    def plot(self):
        plt.plot(self.x, self.phi)
        plt.show()


class MultiGridMethod(GaussSeidelMethod):
    PRE_SMOOTH_ite = 5
    POST_SMOOTH_ite = 5
    MU = 1  # number of cycle in multigrid



if __name__ == '__main__':
    # gs = GaussSeidelMethod()
    # gs.run()
    # gs.plot()

    mg = MultiGridMethod()
    print(mg.MU)