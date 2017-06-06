import numpy as np
import matplotlib.pyplot as plt
import time

class GaussSeidelMethod:
    LENGTH = 1.0  # System length
    LOGN = 7
    N = 1 << LOGN
    DX = LENGTH / N
    EPS = 1e-5
    MAX_ITERATION = 100000


    def __init__(self):
        self.phi = np.zeros(self.N + 1)
        self.rho = np.zeros(self.N + 1)
        self.rho[int(self.N/2)] = 1./self.DX
        self.x = [-self.LENGTH / 2 + self.DX * i for i in range(self.N + 1)]
        self.ans = (np.abs(self.x)-0.5)/2              # answer


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
                break
            else:
                phi = new_phi
        return new_phi

    def run(self):
        start_time = time.time()
        for i in range (self.MAX_ITERATION):
            self.phi = self.solve_GS(self.phi, self.rho)
            if self.relative_err(self.phi, self.ans) < self.EPS:
                break
        calc_time = time.time() - start_time
        print("GS:", calc_time, '(s)')

    def plot(self):
        plt.plot(self.x, self.phi)
        plt.show()


class MultiGridMethod(GaussSeidelMethod):
    PRE_SMOOTH = 5
    POST_SMOOTH = 5
    MU = 1  # number of cycle in multigrid
    LOGN = 7
    LEVEL = LOGN - 2 #gridの数が4が最小
    pre = 5


    def restrict(self, phi):
        harf_n = int((len(phi)-1)/2)
        harf_phi = np.zeros(harf_n + 1)
        for i in range(1, harf_n):
            harf_phi[i] = (phi[2*i - 1]+phi[2*i + 1])/4 + phi[2*i]/2
        return  harf_phi

    def prolong(self, harf_phi):
        n = (len(harf_phi) - 1)*2
        phi = np.zeros(n + 1)
        for i in range(1,n):
            if (i%2==0) : phi[i] = harf_phi[int(i/2)]
            if (i%2==1) : phi[i] = (harf_phi[int(i/2)] + harf_phi[int(i/2+1)])/2
        return phi

    def residue(self,phi, rho):
        n = len(rho) - 1
        dx = self.LENGTH/n
        res = np.zeros_like(rho)
        for i in range(1, n):
            res[i] = rho[i] - ((phi[i - 1] - 2*phi[i] + phi[i + 1])/(dx*dx))
        return res

    def multiGrid(self, level, phi, rho):
        if level == 0:
            phi = self.solve_GS(phi, rho)
        else:
            for i in range(self.PRE_SMOOTH):
                phi = self.gauss_seidel(phi,rho)
            res = self.restrict(self.residue(phi,rho))
            dphi = np.zeros_like(res)
            for i in range(self.MU):
                dphi = self.multiGrid(level-1, phi=dphi, rho=res)
            phi += self.prolong(dphi)
            for i in range(self.POST_SMOOTH):
                phi = self.gauss_seidel(phi, rho)
        return phi

    def solve_MG(self, phi, rho):
        for i in range(self.MAX_ITERATION):
            new_phi = self.multiGrid(level=self.LEVEL, phi=phi, rho=rho)
            err = self.relative_err(phi, new_phi)
            if err < self.EPS:
                break
            else:
                phi = new_phi
        return new_phi

    def run(self): #override
        start_time = time.time()
        self.phi = self.solve_MG(self.phi, self.rho)
        calc_time = time.time() - start_time
        print("MG: ", calc_time, "(s)")

class MultiGridMethod_v2(MultiGridMethod):
    def restrict(self, phi): #override
        harf_n = int((len(phi) - 1)/2)
        harf_phi = np.zeros(harf_n + 1)
        for i in range(1, harf_n):
            harf_phi[i] = phi[2*i]
        return harf_phi

    def restrict_rho(self,rho):
        n = int((len(rho) - 1)/2)
        rho_r = np.zeros(n + 1)
        dx = self.LENGTH/n
        rho_r[int(n/2)] = 1./dx
        return rho_r

    def multiGrid(self, level, phi, rho): # override
        if level == 0:
            phi = self.solve_GS(phi, rho)
        else:
            for i in range(self.PRE_SMOOTH):
                phi = self.gauss_seidel(phi,rho)
            phi_r = self.restrict(phi)
            rho_r = self.restrict_rho(rho)
            for i in range(self.MU):
                phi += self.prolong(self.multiGrid(level-1, phi=phi_r, rho=rho_r) - phi_r)
            for i in range(self.POST_SMOOTH):
                phi = self.gauss_seidel(phi, rho)
        return phi

    def solve_MG2(self, phi, rho):
        for i in range(self.MAX_ITERATION):
            new_phi = self.multiGrid(level=self.LEVEL, phi=phi, rho=rho)
            err = self.relative_err(phi, new_phi)
            if err < self.EPS:
                break
            else:
                phi = new_phi
        return new_phi

    def run(self): #override
        start_time = time.time()
        self.phi = self.solve_MG2(self.phi, self.rho)
        calc_time = time.time() - start_time
        print("MG2: ", calc_time, "(s)")


if __name__ == '__main__':
    gs = GaussSeidelMethod()
    gs.run()
    # gs.plot()

    mg = MultiGridMethod()
    mg.run()
    # mg.plot()

    mg2 = MultiGridMethod_v2()
    mg2.run()
    # mg2.plot()
