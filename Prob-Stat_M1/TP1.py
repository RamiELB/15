# import
import numpy as np
import matplotlib.pyplot as pp

# Générer des nombres aléatoires uniformes [0,1]
va = np.random.random((10000))
print(va)

# faire un histogramme
def get_idx(xi, x0, δ):
    return int((xi-x0) // δ)

def histogram(x, x0, x1, nbins):
    # TODO renvoyer l'histogramme et la norme
    δ = (x1-x0)/nbins
    hist = np.zeros(nbins)

    for xi in x:
        idx = get_idx(xi, x0, δ)

        if ((idx < 0) or (idx >= nbins)):
            print("Range error!")
            break;

        hist[idx] = hist[idx] + 1

    norm = sum(hi*δ for hi in hist)

    return hist,norm

h1,norm = histogram(va*10,0,10,10)
print(h1)
print(h1/norm)


# variables gaussiennes
va_normal = np.random.normal(size=(100000))

# Tester la fonction pour des variables gaussiennes
x0 = min(va_normal)
x1 = max(va_normal)
nbins = 100

δ = (x1-x0) / nbins
xr = np.arange(x0, x1, δ)

h2, norm2 = histogram(va_normal, x0, x1, nbins)

#normalisation
def gauss(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2.0)

pp.plot(xr,h2/norm)
pp.plot(xr,gauss(xr))

# numpy sait le faire
# vérifier la fonction de répartition : faire un histogramme
f,ax = pp.subplots(1,3)
ax[0].hist(va, range=(0,1), bins=10)
ax[1].hist(va_normal,bins=50)
ax[2].hist(va_normal,bins=50,density=True)
ax[2].plot(xr,gauss(xr))


# p(x,y) = p(x|y)p(y)
# p(y) ~ unif(-1,1)
# p(x|y) = Normal(-5,1)\theta(y) + Normal(5,1)\theta(-y)

va_y = np.random.random(size=(100000))*2-1
va_x = np.zeros(va_y.shape[0])
yneg = np.where(va_y < 0)
ypos = np.where(va_y >= 0)
va_x[yneg[0]] = np.random.normal(-5,size=(len(yneg[0])))
va_x[ypos[0]] = np.random.normal(5,size=(len(ypos[0])))

pp.scatter(va_x,va_y)

# plotter un histogramme 2D (cf np.histogram2D)
fig = pp.figure(figsize=(10, 5))
pp.hist2d(va_x,va_y,bins=100,density=True)
pp.colorbar(orientation='horizontal')

# Loi des gds nombres et convergence
# commençons par des variables de bernoulli
N = 10**np.arange(1,9,1)
N

def bernoulli(n):
    """Returns an array of 0/1 Bernoulli values (p=1/2)"""
    # note: `np.random.randint(2)` returns 0 or 1 with equal prob
    return np.random.randint(2, size=n)

def empirical_mean(n):
    realizations = [bernoulli(n)]
    return np.mean(realizations)

empirical_means = np.array([empirical_mean(n) for n in N])
empirical_means

pp.plot(empirical_means-0.5)

# somme de variables aléatoire
n_realizations = 10000 # nombre d'estimation de la valeur moyenne
N = 10 # nombre de variable aléatoire utilisé pour la somme

# générer n_var estimation de la valeur moyenne d'un processus de n_sum variables bernoulli sommé
sums = np.array([np.sum(bernoulli(N)) for _ in range(n_realizations)])

# test
Z = np.sqrt(N)*(sums/N - 0.5)/0.5;

pp.hist(Z, bins=40, density=True)
pp.plot(xr, gauss(xr))
