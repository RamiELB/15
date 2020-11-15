import numpy as np
import matplotlib.pyplot as plt

# Ex. 1.1.
def p(x, λ=1):
    return np.exp(-x/λ)/λ

x = np.arange(0, 10, 0.1)
λ = 2

f, ax = plt.subplots(1, 2)
ax[0].plot(x, p(x, λ=λ))
ax[1].semilogy(x, p(x, λ=λ))

# Ex. 1.2.
n = 10000
λ = 2

samples = np.random.exponential(λ, size=n)

μ = np.mean(samples)
σ = np.std(samples)
ϵ = σ/np.sqrt(n)

print("Moyenne = ", μ)
print("Variance = ", σ**2)
print("Erreur = ", ϵ)

# Ex. 1.3.
r = 10000 # N_t repetitions
n = 10000 # N_s samples

λ = 2

def experience(n, λ = 1):
    x = np.random.exponential(λ, size=n)

    return np.mean(x), np.min(x), np.max(x)

mins = []
maxs = []
means = []

for i in range(r):
    μ, minimum, maximum = experience(n, λ)

    mins.append(minimum)
    maxs.append(maximum)
    means.append(μ)

f,ax = plt.subplots(1,3,figsize=(10,5))
ax[0].hist(means, bins=80, density=True);
ax[1].hist(mins, bins=80, density=True);
ax[2].hist(maxs, bins=80, density=True);

# Ex. 1.4.
S = np.sum(means)
σ = λ

Z = np.sqrt(n) * (S/n - means) / σ
plt.figure()
plt.hist(Z, density = True)

# Ex. 1.5.
f = plt.figure()
plt.hist(mins, bins=80, density=True, log=True)

# Ex. 1.6.
f = plt.figure()
x = np.arange(0, 0.002, 0.0001)
plt.hist(mins, bins=80, density=True, log=True)
plt.plot(x, p(x, λ=np.mean(mins)))

# Ex. 1.7.
def gaussian(x, μ=0, σ=1):
    return 1/np.sqrt(2*np.pi*σ)*np.exp( -(x-μ)**2 / (2*σ**2))

def normalize(x):
    μ = np.mean(x)
    σ = np.std(x)

    return (x - μ) / σ

x = np.arange(-3.5, 3.5, .05)

plt.figure()
plt.hist(normalize(maxs), bins=80, density=True, log=True)
plt.hist(normalize(means), bins=80, density=True, log=True)
plt.plot(x, gaussian(x), 'r')

# 1. On voit que la distribution du minimum suit une loi exponentielle (aucune valeur négative, une pente linéaire en échelle log-lin). On peut donc ajuster la pdf d'une loi exponentielle en ajuste la paramètre $\lambda$ selon la moyenne empirique.
# 2. On voit que la distribution du maximum suit une distribution inconnue. La queue de distribution des grandes valeurs de $x$ semble suivre une décroissance exponentielle (comportement linéaire). La queue pour des petites valeurs de $x$ semble décroître très rapidement.
# 3. On identifie parfaitement la distribution gaussienne, on peut d'ailleurs tracer la pdf en surimpression.
#
# Conclusion : la distributions des valeurs extrêmes est très différentes de celle de la moyenne. On peut caractériser (ici) celle du minimum de façon empirique. Celle du maximum est plus complexe. On voit un décroissance plus lente que la gaussienne pour les $x$ grands et beaucoup plus rapide pour les $x$ petits

# Ex. 2.
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

import urllib.request
local_filename, headers = urllib.request.urlretrieve(
    "http://deeplearning.net/data/mnist/mnist.pkl.gz",
    "mnist.pkl.gz")
html = open(local_filename)
html.close()

f = gzip.open('mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
mnist = u.load()
train_set, valid_set, test_set = mnist

# Ex. 2.1.1.
X3 = train_set[0][np.where(train_set[1]==3)].T
X6 = train_set[0][np.where(train_set[1]==6)].T
X = np.concatenate((X3,X6),1)
X = np.random.permutation(X.T).T

U, s, V = np.linalg.svd(X/np.sqrt(X.shape[1]))

n = 5
f,ax = plt.subplots(n, n, figsize=(15, 10))

for i in range(n):
    for j in range(n):
        ax[i, j].imshow(U[:, i+j].reshape(28,28))

# Ex. 2.1.2.
P3 = X3.T.dot(U)
P6 = X6.T.dot(U)

# Ex. 2.1.3.
f,ax = plt.subplots(1, 3, figsize=(15,10))
ax[0].scatter(P3[:,0], P3[:,1])
ax[0].scatter(P6[:,0], P6[:,1])
ax[1].scatter(P3[:,2], P3[:,3])
ax[1].scatter(P6[:,2], P6[:,3])
ax[2].scatter(P3[:,4], P3[:,5])
ax[2].scatter(P6[:,4], P6[:,5])
plt.show()

# Ex. 2.1.4 - 2.1.5.
# On voit que la direction 2 (axe des ordonnées de la figure 1) permettrait de faire une séparation raisonnable.
n3 = X3.shape[1]
n6 = X6.shape[1]
class3 = np.where(P3[:,1] > 0)[0]
class6 = np.where(P6[:,1] < 0)[0]
print("Fraction de bonnes classif des 3 = ", len(class3)/n3)
print("Fraction de bonnes classif des 6 = ", len(class6)/n6)

# Ex 2.2.

def MSE(x, y):
    return np.sqrt(np.mean(np.square(x-y)))

def reconstruction(x, base, k):
    proj = x.T @ base[:, :k]
    return proj @ base[:, :k].T

def get_pixel_space(X):
    U, S, V = np.linalg.svd(X/np.sqrt(X.shape[1]))

    return U

def error_vs_k(base, samples, k):
    idx = np.random.randint(samples.shape[1])
    sample = samples[:, idx]
    return [MSE(sample, reconstruction(sample, base, k)) for k in range(1, k)]


base3 = get_pixel_space(X3)
errors_3 = error_vs_k(base3, X3, 100)
base6 = get_pixel_space(X6)
errors_6 = error_vs_k(base6, X6, 100)
plt.plot(errors_3)
plt.plot(errors_6)

plt.figure()
for i in range(10):
    Xi = train_set[0][np.where(train_set[1] == i)].T

    errors_3 = error_vs_k(base3, Xi, 100)
    plt.plot(errors_3, label=i)
plt.legend()

plt.figure()
for i in range(10):
    Xi = train_set[0][np.where(train_set[1] == i)].T

    errors_6 = error_vs_k(base6, Xi, 100)
    plt.plot(errors_6, label=i)
plt.legend()
