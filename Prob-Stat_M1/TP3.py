import numpy as np
import matplotlib.pyplot as plt

# # Variance/Covariance
# On revient sur une distribution bien connu.
# Générer $10000$ variables aléatoires gaussiennes de moyenne $0$ et de variance $1$.
#   * Tracer l'histogramme des points, et données (à l'oeil) une estimation de la variance
#   * Changer la variance et observer si la valeur correspond bien à ce que l'on observe sur l'histogramme
# on génère des nombres gaussiens

f,ax = plt.subplots(1,3,figsize=(15,5))
x = np.random.normal(loc=0,scale=1,size=(10000))

ax[0].hist(x,bins=80,density=True)
x = np.random.normal(loc=0,scale=4,size=(10000))

ax[1].hist(x,bins=80,density=True)
x = np.random.normal(loc=0,scale=10,size=(10000))

ax[2].hist(x,bins=80,density=True)
plt.show()

# On va voir maintenant ce qu'il se passe en dimension $d=2$. Utiliser la fonction suivante pour générer des points en deux dimensions
covar = [[5,0],[0,1]]
m = [0,0]
x = np.random.multivariate_normal(mean=m,cov=covar,size=(100000))

#   * tracez l'histogramme 2D (cf le tp précédent) et regardez si vous réussissez à observer l'ordre de grandeur de la variance sur le plot
#   * mettez des valeurs differents pour la variance selon x et selon y, que constatez-vous sur l'histogramme ?
x = np.random.multivariate_normal(mean=[0,0],cov = [[5, 0], [0, 1]],size=(100000))
print(x.shape)
plt.hist2d(x[:,0],x[:,1],bins=100)
plt.show()


# En utilisant la matrice de covariance suivante :
# ```python
# cov = [[5, 1], [1, 1]]
# ```
# Regarder l'histogramme obtenu.
cov = [[5, 1], [1, 1]]
x = np.random.multivariate_normal(mean=[0,0],cov = [[5, 1], [1, 1]],size=(100000))
plt.hist2d(x[:,0],x[:,1],bins=100)
plt.show()

#   * Définir un vecteur unitaire $\vec{u}$ aléatoirement, et calculer la variance le long de ce vecteur

Ns = x.shape[0]
# Vecteur aléatoire
u = 2*np.random.random(size=(2))-1
# Normalisation
u = u/np.linalg.norm(u)
print("u = ",u)

# D'abord on calcule la moyenne
m = np.mean(x,axis=0)
print("mean = ",m)
# puis on la projette le long de u
m_u = m.dot(u)
print("mean_u = ",m_u)

# Variance le long de u
Var_u = np.mean((np.matmul(u,x.T) - m_u)**2)
print("Var_u = ",Var_u)

#   * Calculer l'erreur de reconstruction le long de cette direction
Proj = x - np.matmul(np.matmul(u,x.T).reshape(Ns,1),u.reshape(1,2))
sumn = 0
for i in Proj:
    sumn += np.linalg.norm(i)
print("erreur = ", sumn/Ns)

#   * En observant l'histogramme tracé précédemment, essayer de construire un vecteur $\vec{u}$ qui suivrait la forme du nuage de points.
# On choisit une direction plus intelligemment

Vdir = np.array([1,cov[0][1]]) # np.sqrt(VV[0]*VV[1])])
Vdir
Vdir = Vdir/np.linalg.norm(Vdir)
Vdir

# on montre la direction
plt.hist2d(x[:,0],x[:,1],bins=100);
myx = np.arange(-10,10,0.5)

myy = myx/5 #*cov[0][1]/np.sqrt(cov[0][0]*cov[1][1])
plt.plot(myx,myy,color='red')

#   * Calculer l'erreur de reconstruction le long de cette direction


# Calcul de la variance
m = np.mean(x,0)
m_u = m.dot(Vdir)
Var_u = np.mean((np.matmul(Vdir,x.T) - m_u)**2)
print("Var_u = ",Var_u)

# Calcul de l'erreur
Proj = x - np.matmul(np.matmul(Vdir,x.T).reshape(Ns,1),Vdir.reshape(1,2))
sumn = 0
for i in Proj:
    sumn += np.linalg.norm(i)
print("erreur = ", sumn/Ns)
