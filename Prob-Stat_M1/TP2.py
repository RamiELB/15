# ## Estimation de la moyenne, erreur et variance/covariance
# Le but de ce TP est d'un part d'observer dans des exemples simples les estimations de la valeur moyenne (lorsque l'on connnaît la vérité terrain) et de voir comment mesurer l'erreur sur celle-ci.
# Dans un second temps, on regardera (toujours sur des cas contrôlés) la signification de la covariance et une façon de réduite naïvement la dimension des données.

import numpy as np
import matplotlib.pyplot as plt

#   * Générer $N_s = 10^{2,3,4,5,6,7}$ variables Gaussienns de moyennes 0 et de variance 2.
Ns = np.array(10**np.arange(2,8))
print('Ns =', Ns)

mean = 0
variance = 10

samples = [np.random.normal(mean, np.sqrt(variance), size=n) for n in Ns]

#   * Afficher pour chacune des valeurs l'estimation de la valeur moyenne et de la variance.
empirical_means = [np.mean(s) for s in samples]
empirical_variances = [np.var(s) for s in samples]

print("Moyennes: ", empirical_means)
print("Variances: ", empirical_variances)





#   * Comment calcule-t-on l'erreur sur l'estimation de la moyenne ?
errors = [np.sqrt(var/n) for var, n in zip(empirical_variances, Ns)]
print("Erreurs: ", errors)

#   * Vous afficherez les résultats sur un plot en échelle logarithmique pour l'axe x (pourquoi l'échelle log ?).
#   * A l'aide de la fonction 'errorbar' faire un graphe avec la barre d'erreur.
#   * Que se passe-t-il si la variance n'est pas 2 mais 10
plt.axhline(y=0, c='red', label='0')
plt.errorbar(Ns, empirical_means, yerr=errors, fmt='--o', label='Mean estimate')
plt.xscale('log')
plt.ylabel('estimate')
plt.xlabel('N samples')
plt.legend()
plt.show()

plt.loglog(Ns, errors, '--o')
plt.title("Erreur vs Nb Samples")
plt.ylabel('N samples')
plt.xlabel('error')
plt.show()


# Dans le fichier "data_ping.d", sont enregistrés les données du temps de ping vers un serveur.
#   * Tracer l'histogramme des valeurs, que constatez-vous ?
data = np.genfromtxt("data_ping.d")

import seaborn as sns
sns.distplot(data)
plt.show()

plt.hist(data, bins=30, density=True)
plt.show()

#comportement exponentiel
x_range = np.arange(9,35,0.5)
plt.hist(data, bins=30, log=True, density=True)
plt.plot(x_range, np.exp( - (x_range - 11.0) / 3 ) / 5)
plt.show()

#   * Estimez la valeur moyenne et son erreur
estimate_mean = np.mean(data)
n_samples = data.shape[0]
estimate_mean_error = np.sqrt(np.var(data)/n_samples)
print("estimate mean = {} +/- {}".format(estimate_mean, estimate_mean_error))

# A l'aide de la fonction "np.random.permutation":
#   * obtenez $200$ estimations (dans un tableau) de la valeur moyenne en prenant $30$ valeurs aléatoire parmis le jeu de données.
n_realizations = 2000
N = 30

permutations = [np.random.choice(n_samples, size=N, replace=False) for _ in range(n_realizations)]
estimated_means = np.array([np.mean(data[idxs]) for idxs in permutations])
estimated_means

#   * Faites l'histogramme des valeurs obtenues.
sns.distplot(estimated_means)
plt.title('estimated mean distribution')
plt.show()

#   * En utilisant l'histogramme, pouvez-vous devinez la distribution obtenue ? (vous pouvez aussi tracer l'histogramme avec l'option "log=True" pour mettre l'échelle y en log)
plt.hist(estimated_means, 35)
plt.title('estimated mean distribution')
plt.show()

plt.hist(estimated_means, 35, log=True)
plt.title('estimated mean distribution')
plt.show()

#   * Tracer la distribution intuitée pour s'en assurer, que remarquez-vous ?

# on definit la pdf
def gauss(x, m, s):
    return 1/np.sqrt(2*np.pi*s) * np.exp(-(x-m)**2 / (2.0*s) )

myx = np.arange(11, 14.5, 0.01)
mu, beta = 12.6, 0.4
count, bins, ignored = plt.hist(estimated_means, bins=40, density=True, log=True, label='measured')
plt.ylim(0.01, 1)
plt.semilogy(myx, gauss(myx, np.mean(estimated_means), np.std(estimated_means)), label='gauss')
plt.title('estimated mean distribution')
plt.legend()
plt.show()
