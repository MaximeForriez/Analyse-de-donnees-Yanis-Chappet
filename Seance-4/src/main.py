#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats 
import matplotlib.pyplot as plt




#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)

def plot_discrete_distributions():
    # Paramètres pour les distributions
    a = 0
    n = 10
    n_binomial, p_binomial = 20, 0.5
    mu_poisson = 5
    a_zipf = 1.5

    # Loi de Dirac
    x_dirac = [a]
    y_dirac = [1]
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.stem(x_dirac, y_dirac)
    plt.title('Loi de Dirac')

    # Loi uniforme discrète
    x_uniform = np.arange(1, n + 1)
    y_uniform = np.ones(n) / n
    plt.subplot(2, 3, 2)
    plt.stem(x_uniform, y_uniform)
    plt.title('Loi uniforme discrète')

    # Loi binomiale
    x_binomial = np.arange(n_binomial + 1)
    y_binomial = stats.binom.pmf(x_binomial, n_binomial, p_binomial)
    plt.subplot(2, 3, 3)
    plt.stem(x_binomial, y_binomial)
    plt.title('Loi binomiale')

    # Loi de Poisson
    x_poisson = np.arange(0, 15)
    y_poisson = stats.poisson.pmf(x_poisson, mu_poisson)
    plt.subplot(2, 3, 4)
    plt.stem(x_poisson, y_poisson)
    plt.title('Loi de Poisson')

    # Loi de Zipf-Mandelbrot
    x_zipf = np.arange(1, 11)
    y_zipf = 1 / (x_zipf ** a_zipf)
    y_zipf /= y_zipf.sum()
    plt.subplot(2, 3, 5)
    plt.stem(x_zipf, y_zipf)
    plt.title('Loi de Zipf-Mandelbrot')

    plt.tight_layout()
    plt.show()

def plot_continuous_distributions():
    # Paramètres pour les distributions
    mu_poisson_cont, size_poisson_cont = 5, 1000
    mu_normal, sigma_normal = 0, 1
    sigma_lognormal, mu_lognormal, size_lognormal = 0.5, 0, 1000
    a_uniform, b_uniform = 0, 1
    df_chi2 = 5
    b_pareto, size_pareto = 2, 1000

    # Loi de Poisson continue
    x_poisson_cont = np.random.poisson(mu_poisson_cont, size_poisson_cont)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.hist(x_poisson_cont, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Loi de Poisson continue')

    # Loi normale
    x_normal = np.linspace(-4, 4, 1000)
    y_normal = stats.norm.pdf(x_normal, mu_normal, sigma_normal)
    plt.subplot(2, 3, 2)
    plt.plot(x_normal, y_normal, 'r-', lw=2)
    plt.title('Loi normale')

    # Loi log-normale
    x_lognormal = np.random.lognormal(mu_lognormal, sigma_lognormal, size_lognormal)
    plt.subplot(2, 3, 3)
    plt.hist(x_lognormal, bins=30, density=True, alpha=0.6, color='b')
    plt.title('Loi log-normale')

    # Loi uniforme
    x_uniform = np.random.uniform(a_uniform, b_uniform, 1000)
    plt.subplot(2, 3, 4)
    plt.hist(x_uniform, bins=30, density=True, alpha=0.6, color='m')
    plt.title('Loi uniforme')

    # Loi du χ²
    x_chi2 = np.random.chisquare(df_chi2, 1000)
    plt.subplot(2, 3, 5)
    plt.hist(x_chi2, bins=30, density=True, alpha=0.6, color='c')
    plt.title('Loi du χ²')

    # Loi de Pareto
    x_pareto = np.random.pareto(b_pareto, size_pareto) + 1
    plt.subplot(2, 3, 6)
    plt.hist(x_pareto, bins=30, density=True, alpha=0.6, color='y')
    plt.title('Loi de Pareto')

    plt.tight_layout()
    plt.show()

def calculate_mean_std(distribution, *params):
    if distribution == 'dirac':
        a = params[0]
        return a, 0
    elif distribution == 'uniform_discrete':
        n = params[0]
        mean = (n + 1) / 2
        variance = (n**2 - 1) / 12
        std = np.sqrt(variance)
        return mean, std
    elif distribution == 'binomial':
        n, p = params[0], params[1]
        mean = n * p
        variance = n * p * (1 - p)
        std = np.sqrt(variance)
        return mean, std
    elif distribution == 'poisson':
        mu = params[0]
        return mu, np.sqrt(mu)
    elif distribution == 'zipf':
        a = params[0]
        # Zipf distribution mean and variance are infinite for a <= 2
        return np.nan, np.nan
    elif distribution == 'poisson_cont':
        mu = params[0]
        return mu, np.sqrt(mu)
    elif distribution == 'normal':
        mu, sigma = params[0], params[1]
        return mu, sigma
    elif distribution == 'lognormal':
        mu, sigma = params[0], params[1]
        mean = np.exp(mu + (sigma**2)/2)
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        std = np.sqrt(variance)
        return mean, std
    elif distribution == 'uniform':
        a, b = params[0], params[1]
        mean = (a + b) / 2
        variance = (b - a)**2 / 12
        std = np.sqrt(variance)
        return mean, std
    elif distribution == 'chi2':
        df = params[0]
        mean = df
        variance = 2 * df
        std = np.sqrt(variance)
        return mean, std
    elif distribution == 'pareto':
        b = params[0]
        if b > 1:
            mean = b / (b - 1)
        else:
            mean = np.inf
        if b > 2:
            variance = b / ((b - 1)**2 * (b - 2))
            std = np.sqrt(variance)
        else:
            std = np.inf
        return mean, std

# Exécution des fonctions
plot_discrete_distributions()
plot_continuous_distributions()

# Exemple de calcul de moyenne et écart type pour chaque distribution
print("Moyenne et écart type pour chaque distribution :")
print("Loi de Dirac (a=0):", calculate_mean_std('dirac', 0))
print("Loi uniforme discrète (n=10):", calculate_mean_std('uniform_discrete', 10))
print("Loi binomiale (n=20, p=0.5):", calculate_mean_std('binomial', 20, 0.5))
print("Loi de Poisson (mu=5):", calculate_mean_std('poisson', 5))
print("Loi de Zipf-Mandelbrot (a=1.5):", calculate_mean_std('zipf', 1.5))
print("Loi de Poisson continue (mu=5):", calculate_mean_std('poisson_cont', 5))
print("Loi normale (mu=0, sigma=1):", calculate_mean_std('normal', 0, 1))
print("Loi log-normale (mu=0, sigma=0.5):", calculate_mean_std('lognormal', 0, 0.5))
print("Loi uniforme (a=0, b=1):", calculate_mean_std('uniform', 0, 1))
print("Loi du χ² (df=5):", calculate_mean_std('chi2', 5))
print("Loi de Pareto (b=2):", calculate_mean_std('pareto', 2))

