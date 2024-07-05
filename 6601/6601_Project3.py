#903856482 Xiangjun Pei Project_3
#Should close the first figure and the the second figure will show.

#3)The common distributions experiments base on the website.
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform, gamma, expon, poisson, binom, bernoulli, norm

# Set the parameters for the distributions
distributions = {
    'Uniform': {'start': 0, 'width': 10},
    'Normal': {'loc': 0, 'scale': 1},
    'Gamma': {'a': 1.99},
    'Exponential': {'scale': 1},
    'Poisson': {'mu': 3},
    'Binomial': {'n': 10, 'p': 0.5},
    'Bernoulli': {'p': 0.5}
}

# Number of random variables to generate for each distribution
size = 10000

# Generate random variables for each distribution
simulations = {}
for dist_name, params in distributions.items():
    if dist_name == 'Uniform':
        simulations[dist_name] = uniform.rvs(params['start'], params['width'], size=size)
    elif dist_name == 'Normal':
        simulations[dist_name] = norm.rvs(params['loc'], params['scale'], size=size)
    elif dist_name == 'Gamma':
        simulations[dist_name] = gamma.rvs(params['a'], size=size)
    elif dist_name == 'Exponential':
        simulations[dist_name] = expon.rvs(scale=params['scale'], size=size)
    elif dist_name == 'Poisson':
        simulations[dist_name] = poisson.rvs(params['mu'], size=size)
    elif dist_name == 'Binomial':
        simulations[dist_name] = binom.rvs(params['n'], params['p'], size=size)
    elif dist_name == 'Bernoulli':
        simulations[dist_name] = bernoulli.rvs(params['p'], size=size)

# Plot the histograms for the simulations
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (dist_name, data) in zip(axes, simulations.items()):
    ax.hist(data, bins=50, density=True, alpha=0.6, color='b')
    ax.set_title(f'{dist_name} Distribution')
    ax.grid(True)

# Remove empty subplots
for i in range(len(simulations), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()



#4) 2 most important approximation for the binomial distribution

n = 100
p = 0.5


mu = n * p
sigma = np.sqrt(n * p * (1 - p))
x = np.arange(0, n+1)

binomial_pmf = binom.pmf(x, n, p)
normal_pdf = norm.pdf(x, mu, sigma)


lambda_ = n * p


poisson_pmf = poisson.pmf(x, lambda_)
plt.figure(figsize=(14, 7))


plt.vlines(x, 0, binomial_pmf, colors='black', lw=5, label='Binomial PMF')


plt.plot(x, normal_pdf, 'r--', lw=2, label='Normal Approximation')
plt.plot(x, poisson_pmf, 'g-.', lw=2, label='Poisson Approximation')

plt.title(f'Binomial Distribution and its Approximations n={n}, p={p}')
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
