"""
=========================================================
Demo of the histogram (hist) function with a few features
=========================================================

In addition to the basic histogram, this demo shows a few optional
features:

    * Setting the number of data bins
    * The ``normed`` flag, which normalizes bin heights so that the
      integral of the histogram is 1. The resulting histogram is an
      approximation of the probability density function.
    * Setting the face color of the bars
    * Setting the opacity (alpha value).

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

np.random.seed(0)

# example data
mu = 0  # mean of distribution
sigma = 15  # standard deviation of distribution
#x = mu + sigma * np.random.randn(437)
x=np.loadtxt('input.txt',delimiter=',')

num_bins = 200

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x[:,2]*1e6, num_bins, normed=1)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
ax.plot(bins, y, '--')
ax.grid(color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel(r'y [$\mu$m]', fontsize=15)
ax.set_ylabel('Count rate', fontsize=15)
ax.set_title(r'Electron beam profile: $N=50,000\ particles$', fontsize=15)
ax.set_xticks(np.arange(-200,201,50)) 
ax.set_xlim(-200,200)

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
