import numpy as np
import matplotlib.pyplot as plt
import corner


data = np.loadtxt('chains/1-post_equal_weights.dat')

corner.corner(data, levels = (1-np.exp(-0.5),(1-np.exp(-2))), smooth = 1)
plt.show()