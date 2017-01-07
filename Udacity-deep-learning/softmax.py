"""Softmax."""
import math as math
import numpy as np
scores = np.array([3.0, 1.0, 0.2])



def softmax(x):
    result = np.exp(x) / np.sum(np.exp(x), axis=0)
    print("x", x)
    return result

print("result",softmax(scores/10))


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores/10).T, linewidth=2)
plt.show()