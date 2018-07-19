import numpy as np
import matplotlib.pyplot as plt


a = np.arange(10)
b = np.arange(10, 20)
c = np.arange(10, 20, 3)
print(a,b,c)

# 0 에서 10까지의 5등분(그릿)
a = np.linspace(0, 10, 5)
print(a)

x = np.linspace(0, np.pi, 10)
y = np.sin(x)

plt.plot