

import numpy as np



list = []

inner = []

for i in range(0,5):
    for j in range(0,5):
        inner.append(j)

    list.append(inner)

print(list)

x = np.array(list)


print(x, type(x))

