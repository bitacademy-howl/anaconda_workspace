import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

n=10
p=0.3
x=2

st.binom.pmf(x, n, p)
print(st.binom.pmf(x, n, p))

print(st.binom.cdf(5,n,p))

print(st.binom.cdf(7,n,p)-st.binom.cdf(2,n,p))

x=np.arange(0,11)
result = 0
for val in st.binom.pmf(x,10,p):
    result += val
print(result)

plt.scatter(x, st.binom.pmf(x,10,p),color='red')

plt.show()

lamb = 2

print(st.poisson.pmf(2,lamb))

# P(0 <= x <= 5)
print(st.poisson.cdf(5,lamb))