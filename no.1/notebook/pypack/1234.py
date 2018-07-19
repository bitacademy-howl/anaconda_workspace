import numpy as np
import scipy.stats as st

mu = 172
sigma = 5
st.norm.pdf(0, loc=mu, scale=sigma)
print(st.norm.cdf(180,loc = mu, scale=sigma) - st.norm.cdf(175,loc = mu, scale=sigma))