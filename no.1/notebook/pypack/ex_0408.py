import numpy as np
import scipy.stats as st

# 95% 신뢰구간.
p_mean = 5500

n = 25

SEM = np.sqrt(p_mean*(1-p_mean)/n_total)
st.norm.interval(0.95, loc = p_mean, scale=SEM)