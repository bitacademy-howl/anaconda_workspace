import pandas as pd
import numpy as np
import scipy.stats as st

# 다음은 미국에서 592명 성인 남녀의 눈색을 정리해 놓은 도수분포표 이다.
# 카이제곱 검정을 실시해 보자.
data = np.array([220, 215, 93, 64])
# data = np.array([160, 150, 140, 200])

eye = pd.Series(data, index=['Brown', 'Blue', 'Hazel', 'Green'])
print(eye)

# 디폴트 모형과 비교.
print(st.chisquare(eye, ddof=0))

# 모형 제공.
r = st.chisquare(f_obs=eye, f_exp=len(eye)*[eye.sum()/len(eye)], ddof=1)
print(r)

# 통계량 계산해보기
print(np.sum((eye-eye.mean())**2/eye.mean()))

# Goodness of Fit
# 실제 통계데이터를 비교 모형으로 사용
r = st.chisquare(f_obs=eye,f_exp = np.array([0.41,0.32,0.15,0.12])*eye.sum() ,ddof = 1)
print(r)