import pandas as pd
import numpy as np
import scipy.stats as st


# 다음은 미국에서 592명 성인 남녀의 눈색과 머리칼 색을 정리해 놓은 분활표이다.
# 개개 행은 머리칼 색, 개개 열은 눈색을 나타낸다.
# 분활표카이제곱검정 (독립성 검정)을 실시해 보자.

data_eye = {'Brown': [68,119,26,7], 'Blue':[20,84,17,94], 'Hazel':[15,54,14,10], 'Green':[5, 29, 14,16]}
data = pd.DataFrame(data_eye,index=['Black','Brown','Red','Blonde'])
print(data)

c2c = st.chi2_contingency(data)
print(c2c)

# 다음은 타이타닉호의 객실 등급을 나타내는 변수와 생존여부를 나타내는 분활표이다.
# 변수사이의 독립성을 검정해 보자.

data_survived = {'Yes':[122, 167, 528, 673], 'No':[203,118,178,212]}
data = pd.DataFrame(data_survived,index=['1st','2nd','3rd','Crew'])
print(data)

# 통계량, p-값, 자유도.
c2c = st.chi2_contingency(data)
print(c2c)