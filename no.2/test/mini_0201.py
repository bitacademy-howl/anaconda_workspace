# 1. 1). A학급의 학생 신장을 조사해 보니 다음과 같다:
#
# 161.5
# 155.9
# 168.7
# 163.1
# 170.8
# 167.2
#
# 위 자료를 리스트로 정리해본다.
list1 = [161.5, 155.9 , 168.7 , 163.1 , 170.8 , 167.2 ]

# 리스트와 for 반복문을 사용해서 평균 신장을 구해본다.
sum = 0
for i in list1:
    sum += i
avr1 = sum/len(list1)
print(avr1)

# 2). B학급의 학생 신장을 조사해 보니 다음과 같다:
#
# 172.5
# 162.9
# 161.7
# 160.1
# 161.8
# 159.2
#
# A학급과 같은 방법으로 평균 신장을 구해본다.

list2 = [172.5 , 162.9 ,161.7 ,160.1 ,161.8 ,159.2 ]

sum = 0
for i in list2:
    sum += i

avr2 = sum/len(list2)
print(avr2)

# 3). A학급과 B학급 신장 자료를 통합한 리스트 AB를 만든다. (이어 붙이기)
#
# 이 통합 리스트 AB를 가지고 평균을 구한다.

list_join = list1+list2
list1.extend(list2) # +연산도 가능

print(list1)
print(list_join)

# 4). 통합 리스트 AB를 정렬해 본다.
list1.sort()
list2.sort()
list_join.sort()

print(list1, list2, list_join)

mininlist = list_join[0]
maxinlist = list_join[-1]
print(mininlist, maxinlist)

# 6). 통합 리스트 AB에서 가장 큰 값을 가져오시오.

# 7). 통합 리스트 AB를 사용하여 신장이 165이하인 학생들 만의 리스트 C를 만든다.

st_lower_165 = []
for i in list_join:
    if i <= 165:
        st_lower_165.append(i)
        continue
    else:
        break

print(st_lower_165)

sum = 0
for i in st_lower_165:
    sum += i

avr_st_lower_165 = sum/len(st_lower_165)
print("avr_st_lower_165 is", avr_st_lower_165)

# 8). 통합 리스트 AB를 사용하여 신장이 160이상 170이하인 학생들 만의 리스트 D를 만든다.
# solution 1 : for 문을 모두 돈다 조건문 두번 수행
# 장점 : sorting 되지 않아도 사용 가능
# 단점 : 자료가 많은 경우 모든 자료를 비교하기에 빅데이터 관점에서는 개별적 모듈로 부적합
st_between_160_170 = []
for i in list_join:
    if i > 160 and i < 170:
        st_between_160_170.append(i)

print(st_between_160_170)


# solution 2 : for 문을 빠져나오는 조건을 i < 170 으로...
# 장점 : for 문을 전부 돌지 않는다. for 문 내 조건문 1번 수행
# 한계 : sorting 된 자료를 순차탐색 하는 경우에만 국한
list_join.sort()
st_between_160_170 = []

for i in list_join:
    if i > 170:
        break
    elif i >= 160:
        st_between_160_170.append(i)
        # continue

print(st_between_160_170)


# 리스트 D의 평균을 구하여라.
sum = 0
for i in st_between_160_170:
    sum += i

st_between_160_170 = sum/len(st_between_160_170)
print("average of st_between_160_170 :", st_between_160_170)