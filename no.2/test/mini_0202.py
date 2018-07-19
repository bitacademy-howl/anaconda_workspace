# 1). A학급의 학생 이름은 다음과 같다:
#
# Bob
# John
# Sara
# Jack
# John
# Paul
# Belinda
# Jessica
#
# 위 자료를 리스트 a로 정리해본다.
x = '''Bob 
John 
Sara 
Jack 
John 
Paul 
Belinda 
Jessica '''

list1 = x.replace(' ', '').split('\n')
print(list1)

# 학생 이름을 대문자화 한 리스트 A를 만들어 보시오.
x = x.upper()
list1 = x.replace(' ', '').split('\n')
print(list1)

#3). A학급과 B학급 명부를 통합한 리스트 AB를 만든다. (이어 붙이기) 전체 학생수는?
# 2). A학급의 학생 이름은 다음과 같다:
#
# John
# John
# Rebecca
# Paula
# Brandon
# Elilzabeth
# Sara
#
# A학급과 같은 방법으로 대문자화 한 리스트 B를 만들어 보시오.

y = '''John 
John 
Rebecca 
Paula 
Brandon 
Elilzabeth 
Sara '''
y = y.upper()
list2 = y.replace(' ', '').split('\n')
print(list2)

list_total = list1 + list2

# 4). 통합 리스트 AB를 알파벳 순으로 정렬해 본다.
list_total.sort()
print(list_total)

# 5). 통합 리스트 AB에서 서로 다른 이름의 가짓수는?
# sol 1 : set을 이용한 방법
set_st = set()

for i in list_total:
    set_st.add(i)
print(type(set_st), len(set_st), set_st)

# 만약 set을 리스트로 바꾸려면
result = list(set_st)
print(type(result), len(result), result)

# sol 2 : for 문 돌면서 비교 후 결과리스트에 저장하거나 리스트 자체에서 제거
# 코드도 길어지고, 의미도 별로 없으므로 skip
# 스튜던트를 각각 객체로 만들고, 학생마다 ID 를 주어야 할 것!
# 이 후 ID 로 루프 돌면서 이름이 같은 학생은 list에 append 하지 않고, 어쩌구...
# for 문은 이중 포문
# for 1 : 본래의 리스트를 반복
# for 2 : 반복문 내 i 와 생성된 리스트를 비교

# 6). 통합 리스트 AB에서 'J'로 시작하는 이름만 출력해 본다.
for i in list_total:
    if i[0] == 'J':
        print(i)

# 7). 통합 리스트 AB에서 'A'로 끝나는 이름만 출력해 본다.
for i in list_total:
    if i[-1] == 'A':
        print(i)