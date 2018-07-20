import numpy as np

# class A :
#     val = None
#     def __init__(self, val):
#         self.val = val
#
# a1 = A(1)
# a2 = A(2)
# a3 = A(3)
#
# container = [a1, a2, a3]
#
# for i in container:
#     print(i.val)
#
# print(container[0].val, container[1].val)
#
# print(container)
#
#
#
# # 자바 2차원 배열 같이 대입 및 선언 못함
# # a = [1,2,3][4,5,6]
# a = ([[1,2,3],[4,5,6],[7,8,9]])
# print(a)
#
# # 하지만 numpy 를 통해 배열을 선언하고 처리할 수 있다.
# # 배열의 선언
# a = np.zeros((3,4))
# print(a.dtype)
#
# print(type(a[1][1]), a[1][1])
# print(a)
#
# print(a.imag)

# a = np.arange(15)
# print(a.reshape(3,5))
#
# print(a)
# a.shape = (3,5)
# print(a)

#
# a = [[1,2,3,4],[1,3,5,7],[2,3,4,5]]
# print(a)
#
# for i in range(0, len(a)):
#     print(a[i])
#
#
# a = np.arange(25)
# print(a.reshape(5,5))
# print(a)
# a.shape = 5,5
# print(a)
#
# a = np.arange(10)
# b = a.reshape(2, 5)
#
# print(type(b), type(a))

# print(np.__version__)

a = 12
print(a, type(a))
arr1 = np.array([1,3,5,7,9], int)
arr2 = np.array([1,3,5,7,9])

# print("arr1 : ", arr1)
#
# print(arr1, arr1.dtype, type(arr1[1]))
#
# b = arr1[1] + a
# c = a + arr1[1]
# d = int(c)
# print(b,d,type(b), type(d))

# arr3 = arr1
#
# id_of_arr1 = id(arr1)
#
# print(id_of_arr1)
# print(id(arr2))
# print(id(arr3))
#
# arr4 = arr1.copy()      # numpy.array 객체의 copy() 메서드도 오버라이딩 되었고,
#                         # 객체 내 멤버변수도 일반 리스트가 아니므로 기존 파이썬의 deepcopy가 아니더라도
#                         # deep 카피와 같이 동작하게 된다. ---> numpy 정의
# print(id(arr4))
#
# arr2[3] = 1092
# print(type(arr2[3]))
# print(arr2)
#
# np.linspace(1,10,5)
# arr = np.linspace(0,10,5, retstep=True)
# print(arr, type(arr))
# print(arr[0], type(arr[0]))
# print(arr[1], type(arr[1]))
#
aa = np.array([1,2,3,4,5,6])

a = np.arange(10) # 0부터 10 미만의 숫자를 numpy array로 생성
print(a, type(a))

# 아래 코드의 메커니즘
# 1. numpy ndarray 형 객체로 생성
# 2. 10 부터 20 미만의 숫자를 numpy.int32 객체자료형의 데이터로 필드에 저장
b = np.arange(10, 20) #slicing 에 사용되는 start, end 값과 동일
print(b, type(b))
print(b[0], b[1], type(b[1]))

c = np.arange(10, 20, 3) # 10이상 20 미만의 숫자 중 3을 증가값으로 하여 ndarray에 생성 후 저장
print(c, type(c), c[0])




arr = np.arange(10)
print(len(arr)) # 10개
print(arr.size) # ndarray 클래스 멤버로 len 이 정의되어 있다.
# 기본내장 함수인 len() 메서드는 객체의 __len__ 속성을 get 하게 되고, __len__메서드를 정의 하면
# 객체의 len(객체)를 호출 하였을 때 불러올 값을 지정하게 된다.
#
# ex) 아래는 len 메서드 정의의 예제
class Myclass:
    inputlist = []
    def __init__(self, inputlist):
        self.inputlist = inputlist
    def __len__(self):
        return len(self.inputlist)
    def __str__(self):
        return str(self.inputlist[0:3])

myclass = Myclass([1,2,3,4,5])
print(len(myclass))

# 같은 방법으로 print 함수가 불러올 __str__ 을 지정할 수 있다.
# 단 __str__의 반환값은 string 이어야 한다.
print(myclass)
# GOOD!!!

print(id(myclass))
# 여기서 toString 을 지정해줌으로써 print(객체)의 형태로 객체 비교가 어렵게 된다..
# id 라는 메서드를 사용하여 실제 객체의 주소값을 반환받을 수도 있다.
# 프로그램 재실행 때마다 당연히 id 는 달라짐


# 리스트 연산과 numpy 연산의 차이

# 위에서 실습한 내용과 같은 메커니즘으로, 파이썬의 연산자는 각각의 클래스의 멤버속성으로 정의된
# __add__ 등의 메서드를 재정의 (연산자 오버로딩)으로 구현되어있으므로
# http://blog.eairship.kr/287 참고

# numpy에서 정의한 __add__는  다음과 같이 동작한다.
# __add__를 정의할 때 __radd__ 까지 정의해주어야 연산자 기준으로 순서가 바뀌어도 에러가 발생하지 않음
# 각각의 ndarray에 저장된 값들을 같은 index 끼리 더함
# 아래의 메커니즘
# for i in range(len(객체 내 numpy.ndarray 자료형의 속성)):
#     a[i] + b[i]

# Numpy 배열의 연산 중
# 배열끼리의 * , / 은 수학적 벡터의 연산과는 별개로
# 각각의 어레이의 값들의 스칼라곱 및 나눗셈으로 동작



x = np.array([1,2,3,4])
print(pow(10, x))  # 제곱



# ################################################################################################
np.random.seed(123)  # 랜덤 넘버 생성에 사용될 seed
                     # 실제 seed 에 의한 랜덤넘버 생성은
                     # (특히, 반복적으로 사용할 경우 system clock 과의 문제)
                     # 엄밀한 정의의 Random ! 이라고 보기는 어려우니 가급적 사용자제
# 랜덤변수의 생성은 계속 알아볼것!!!
###################################################################################################

# np.random.randint()
print(x, type(x))



import numpy as np

a = [1, 2, 3]
print(a + a)   # 리스트의 + 연산자
                # 리스트끼리 연결(확장, extend 하여 return 값으로 반환)

print(a*3)      # 리스트의 * 연산자
                # 리스트를 3번 extend (자신과)

b = np.array([1, 2, 3])     # 넘파이 어레이 선언 및 할당

print(b*3)                  # 각각의 ndarray 값들과 3에대한 스칼라곱을 개별적으로 수행

print(b + b)                # 각각의 ndarray 값들끼리 더함

# print(b, type(b))
# print(b[0:2], type(b[0:2]))
# print(b + b[0:2])           # 연산자 좌우의 크기가 같지 않으면 valueError

c = np.repeat(b, 3)
print(c[0],c[1],c[2], type(c))

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 아래의 사칙연산들은 벡터의 그것과는 별개로
# 각각 개별 요소들의 스칼라 연산으로 동작
print(a+b)
print(a-b)
print(a*b)
print(a/b)

# 연산의 벡터화
x = np.array([0,1,2,3])
y = pow(10, x)
print(y, type(y))             # 벡터 자체를 함수의 인자로 대입하였을 때,
                                # 그 반환값도 동일한 객체로 돌려받음

# x = [0,1,2,3]

# y = pow(10, x)              # 이건 어떻게 ????ㅋㅋ

print(y)
# 시나리오 1 : pow 라는 함수가 디폴트로 불러오는 객체의 메서드 정의
# pow 를 정의하고 __rpow__ 정의하면 가능!!
# __pow__(self, other)
#
# pow(), **
#
# pow(A, B), A ** B

#  OK!!!!!!


# np.sum(ndarray) ======>>>  개개 속성들의 총합
print(np.sum([1,2,3,4,5]))    # 이거 리스트도 사용가능
print(sum([1,2,3,4,5]))         # 근데 리스트를 이렇게 쓸거면 굳이 np 필요 없음


# 벡터의 외적과 내적
# 리스트로도 사용 가능
x = [1,2,3,4,5]
y = [1,2,3,4,5]

x1 = np.array([1,3,5])
y1 = np.array([2,4,6])

print(1+4+9+16+25) # 벡터 내적의 수학적 정의
print(np.dot(x,y)) # 코드

print(2+12+30)       # 벡터 내적의 수학적 정의
print(np.dot(x1,y1)) # ndarray 끼리의 연산

# dot 함수는 행렬간 연산일 경우, 행렬의 곱을 수행한다.
# ex) 아래
x = np.array([[1,2,3]])
y = np.array([[1],[2],[3]])

# 수학적 행렬의 곱 정의
# [1, 2, 3] * [1] = [1+4+9] = [14]
#             [2]
#             [3]
#   --> scalar
print(np.dot(x, y))

# 수학적 행렬의 곱 정의 2   ====>>>  [n by m] * [m by p] = [n by p]
# [1] * [1][2][3] = [1][2][3]
# [2]               [2][4][6]
# [3]               [3][6][9]

print('-----------\n', np.dot(y, x), '\n-----------')

# 대각행렬 (diagonal matrix)
x = np.diag([1,2,3]) # 대각행렬 ---> 단위행렬의 형태를 띔
print(x)

# 아래의 방법으로 단위행렬 생성이 가능하다.
n = 4
y = np.diag([1]*n)
y = np.eye(4) # 로도 생성가능

print(y)

x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
x2 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])

print(x * y)    # 이건 각각 개별 요소끼리의 곱...
                # 그러므로 행렬의 크기가 동일해야 연산가능
# print(x2 * y) # 오류

# 전치 행렬 => 행과 열의 뒤바뀜
transpose_x2 = np.transpose(x2)
print(transpose_x2, '\n', x2)

# transpose 메서드의 활용
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,8,9],[10,11,12]])
result = np.dot(x, np.transpose(y))
print('@@@@@@@전치행렬 행렬의 곱@@@@@@@\n', result)
# 역행렬 #################################################################################################

# 수학적 정의 ======> n by n 행렬이라 가정할 때

# AA-1 = identity matrix(n x n)

##########################################################################################################
# 여기서, 행렬의 기본적인 내용을 보면 (!!! caution !!!)
# 1. 단위행렬은 n by n 즉, 정방행렬 일때에만 정의 된다.
# 2. 즉 역행렬은 정방행렬에 대해서만 존재한다. -----> 가역행렬의 조건
# 3. 영행렬의 역행렬은 존재하지 않음
# 참고 : 위키 . 가역행렬 #################################################################################

# https://ko.wikipedia.org/wiki/%EA%B0%80%EC%97%AD%ED%96%89%EB%A0%AC

#########################################################################################################
# 행렬의 역행렬
m = np.array([[1,2],[3,4]])
print("################################## 2X2 행렬 ########################################")
print(m)
minv = np.linalg.inv(m)
print("################################## 역행렬 ########################################")
print(minv)

print("################################## 역행렬과의 곱 ##################################")
a_dot_ai = np.dot(m, minv) # 왜 m(1,2) 인자가 0 이 아니쥐..??????
print(a_dot_ai)

# 연립방정식의 해
a = np.array([[5,8],[6,4]])
b = np.array([[30], [25]])

# 풀이 1 : 수학적 풀이
ainv = np.linalg.inv(a)
result = np.dot(ainv, b)
# 풀이 2 : 연립방정식의 해는 아래의 방법으로도 구할 수 있다 - solve 메서드의 활용
result1 = np.linalg.solve(a, b) 
print("################################## 연립방적식의 해 ##################################")
print(result)
print(result1)

print(1*1.-2*0.5)

#########################################################################################################
# 행렬의 도함수 찾아볼것!!
#########################################################################################################
