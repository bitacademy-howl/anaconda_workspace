try:
    a = 1
    b = 2
    c = a+b / 0
except ZeroDivisionError as zE:
    print(zE)
else:
    print(a+b)
finally:
    print(a, b)