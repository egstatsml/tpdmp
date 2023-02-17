import numpy as np

def fn(a):
  return a, a + 2

l = lambda x: fn(x)

b = 2
print(l(b))
