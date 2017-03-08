import numpy as np


def test(ar):
	ar = ar*2
	print("ar",ar)
	return

if __name__=='__main__':
	a = [[1,2],[3,4]]
	a = np.array(a)
	print(a)
	print(type(a))
	test(a)
	print(a)
