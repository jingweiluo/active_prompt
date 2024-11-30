import numpy as np # type: ignore
import random

myList = []
for i in range(10):
    sublist=[i]
    myList.extend(sublist)


randList = random.sample(myList, 5)
print(list(range(4)))

