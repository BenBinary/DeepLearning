from numpy import load
import numpy as np
import cv2

data = load('./mnist.npz')
lst = data.files



for item in lst:
    print(item)
    #print(data[item][0])
    #print(item['x_train'][0])


with np.load('./mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']


#print(x_train[0])
#print(y_train[0])
print(x_test[0])

# Train Export
for i in range(0, 10000):
    cv2.imwrite("./export/train/f" + str(i) + ".png", x_train[i])

# Test Export
for i in range(0, 10000):
    cv2.imwrite("./export/test/f" + str(i) + ".png", x_test[i])