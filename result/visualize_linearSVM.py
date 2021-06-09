from matplotlib import colors
import matplotlib.pyplot as plt
import re

C_list = []
train_acc = []
test_acc = []

with open('linearSVM_tuning.txt') as f:
    C_line = f.readline()
    while 1:
        C_list.append(float(re.findall("\d+\.\d+", C_line)[0]))
        train_acc.append(float(f.readline().split(':')[1]))
        test_acc.append(float(f.readline().split(':')[1]))
        C_line = f.readline()
        if (C_line == ''):
            break

plt.plot(C_list[1:], train_acc[1:], label = 'train accuracy')
plt.plot(C_list[1:], test_acc[1:], label = 'test accuracy')
plt.legend()
plt.savefig('bestC.png')
plt.show()