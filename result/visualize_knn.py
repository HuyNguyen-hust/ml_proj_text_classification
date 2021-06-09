import matplotlib.pyplot as plt

k_list = []
train_acc = []
test_acc = []

with open('knn_tuning.txt') as f:
    k_line = f.readline()
    while 1:
        k_list.append([num for num in k_line.split() if num.isdigit()][0])
        train_acc.append(float(f.readline().split(':')[1]))
        test_acc.append(float(f.readline().split(':')[1]))
        k_line = f.readline()
        if (k_line == ''):
            break


plt.plot(k_list[:11], train_acc[:11], label = 'train accuracy')
plt.plot(k_list[:11], test_acc[:11], label = 'test accuracy')
plt.legend()
plt.savefig('bestk1.png')
plt.show()
plt.clf()

plt.plot(k_list[12:], train_acc[12:], label = 'train accuracy')
plt.plot(k_list[12:], test_acc[12:], label = 'test accuracy')
plt.legend()
plt.savefig('bestk2.png')
plt.legend()
plt.show()