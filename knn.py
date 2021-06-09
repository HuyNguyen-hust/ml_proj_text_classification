import sys
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import os

processed_path = 'processed\\'
model_path = 'model\\'
dataset_path = 'dataset\\'

def get_topic_list(path):
    default_topics_list = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe', 'The gioi', 'The thao', 'Van hoa', 'Vi tinh']
    assert os.path.exists(path)
    dirs = [path + dirname + '\\' for dirname in os.listdir(path) if not os.path.isfile(path + dirname)]
    if dirs == []:
        return default_topics_list
    train_dir = dirs[0] if 'train' in dirs[0].lower() else dirs[1]
    topics_list = [topic for topic in os.listdir(train_dir)]
        
    return topics_list

topics_list = get_topic_list(dataset_path)

X_train = joblib.load(processed_path + 'train_tfidf.pkl')
X_test = joblib.load(processed_path + 'test_tfidf.pkl')

y_train = []
y_test = []
with open(processed_path + 'processed_train_data.txt', encoding = 'utf-16') as f:
    lines = f.read().splitlines()
    for line in lines:
        label = line.split('<fff>')[0]
        y_train.append(int(label))

with open(processed_path + 'processed_test_data.txt', encoding = 'utf-16') as f:
    lines = f.read().splitlines()
    for line in lines:
        label = line.split('<fff>')[0]
        y_test.append(int(label))

y_train = np.array(y_train)
y_test = np.array(y_test)
assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)

n_neighbors = int(sys.argv[1])
classifier = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X_train, y_train)

#evaluate on training set
y_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_pred, y_train)
print('accuracy on training set: ', train_accuracy)

#evaluate on test set
y_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_pred, y_test)
print('accuracy on test set: ', test_accuracy)


#confusion matrix
cf_mat = confusion_matrix(y_test, y_pred)
for i in range(len(topics_list)):
    cf_mat[i][i] = 0
df_cm = pd.DataFrame(cf_mat, index = topics_list, columns = topics_list)
plt.figure(figsize = (16,5))
sn.heatmap(df_cm, annot = True, cmap = 'YlGnBu', linewidths= .5)

plt.savefig('result\\knn_cfmat.png')

#save classifier
joblib.dump(classifier, model_path + 'knn.pkl')

'''
with open('result\\knn_tuning.txt', 'a') as f:
    f.write('k = ' + str(n_neighbors) + '\n')
    f.write('train_accuracy: ' + str(train_accuracy) + '\n')
    f.write('test_accuracy: ' + str(test_accuracy) + '\n')
'''