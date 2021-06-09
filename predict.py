import joblib
import os
import sys
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

dataset_path = 'dataset\\'
processed_path = 'processed\\'
model_path = 'model\\'

def get_topic_list(path):
    default_topics_list = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe', 'The gioi', 'The thao', 'Van hoa', 'Vi tinh']
    assert os.path.exists(path)
    dirs = [path + dirname + '\\' for dirname in os.listdir(path) if not os.path.isfile(path + dirname)]
    if dirs == []:
        return default_topics_list
    train_dir = dirs[0] if 'train' in dirs[0].lower() else dirs[1]
    topics_list = [topic for topic in os.listdir(train_dir)]
        
    print(topics_list)
    return topics_list

topics_list = get_topic_list(path = dataset_path)

vectorizer = joblib.load(processed_path + 'vectorizer.pkl')
linear_SVM_classifier = joblib.load(model_path + 'linear_SVM.pkl')
knn_classifier = joblib.load(model_path + 'knn.pkl')
naive_bayes_classifier = joblib.load(model_path + 'naive_bayes.pkl')

url = sys.argv[1]
article = Article(url = url, language = 'vi')
article.download()
article.parse()
new_text = article.text

print(new_text, end = '\n\n')

processed_text = vectorizer.transform([new_text])

linear_SVM_pred = linear_SVM_classifier.predict(processed_text)
knn_pred = knn_classifier.predict(processed_text)
naive_bayes_pred = naive_bayes_classifier.predict(processed_text)

print('linear SVM prediction: ',topics_list[int(linear_SVM_pred)])
print('knn prediction: ', topics_list[int(knn_pred)])
print('naive bayes prediction: ',topics_list[int(naive_bayes_pred)])