import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

processed_path = 'processed\\'

corpus = []
train = []
test = []
with open(processed_path + 'processed_train_data.txt', encoding = 'utf-16') as f:
    lines = f.read().splitlines()
    for line in lines:
        text = line.split('<fff>')[2]
        train.append(text)
        corpus.append(text)

with open(processed_path + 'processed_test_data.txt', encoding = 'utf-16') as f:
    lines = f.read().splitlines()
    for line in lines:
        text = line.split('<fff>')[2]
        test.append(text)
        corpus.append(text)

vectorizer =  TfidfVectorizer(token_pattern = u'(?ui)\\b\\w*[a-z]+\\w*\\b', min_df = 10)
vectorizer.fit(corpus)

train_tfidf = vectorizer.transform(train)
test_tfidf = vectorizer.transform(test)

print('building succesfully!')

joblib.dump(vectorizer, processed_path + 'vectorizer.pkl')
joblib.dump(train_tfidf, processed_path + 'train_tfidf.pkl')
joblib.dump(test_tfidf, processed_path + 'test_tfidf.pkl')