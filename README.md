# text_classification

Text classification using mutiple methods: kNN, Naive Bayes and Linear SVM.

# dataset
Data used for this project was downloaded from [VNTC dataset](https://github.com/duyvuleo/VNTC)

# training and evaluating
Firstly install all the necessary libraries:
```
>> pip install -r requirements.txt
```

To train and evaluate your model using your own dataset you have to construct your dataset directory exactly as below:  
  
dataset  
--> train  
----> class 1  
----> class 2  
...  
--> test  
----> class 1  
----> class 2  
...  
--> stopwords.txt  
  
Then run all these following commands to preprocess your raw data and building the coressponding vectorizer:  
```
>> python preprocess.py
```
```
>> python building_tfidf_vectorizer.py
```
After that run these commands to train 3 models (kNN, Naive Bayes, Linear SVM) and evaluate their performance:
```
>> python knn.py
```
```
>> python naive_bayes.py
```
```
>> python linear_SVM.py
```

The vectorizer and 3 models are saved into processed directory and model directory respectively for later uses

# result
| Model| kNN | Naive Bayes | Linear SVM |
| :-: | :-: | :-: | :-: |
| Train Accuracy | 0.8338 | 0.8514 | 0.9949 |
| Test Accuracy | 0.8257 | 0.8321 | 0.8907 |


# predicting
For those who only want to predict new text without training, run this:
```
>> python predict.py <newspaper_url>
```
It crawls content from newspaper_url and return classifying results
