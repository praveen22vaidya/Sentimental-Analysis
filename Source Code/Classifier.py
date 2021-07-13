# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 01:34:48 2019

@author: Praveen
"""
#Importing required packages
import pandas as pd
import json
import re
import pickle
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
import pickle 
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, Dropout, LSTM
from keras.utils.np_utils import to_categorical

#Opening pre trained logistic classifier
with open('classifier.pickle','rb') as f:
    classifier = pickle.load(f)
    
#Opening pre trained tf-idf model    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)    
    

full_test_list = []
created_at_list = []  

#opening twitter file for reading  
with open('D:\\json-preprocess-data\\tweetsSample.json', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for eachRecord in data:
       full_test_list.append(eachRecord['full_text'])
       created_at_list.append(eachRecord['created_at'])

#Cleaning data using Regular expressions
corpus = []
for tweet in range(0, len(full_test_list)):
    review = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", ' ', str(full_test_list[tweet]))
    review = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", review)
    review = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", review)
    review = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", review)
    review = review.lower()
    review = re.sub(r"that's","that is",str(review))
    review = re.sub(r"there's","there is",review)
    review = re.sub(r"what's","what is",review)
    review = re.sub(r"where's","where is",review)
    review = re.sub(r"it's","it is",review)
    review = re.sub(r"who's","who is",review)
    review = re.sub(r"i'm","i am",review)
    review = re.sub(r"she's","she is",review)
    review = re.sub(r"he's","he is",review)
    review = re.sub(r"they're","they are",review)
    review = re.sub(r"who're","who are",review)
    review = re.sub(r"ain't","am not",review)
    review = re.sub(r"wouldn't","would not",review)
    review = re.sub(r"shouldn't","should not",review)
    review = re.sub(r"can't","can not",review)
    review = re.sub(r"couldn't","could not",review)
    review = re.sub(r"won't","will not",review)
    review = re.sub(r"\W"," ",review)
    review = re.sub(r"\d"," ",review)
    review = re.sub(r"\s+[a-z]\s+"," ",review)
    review = re.sub(r"\s+[a-z]$"," ",review)
    review = re.sub(r"^[a-z]\s+"," ",review)
    review = re.sub(r"\s+"," ",review)
    corpus.append(review)
    
#Assigning sentiment labels to cleaned text through pre trained classifier   
sent = classifier.predict(tfidf.transform(corpus).toarray())

#World cloud to show most used words and how often do they repeat
all_words = ' '.join([text for text in corpus])
wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
   


#Creating to Bag of Words model
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


# Creating the Tf-Idf Model
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

#Splitting the data set into train and test set
text_train, text_test, sent_train, sent_test = train_test_split(X, sent, test_size = 0.30, random_state = 0)


# Training the classifier using Logistic regression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)



# Testing model performance
sent_pred = classifier.predict(text_test)


#Printing Comfusion matrix,accuracy score and classification report for logistic regression
ConfusionMatrix = confusion_matrix(sent_test, sent_pred)
ConfusionMatrix
print(accuracy_score(sent_test, sent_pred))
print(classification_report(sent_test, sent_pred, digits=5))

# Training the classifier using Multinomial Naive Bayes
NaiveBayes=MultinomialNB(1.5,fit_prior=False)
NaiveBayes.fit(text_train,sent_train)

# Testing model performance
NaivePred=NaiveBayes.predict(text_test)

#Printing Comfusion matrix,accuracy score and classification report for Multinomial Naive Bayes
print(accuracy_score(sent_test,NaivePred))
print(confusion_matrix(sent_test,NaivePred))
print(classification_report(sent_test, NaivePred, digits=5))

#Creating Count vectorizer model
count_vectorizer = CountVectorizer(stop_words='english') 
cv = count_vectorizer.fit_transform(corpus)
cv.shape

#Splitting the data into train and test set for XGBoost
XY_train,XY_test,yz_train,yz_test = train_test_split(cv,sent, test_size=.3,stratify=sent, random_state=42)

# Training the classifier using XGboost
xgbc = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3)
xgbc.fit(XY_train,yz_train)

# Testing model performance
prediction_xgb = xgbc.predict(XY_test)

#Printing Comfusion matrix,accuracy score and classification report for XGBoost
print(confusion_matrix(sent_test,prediction_xgb))
print(accuracy_score(prediction_xgb,yz_test))
print(classification_report(yz_test, prediction_xgb, digits=5))

#Tokenizing the clean data
tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(corpus)

#Converting into sequences
XT = tokenizer.texts_to_sequences(corpus)

# padding our text vector so they all have the same length
XT = pad_sequences(XT) 
XT[:5]

#Defining LSTM model
model = Sequential()
model.add(Embedding(5000, 256, input_length=XT.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

YS= pd.get_dummies(sent).values
[print(sent[i], YS[i]) for i in range(0,5)]

#Splitting data into train and test set    
X_train, X_test, y_train, y_test = train_test_split(XT, YS, test_size=0.3, random_state=0)

#Defining Batch size and Epochs
batch_size = 32
epochs = 8

#Training using LSTM model
mod=model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)

#Summarizing history for accuracy 
plt.plot(mod.history['accuracy'])
plt.plot(mod.history['val_accuracy'])
plt.legend(['train', 'test'])
plt.title('model loss')
plt.xlabel('Epoch')
plt.show()  

#Summarizing history for loss
plt.plot(mod.history['loss'])
plt.plot(mod.history['val_loss'])
plt.legend(['train', 'test'])
plt.title('model loss')
plt.xlabel('Epoch')
plt.show()