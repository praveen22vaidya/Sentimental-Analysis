# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:32:30 2019

@author: Praveen
"""
#Importing required packages
import pandas as pd 
import re
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import pickle 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Reading Sentiment 140 dataset
cols = ['sentiment','text']
df = pd.read_csv('C:/Users/Praveen/Desktop/sentiment140/training.1600000.processed.noemoticon.csv', encoding='latin1', names=cols)

#Defining patterns for Regular Expressions
pat1= '#[^ ]+'
pat2 = 'www.[^ ]+'
pat3 = '@[^ ]+'
pat4 = '[0-9]+'
pat5 = 'http[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",   
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
pattern = '|'.join((pat1,pat2,pat3,pat4,pat5))
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
stop_words = stopwords.words('english')


#Pre processing the tweets
clean_tweets = []
for t in df['text']:
    t.lower()
    t = re.sub(pattern,'',t)
    t = neg_pattern.sub(lambda x: negations_dic[x.group()], t)
    t = word_tokenize(t)
    t = [x for x in t if len(x) >1]
    t = [x for x in t if x not in stop_words]
    t = [x for x in t if x.isalpha()]
    t = " ".join(t)
    t = re.sub("n't","not",t)
    t = re.sub("'s","is",t)
    clean_tweets.append(t)
    
#Storing the cleaned tweets as data frame
clean_df = pd.DataFrame(clean_tweets, columns=['text'])
#Replacing sentiment 4 with 1 for positive sentiments
clean_df['sentiment'] = df['sentiment'].replace({4:1})
#Checking details about cleaned 
clean_df.head()
clean_df.info()

#Assigning sentiments to negative and positive tweets
neg_tweets = clean_df[clean_df['sentiment']==0]
pos_tweets = clean_df[clean_df['sentiment']==1]

#Assigning clean text and Sentiment to variables X and y
X = clean_df['text']
y = clean_df['sentiment']

#Dumping X and y into pickle objects
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)

# Unpickling dataset
X_in = open('X.pickle','rb')
y_in = open('y.pickle','rb')
X = pickle.load(X_in)
y = pickle.load(y_in)


        

# Creating the BOW model
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(X).toarray()


# Creating the Tf-Idf Model
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()



# Splitting the dataset into the Training set and Test set
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# Training the classifier
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)


# Testing model performance
sent_pred = classifier.predict(text_test)


#Printing confusion matrix and accuracy score
ConfusionMat = confusion_matrix(sent_test, sent_pred)
ConfusionMat
print(accuracy_score(sent_pred,sent_test ))



# Saving our classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# Saving the Tf-Idf model
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    

# Using our classifier
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    
#Checking our pre tained classifier on a sample text    
sample = ["I  feel awesome"]
sample = tfidf.transform(sample).toarray()
sentiment = clf.predict(sample)
sentiment
