# Sentimental-Analysis
Detecting Depression from Social Media posts using Machine Learning Techniques

# Abstract
The use of Social Network Sites (SNS), particularly by the younger generations, has seen rapid growth nowadays. Social media platforms like Twitter which is a microblogging tool enable its users to express their feelings, emotions, and opinions through short text messages. Detecting the emotions in a text can help one identify the anxiety and depression of an individual. Depression is a mental health problem that can happen to anyone, at any age. There is a lack of systematic and efficient methods to identify the psychological state of an individual. With more than 58 million tweets generated daily, Twitter can be used to detect the sign of depression in a faster way. Recent studies have demonstrated that Twitter can be used to prevent one from taking an extreme step. Our Proposed depression detection and prevention system can detect any depression-related words or phrases from Tweets and also classify the tweets into depressed or not. This system is proposed to diagnose depression and prevent it.

# Sentiment Analysis Architecture

![image](https://user-images.githubusercontent.com/49193241/125484516-6adfdff6-525b-4dd4-9410-a68c53c76e3b.png)

# Training 

In the training phase, A pre-trained classifier was developed using a labeled dataset.Sentiment 140 datasets from kaggle.com were taken as a labeled dataset for pre-training our model. It contains 1,600,000 tweets extracted using the twitter API. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to train our classifier to detect the sentiment. Out of 16,00,000 tweets, only 3,00,000 tweets were chosen for our study due to limited computational power. As Target and text were the only two columns that were relevant and required for pre-training our model, the rest of the columns were removed and a new data set consisting of 1.5 lakhs positive and negative tweets each along with their respective sentiments was created.Bag of words,Count vectorizer and TF-IDF models were generated for the data which was then split by 70-30 to training and test sets. Finally, the Logistic regression model was to train the classifier. After applying Logistic Regression, We obtained a pre-trained model that can be used on twitter data in the testing phase to calculate Sentiment Scores.

# Testing Phase

Once we have a pre-trained classifier, Our Next task was to move on to testing Phase.The first step in the testing phase was to collect the data from twitter for depression classification. For this, a twitter developer account was created and necessary approval was obtained to scrape and use the data from Twitter users.In this study, the Twitter API was used to gather tweets from users. Tweets from users with keywords such as 'depression' ,'anxiety' ,'mental health', 'suicide', 'stress' ,'sad' ,'sadness' ,'anxious' ,'stressful' ,'depressed', 'worthless', 'stressed' or 'anxiousness' ,'boredom' ,'lonely' , 'depressing' ,'devastated' , 'frustrated' ,'frustrating' ,'hopeless' and 'unhappy' were fetched through twitter API. A total of 65,499 tweets were fetched based on keyword matching. The output of twitter API will be in JSON format.Then the fetched tweets were preprocessed and were passed to pre-trained logistic classifier which assigns sentiment to the tweet. Sentiment 0 means that the person is depressed and Sentiment 1 means that the person is not depressed.After getting Sentiment scores, we applied Logistic regression, Naive bayes,XG boost and LSTM models.

# Results

Logistic Regression:

![image](https://user-images.githubusercontent.com/49193241/125488574-401567d4-d717-4289-8edf-05d8275d7ded.png)

Multinomial Naïve Bayes:

![image](https://user-images.githubusercontent.com/49193241/125488663-76c2db02-c715-418d-809a-532e3647af53.png)

XGBoost:

![image](https://user-images.githubusercontent.com/49193241/125489221-01571e2c-aca1-47f5-b41d-1f3bd77dbc77.png)

LSTM:

![image](https://user-images.githubusercontent.com/49193241/125488357-566b4735-fbee-4f22-8b2b-6124593fadcf.png)

# Conclusion and Future work
We have demonstrated the potential of using twitter as a tool for measuring and predicting depressive disorder in individuals. First, we used Labelled Sentiment 140 dataset to pre-train our model using logistic regression. Then we extracted the tweets based on keywords about depression from twitter. Next, we proposed various preprocessing techniques for cleaning the data such as Regex, Stop word removal, Stemming and so on. Then we fed the cleaned data into a pre-trained classifier to assign sentiment scores to the tweets. After that, we used the Bag of Words and TF IDF approach to covert the data into an intermediate matrix so it can be fed to the classifier. Finally, we used four classifiers which may predict the likelihood of depression within an individual.We yielded promising results with a 95% being the highest classification accuracy. The following specific results were obtained: a Logistic regression algorithm yielded an accuracy score of 0.95 which was closely followed by XGBoost with a 0.94 accuracy score. A Multinomial approach to Naïve Bayes classifier yielded an accuracy score of 0.80 which was much lesser than the other two models.In this study we have considered only the text part and emoticons are ignored, emoticons also play major role in the detection of depression. In near future, we hope to include emoticons along with text and produce better accuracies. The system focused only on English sentences, but Twitter has many international users. It should be possible to use this system to classify depression in other languages.




