# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:21:47 2020

@author: sumit
"""
#As usual, I start out by loading all the packages
import nltk
import string
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
tweets = pd.read_csv('E:/aug15_sample.csv')

#This is what the dataset looks like:
tweets.head()

#Tokening text and getting rid of all the stopwords. Finally store most frequently mentioned keywords in rslt variable.

top_N = 30
stopwords = nltk.corpus.stopwords.words('english')
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
words = (tweets['full_text']
           .str.lower()
           .replace([r'\|', RE_stopwords, r"(&amp)|,|;|\"|\.|\?|’|!|'|:|-|\\|/|https"], [' ', ' ', ' '], regex=True)
           .str.cat(sep=' ')
           .split()
)

rslt = pd.DataFrame(Counter(words).most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')

rslt = rslt.iloc[1:]

#Here comes the top keywords uttered by Twitter users
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = [30.0, 20.0]
ax = sns.barplot(y=rslt.index, x='Frequency', data=rslt)
ax.set_xlabel("Frequency",fontsize=20)
ax.set_ylabel("Words",fontsize=20)
ax.tick_params(labelsize=30)

#Most frequently used hashtags
tags = (tweets['hashtags']
           .str.lower()
           .str.cat(sep=' ')
           .split()
)

hashtgs = pd.DataFrame(Counter(tags).most_common(top_N),
                    columns=['Hashtags', 'Frequency']).set_index('Hashtags')
hashtgs = hashtgs.iloc[1:]
hashtgs

#Plotting most frequently used hashtags
sns.set_style("darkgrid")
ax = sns.barplot(y=hashtgs.index, x='Frequency', data=hashtgs)
ax.set_xlabel("Frequency",fontsize=20)
ax.set_ylabel("Hashtag",fontsize=20)
ax.tick_params(labelsize=30)


tweets['created_at'] = pd.to_datetime(tweets['created_at'])
tweets = tweets.set_index('created_at')
df = tweets[['id']]
tweet_volume = df.resample('10min').count()

#Let's see the tweets traffic timing 
ax = sns.pointplot(x=tweet_volume.index, y='id', data=tweet_volume)
ax.set_xlabel("When did they tweet?",fontsize=20)
ax.set_ylabel("How many tweets?",fontsize=20)

ax.tick_params(labelsize=25)

for item in ax.get_xticklabels():
    item.set_rotation(90)


influential = tweets[['user_name', 'followers_count']]
influential = influential.sort_values('followers_count', ascending=False)


#We have a list of users in the conversation who has the biggest number of followers. Not surprisingly, all of them were media organizations such as CNN or Reuters.
influential.groupby('user_name').first().sort_values(by='followers_count', ascending=False)[:10]

#These are the users who tweeted the most over the span of 3 hours. We have three users who tweeted over 100 times in 3 hours! Some people are really passionate about this topic.
tweets['screen_name'].value_counts()[:10]

# clustering algorithms
pd.options.mode.chained_assignment = None
# nltk for nlp
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# list of stopwords like articles, preposition
stop = set(stopwords.words('english'))
from string import punctuation
from collections import Counter

def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
        
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent

        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        tokens = list(filter(lambda t: t not in punctuation, tokens))
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', u'amp', u'https',
                                                u'via', u"'re"], tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    except Error as e:
        print(e)
        
tweets['tokens'] = tweets['full_text'].map(tokenizer)
for full_text, tokens in zip(tweets['full_text'].head(5), tweets['tokens'].head(5)):
    print('full text:', full_text)
    print('tokens:', tokens)
    print()         


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 2))
vz = vectorizer.fit_transform(list(tweets['full_text']))
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']
tfidf.tfidf.hist(bins=50, figsize=(15,7))


tfidf.sort_values(by=['tfidf'], ascending=False).head(30)

#We used k-means clustering algorithms to generate a list of words that appear frequently together, and the results are shown above.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.cluster import MiniBatchKMeans

num_clusters = 10
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)

#Here we can see that there is a thread of conversation that we couldn’t detect from the word frequency list. One example of this is Cluster #3, where a pocket of people expressed their displeasure with CNN coverage. K-means clustering is surely a great way to complement our word frequency tally.
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in sorted_centroids[i, :10]:
        aux += terms[j] + ' | '
    print(aux)
    print()