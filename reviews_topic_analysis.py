#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 09:54:30 2017

@author: angus
"""

import os
os.chdir('/home/angus/projects/review_analysis')

import numpy as np
import pandas as pd
import gzip
import nltk
import re
import string
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import lda


####################
### read in data ###
####################

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

review_data = getDF('reviews_data/reviews_Video_Games_5.json.gz')


#######################################
### create functions to do analysis ###
#######################################


### creating tf-idf and term weighted tf-idf matrices
# function which takes the list of documents and changes each to lower case, removing non-alphanumerics
def lowercase_and_remove_punct(documents):
    documents = [x.lower() for x in documents]
    documents = [re.sub('[^a-zA-Z _]+', ' ', x) for x in documents]
    return documents

# function to tokenize speeches
def tokenize(documents):
    documents = [word_tokenize(x) for x in documents]
    return documents

# function to remove stopwords
def remove_stopwords(tokenized_documents, stopwords):
    count = 0
    for doc in tokenized_documents:
        tokenized_documents[count] = [word for word in doc if word not in stopwords]
        count += 1
    return tokenized_documents

# function to stem a list of words
def word_stemmer(tokenized_documents):
    count = 0
    for doc in tokenized_documents:
        tokenized_documents[count] = [PorterStemmer().stem(word) for word in doc]
        count += 1
    return tokenized_documents

# function to put tokenized words back into one string
def untokenizer(tokenized_documents):
    count = 0
    for doc in tokenized_documents:
        tokenized_documents[count] = " ".join(doc)
        count += 1
    return tokenized_documents


### putting it all together and generating document term matrix
def dtm_generator(documents, stopwords):
    # preping docs
    lowercase = lowercase_and_remove_punct(documents)
    tokenized = tokenize(lowercase)
    rmstopwds = remove_stopwords(tokenized, stopwords)
    stemwords = word_stemmer(rmstopwds)
    untokened = untokenizer(stemwords)
    
    # generating dtm
    countvec = CountVectorizer()
    dtm = countvec.fit_transform(untokened)
    dtm = pd.DataFrame(dtm.toarray(), index=documents.index, columns=countvec.get_feature_names())
    return dtm

# using the dtm_generator to make tf-idf
def tfidf_document_level(documents, stopwords):
    dtm = dtm_generator(documents, stopwords)
    tf = dtm.applymap(lambda x: 0 if x == 0 else 1 + np.log(x))
    D = dtm.shape[0]
    df = dtm.astype(bool).sum(axis=0)
    idf = np.log(D/df)
    return tf*idf

# and to make a corpus level tf-idf
def tfidf_corpus_level(documents, stopwords):
    dtm = dtm_generator(documents, stopwords)
    tm = dtm.sum(axis=0)
    tf = tm.map(lambda x: 0 if x == 0 else 1 + np.log(x))
    D = dtm.shape[0]
    df = dtm.astype(bool).sum(axis=0)
    idf = np.log(D/df)
    return tf*idf



### functions for analysing best parameters
# function to calculate perplexity given a log-likelihood
def perplexity(log_likelihood, dtm):
    divide_by_wordcount = log_likelihood/(np.sum(dtm))
    perplexity = np.exp(-divide_by_wordcount)
    return perplexity

# setting function to fit the collapsed gibbs model and return perplexities
def collapsed_gibbs_perplexities(dtm, K, iterations, alpha, eta, random_state):
    # fitting the collapsed gibbs lda model
    dtm = np.asarray(dtm)
    collapsed_gibbs = lda.LDA(n_topics=K, n_iter=iterations, alpha=alpha, eta=eta, random_state=random_state)
    collapsed_gibbs.fit(dtm)
    
    # getting the log likelihoods from every tenth iteration and converting these into perplexities
    log_likelihoods = collapsed_gibbs.loglikelihoods_
    perplexities = pd.Series(log_likelihoods).map(lambda x: perplexity(x, dtm))
    
    return perplexities

### function to separate into topics and provide doc_topics and topic_words
def lda_do_topic_words(dtm, K, iterations, alpha, eta, random_state):
    dtm = np.asarray(dtm)
    model = lda.LDA(n_topics=K, n_iter=iterations, alpha=alpha, eta=eta, random_state=random_state)
    model.fit(dtm)
    # store parameters
    doc_topics = model.doc_topic_
    topic_words = model.topic_word_
    log_likelihoods = model.loglikelihoods_
    return(doc_topics, topic_words, log_likelihoods)



################
### analysis ###
################
# nb: struggles when you get to tens of thousands of documents, due to word stemming
# maybe don't want to do this for word cloud anyway??
stops = stopwords.words('english')

### five star reviews
reviews_5star = review_data['reviewText'][review_data['overall']==5]
reviews_5star = reviews_5star[np.asarray([len(doc) for doc in reviews_5star])!=0]

# create dtm matrix
dtm_5star = dtm_generator(reviews_5star[:2000], stopwords=stops)
vocab = list(dtm_5star)

# use tfidf weighting to remove non-meaningful but common words (e.g. 'game', 'play' etc)
tfidf_5star = tfidf_corpus_level(reviews_5star[:2000], stopwords=stops)
tfidf_cutoff_5star = np.median(tfidf_5star)

dtm_trunc_5star = np.asarray(dtm_5star)[:,np.asarray(tfidf_5star>tfidf_cutoff_5star)]
vocab_trunc = np.asarray(vocab)[np.asarray(tfidf_5star>tfidf_cutoff_5star)]


# set parameters
K=5
iterations=1000
alpha=0.001
eta=0.5
random_state=1

# test params
collapsed_gibbs_perplexities(dtm=dtm_trunc_5star, K=K, iterations=iterations, alpha=alpha, eta=eta, random_state=random_state)

# train model
doc_top_5star = lda_do_topic_words(dtm=dtm_trunc_5star, K=K, iterations=iterations, alpha=alpha, eta=eta, random_state=random_state)

# store words in order of most likely for each topic
doc_topics_5star = doc_top_5star[0]
topic_words_5star = doc_top_5star[1]

ordered_vocab_5star = np.asarray(vocab_trunc)[np.argsort(-topic_words_5star)]
word_probs_5star = np.sort(topic_words_5star)[:,::-1]

# save each individual topic separately and rescale probs to start at 1
for r in range(ordered_vocab_5star.shape[0]):
    
    # rescaling probabilities so that highest is 1 in order to make font sizing work
    font_size = word_probs_5star[r,:200]/word_probs_5star[r,0]
    
    # creating 'opacity' so that words are virtually invisible by around word 200
    opacity = 1 - (1 - 0.1)*(word_probs_5star[r,0]-word_probs_5star[r,:200])/(word_probs_5star[r,0]-word_probs_5star[r,200])
    
    # save file
    np.savetxt("interface/data/5star_topic" + str(r+1) + ".csv",
               np.row_stack((['word','probability','font_size','opacity'], np.column_stack((ordered_vocab_5star[r,:200], word_probs_5star[r,:200], font_size, opacity)))),
               delimiter=",", fmt='%s')





### four star reviews
reviews_4star = review_data['reviewText'][review_data['overall']==4]
reviews_4star = reviews_4star[np.asarray([len(doc) for doc in reviews_4star])!=0]

# create dtm matrix
dtm_4star = dtm_generator(reviews_4star[:5000], stopwords=stops)
vocab = list(dtm_4star)

# use tfidf weighting to remove non-meaningful but common words (e.g. 'game', 'play' etc)
tfidf_4star = tfidf_corpus_level(reviews_4star[:5000], stopwords=stops)
tfidf_cutoff_4star = np.median(tfidf_4star)

dtm_trunc_4star = np.asarray(dtm_4star)[:,np.asarray(tfidf_4star>tfidf_cutoff_4star)]
vocab_trunc = np.asarray(vocab)[np.asarray(tfidf_4star>tfidf_cutoff_4star)]


# set parameters
K=5
iterations=1000
alpha=0.001
eta=0.5
random_state=1

# train model
doc_top_4star = lda_do_topic_words(dtm=dtm_trunc_4star, K=K, iterations=iterations, alpha=alpha, eta=eta, random_state=random_state)

# store words in order of most likely for each topic
doc_topics_4star = doc_top_4star[0]
topic_words_4star = doc_top_4star[1]

ordered_vocab_4star = np.asarray(vocab_trunc)[np.argsort(-topic_words_4star)]
word_probs_4star = np.sort(topic_words_4star)[:,::-1]

# save each individual topic separately and rescale probs to start at 1
for r in range(ordered_vocab_4star.shape[0]):
    
    # rescaling probabilities so that highest is 1 in order to make font sizing work
    font_size = word_probs_4star[r,:200]/word_probs_4star[r,0]
    
    # creating 'opacity' so that words are virtually invisible by around word 200
    opacity = 1 - (1 - 0.1)*(word_probs_4star[r,0]-word_probs_4star[r,:200])/(word_probs_4star[r,0]-word_probs_4star[r,200])
    
    # save file
    np.savetxt("interface/data/4star_topic" + str(r+1) + ".csv",
               np.row_stack((['word','probability','font_size','opacity'], np.column_stack((ordered_vocab_4star[r,:200], word_probs_4star[r,:200], font_size, opacity)))),
               delimiter=",", fmt='%s')




### three star reviews
reviews_3star = review_data['reviewText'][review_data['overall']==3]
reviews_3star = reviews_3star[np.asarray([len(doc) for doc in reviews_3star])!=0]

# create dtm matrix
dtm_3star = dtm_generator(reviews_3star[:2000], stopwords=stops)
vocab = list(dtm_3star)

# use tfidf weighting to remove non-meaningful but common words (e.g. 'game', 'play' etc)
tfidf_3star = tfidf_corpus_level(reviews_3star[:2000], stopwords=stops)
tfidf_cutoff_3star = np.median(tfidf_3star)

dtm_trunc_3star = np.asarray(dtm_3star)[:,np.asarray(tfidf_3star>tfidf_cutoff_3star)]
vocab_trunc = np.asarray(vocab)[np.asarray(tfidf_3star>tfidf_cutoff_3star)]


# set parameters
K=5
iterations=1000
alpha=0.001
eta=0.5
random_state=1

# train model
doc_top_3star = lda_do_topic_words(dtm=dtm_trunc_3star, K=K, iterations=iterations, alpha=alpha, eta=eta, random_state=random_state)

# store words in order of most likely for each topic
doc_topics_3star = doc_top_3star[0]
topic_words_3star = doc_top_3star[1]

ordered_vocab_3star = np.asarray(vocab_trunc)[np.argsort(-topic_words_3star)]
word_probs_3star = np.sort(topic_words_3star)[:,::-1]

# save each individual topic separately and rescale probs to start at 1
for r in range(ordered_vocab_3star.shape[0]):
    
    # rescaling probabilities so that highest is 1 in order to make font sizing work
    font_size = word_probs_3star[r,:200]/word_probs_3star[r,0]
    
    # creating 'opacity' so that words are virtually invisible by around word 200
    opacity = 1 - (1 - 0.1)*(word_probs_3star[r,0]-word_probs_3star[r,:200])/(word_probs_3star[r,0]-word_probs_3star[r,200])
    
    # save file
    np.savetxt("interface/data/3star_topic" + str(r+1) + ".csv",
               np.row_stack((['word','probability','font_size','opacity'], np.column_stack((ordered_vocab_3star[r,:200], word_probs_3star[r,:200], font_size, opacity)))),
               delimiter=",", fmt='%s')





### two star reviews
reviews_2star = review_data['reviewText'][review_data['overall']==2]
reviews_2star = reviews_2star[np.asarray([len(doc) for doc in reviews_2star])!=0]

# create dtm matrix
dtm_2star = dtm_generator(reviews_2star[:5000], stopwords=stops)
vocab = list(dtm_2star)

# use tfidf weighting to remove non-meaningful but common words (e.g. 'game', 'play' etc)
tfidf_2star = tfidf_corpus_level(reviews_2star[:5000], stopwords=stops)
tfidf_cutoff_2star = np.median(tfidf_2star)

dtm_trunc_2star = np.asarray(dtm_2star)[:,np.asarray(tfidf_2star>tfidf_cutoff_2star)]
vocab_trunc = np.asarray(vocab)[np.asarray(tfidf_2star>tfidf_cutoff_2star)]


# set parameters
K=5
iterations=1000
alpha=0.001
eta=0.5
random_state=1

# train model
doc_top_2star = lda_do_topic_words(dtm=dtm_trunc_2star, K=K, iterations=iterations, alpha=alpha, eta=eta, random_state=random_state)

# store words in order of most likely for each topic
doc_topics_2star = doc_top_2star[0]
topic_words_2star = doc_top_2star[1]

ordered_vocab_2star = np.asarray(vocab_trunc)[np.argsort(-topic_words_2star)]
word_probs_2star = np.sort(topic_words_2star)[:,::-1]

# save each individual topic separately and rescale probs to start at 1
for r in range(ordered_vocab_2star.shape[0]):
    
    # rescaling probabilities so that highest is 1 in order to make font sizing work
    font_size = word_probs_2star[r,:200]/word_probs_2star[r,0]
    
    # creating 'opacity' so that words are virtually invisible by around word 200
    opacity = 1 - (1 - 0.1)*(word_probs_2star[r,0]-word_probs_2star[r,:200])/(word_probs_2star[r,0]-word_probs_2star[r,200])
    
    # save file
    np.savetxt("interface/data/2star_topic" + str(r+1) + ".csv",
               np.row_stack((['word','probability','font_size','opacity'], np.column_stack((ordered_vocab_2star[r,:200], word_probs_2star[r,:200], font_size, opacity)))),
               delimiter=",", fmt='%s')






### one star reviews
reviews_1star = review_data['reviewText'][review_data['overall']==1]
reviews_1star = reviews_1star[np.asarray([len(doc) for doc in reviews_1star])!=0]

# create dtm matrix
dtm_1star = dtm_generator(reviews_1star[:5000], stopwords=stops)
vocab = list(dtm_1star)

# use tfidf weighting to remove non-meaningful but common words (e.g. 'game', 'play' etc)
tfidf_1star = tfidf_corpus_level(reviews_1star[:5000], stopwords=stops)
tfidf_cutoff_1star = np.median(tfidf_1star)

dtm_trunc_1star = np.asarray(dtm_1star)[:,np.asarray(tfidf_1star>tfidf_cutoff_1star)]
vocab_trunc = np.asarray(vocab)[np.asarray(tfidf_1star>tfidf_cutoff_1star)]


# set parameters
K=5
iterations=1000
alpha=0.001
eta=0.5
random_state=1

# train model
doc_top_1star = lda_do_topic_words(dtm=dtm_trunc_1star, K=K, iterations=iterations, alpha=alpha, eta=eta, random_state=random_state)

# store words in order of most likely for each topic
doc_topics_1star = doc_top_1star[0]
topic_words_1star = doc_top_1star[1]

ordered_vocab_1star = np.asarray(vocab_trunc)[np.argsort(-topic_words_1star)]
word_probs_1star = np.sort(topic_words_1star)[:,::-1]

# save each individual topic separately and rescale probs to start at 1
for r in range(ordered_vocab_1star.shape[0]):
    
    # rescaling probabilities so that highest is 1 in order to make font sizing work
    font_size = word_probs_1star[r,:200]/word_probs_1star[r,0]
    
    # creating 'opacity' so that words are virtually invisible by around word 200
    opacity = 1 - (1 - 0.1)*(word_probs_1star[r,0]-word_probs_1star[r,:200])/(word_probs_1star[r,0]-word_probs_1star[r,200])
    
    # save file
    np.savetxt("interface/data/1star_topic" + str(r+1) + ".csv",
               np.row_stack((['word','probability','font_size','opacity'], np.column_stack((ordered_vocab_1star[r,:200], word_probs_1star[r,:200], font_size, opacity)))),
               delimiter=",", fmt='%s')























