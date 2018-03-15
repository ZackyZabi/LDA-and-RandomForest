from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import subprocess
import nltk
import random
import string
import re

train = pd.read_csv('NewTrain.csv')
test = pd.read_csv('NewTest.csv')

def description_to_wordlist(bug,remove_stopwords=False):
		bug_text=BeautifulSoup(bug,"lxml").get_text()
		bug_text=re.sub("[^a-zA-Z]"," ",bug_text)
		words=bug_text.lower().split()
		if remove_stopwords:
			stops=set(stopwords.words("english"))
			words=[w for w in words if not w in stops]
		return (words)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

#this is for changing the training data's desired column to dictionary
clean_train_bug = []
for i in xrange( 0, len(train["Description"])):
	clean_train_bug.append(" ".join(description_to_wordlist(train["Description"][i], True)))

doc_clean_train = [clean(doc).split() for doc in clean_train_bug]
dictionary_train = corpora.Dictionary(doc_clean_train)
doc_term_matrix_train = [dictionary_train.doc2bow(doc) for doc in doc_clean_train]

#this is for changing the testing data's desired column to dictionary
clean_test_bug = []
for i in xrange( 0, len(test["Description"])):
	clean_test_bug.append(" ".join(description_to_wordlist(test["Description"][i], True)))

doc_clean_test = [clean(doc).split() for doc in clean_test_bug]
dictionary_test = corpora.Dictionary(doc_clean_test)
doc_term_matrix_test = [dictionary_test.doc2bow(doc) for doc in doc_clean_test]

#finding topics based on the dictionary of training data using LDA
lda = models.ldamodel.LdaModel(doc_term_matrix_train, num_topics=99, id2word = dictionary_train, passes=5)
subprocess.call('reset')
#updating LDA with dictionary of testing data
lda.update(doc_term_matrix_test)

#it is created to create matrix
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

#finding the topics,product and component of training data
#topics
topicTrain = []
for i in xrange(0, len(train["Description"])):
    r = lda.get_document_topics(doc_term_matrix_train[i])
    r = random.choice(r)
    r = r[0]
    r = str(r)
    np.asarray(r)
    topicTrain.append(''.join(r))
#product
productTrain = []
for i in xrange(0, len(train["Description"])):
	    productTrain.append(train["Product"][i])
#component
componentTrain = []
for i in xrange(0, len(train["Description"])):
	    componentTrain.append(train["Component"][i])
#finding affinity of the training data
affinityTrain = []
for i in xrange(0, len(topicTrain)):
            m = topicTrain[i]+" "+productTrain[i]+" "+componentTrain[i]
            np.asarray(m)
            affinityTrain.append(m)

#finding the topics,product and component of testing data
#topics
topicTest = []
for i in xrange(0, len(test["Description"])):
    s = lda.get_document_topics(doc_term_matrix_test[i])
    s = random.choice(s)
    s = s[0]
    s = str(s)
    np.asarray(s)
    topicTest.append(''.join(s))
#product
productTest = []
for i in xrange(0, len(test["Description"])):
	    productTest.append(test["Product"][i])
#component
componentTest = []
for i in xrange(0, len(test["Description"])):
	    componentTest.append(test["Component"][i])
#finding affinity of the testing data
affinityTest = []
for i in xrange(0, len(topicTest)):
            m = topicTest[i]+" "+productTest[i]+" "+componentTest[i]
            np.asarray(m)
            affinityTest.append(m)

#training the affinityTrain using randomForest after vectorizing
train_data = vectorizer.fit_transform(affinityTrain)
np.asarray(train_data)
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data, train["Fixer"] )
#testing the affinityTest using randomForest after vectorizing
test_data = vectorizer.transform(affinityTest)
np.asarray(test_data)
result = forest.predict(test_data)

output = pd.DataFrame( data={"Product":test["Product"], "Component":test["Component"], "description":test["Description"], "fixer":result} )
print output
