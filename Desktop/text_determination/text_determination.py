
# coding: utf-8

# In[ ]:


import os
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

from pprint import pprint
pprint(list(newsgroups_train.target_names))


# In[ ]:


newsgroups_train
   


# In[ ]:


x = newsgroups_train.data
categories = newsgroups_train.target


# In[ ]:


import nltk


# In[ ]:


from nltk.tokenize import sent_tokenize , word_tokenize


# In[ ]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x , categories , test_size=0.25, random_state=0)


# In[ ]:


# A list of common english words which should not affect predictions
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
             'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount',
             'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around',
             'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
             'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
             'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
             'describe', 'detail', 'did', 'do', 'does', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg',
             'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone',
             'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for',
             'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
             'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
             'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less',
             'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
             'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine',
             'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once',
             'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
             'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 
             'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
             't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
             'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this',
             'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
             'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
             'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom',
             'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
             'yourselves']


# In[ ]:


text_into_words = []
for documents in x_train:
    text_into_words.append(word_tokenize(documents))


# In[ ]:


text_into_words


# In[ ]:


punctuations = list(string.punctuation)


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


nltk.download("stopwords")


# In[ ]:


stop = stopwords.words("english")


# In[ ]:


stop = stop + punctuations


# In[ ]:


clean_words = []
for text in text_into_words:
    clean = [w for w in text if not w in stop]
    clean_words.append(clean)


# In[ ]:


clean_words


# In[ ]:


clean_documents = []
i = 0
for text in clean_words:
    clean_documents.append((text , y_train[i]))
    i = i+1


# In[ ]:


clean_documents


# In[ ]:


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


# In[ ]:


lemmatizer = WordNetLemmatizer() 
def clean_documents(words):
    output_documents = []
    for w in words:
        if w.lower() not in stop:
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w , pos = get_simpler_pos(pos[0][1]))
            output_documents.append(clean_word.lower())
    return output_documents


# In[ ]:


def get_simpler_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[ ]:


from nltk.corpus import sentiwordnet as swm , wordnet


# In[ ]:


nltk.download("wordnet")


# In[ ]:


final_word_list = [clean_documents(text) for text in clean_words]


# In[ ]:


words = clean_documents(clean_words[0])
words


# In[ ]:


clean_documents_1 = []
i=0
for text in final_word_list:
    clean_documents_1.append((text , categories[i]))
    i = i+1


# In[ ]:


all_words = []

for word in final_word_list:
    all_words = all_words + word


# In[ ]:


freq = nltk.FreqDist(all_words)
freq


# In[ ]:


common = freq.most_common(3000)
common


# In[ ]:


features = [i[0] for i in common]


# In[ ]:


features


# In[ ]:


def get_features_dict(words):
    current_features = {}
    words_set = set(words)
    for w in features:
        current_features[w] = w in words_set
    return current_features


# In[ ]:


training_data = [(get_features_dict(doc) , category) for doc , category in clean_documents_1]


# In[ ]:


from nltk import NaiveBayesClassifier


# In[ ]:


classifier =  NaiveBayesClassifier.train(training_data)


# In[ ]:


test_text_into_words = []
for documents in x_test:
    test_text_into_words.append(word_tokenize(documents))


# In[ ]:


test_text_into_words


# In[ ]:


clean_words_test = []
for text in test_text_into_words:
    clean = [w for w in text if not w in stop]
    clean_words_test.append(clean)


# In[ ]:


final_word_list_text = [clean_documents(text) for text in clean_words_test]


# In[ ]:


clean_words_test


# In[ ]:


clean_documents_test = []
i=0
for text in final_word_list_text:
    clean_documents_test.append((text , y_test[i]))
    i = i+1


# In[ ]:


clean_documents_test


# In[ ]:


x_test


# In[ ]:


testing_data = [(get_features_dict(doc) , category) for doc , category in clean_documents_test]


# In[ ]:


testing_data


# In[ ]:


nltk.classify.accuracy(classifier , testing_data)


# In[ ]:


classifier.show_most_informative_features(100)


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


from nltk.classify.scikitlearn import SklearnClassifier


# In[ ]:


svc = SVC()
sklearn_classifier = SklearnClassifier(svc)


# In[ ]:


sklearn_classifier.train(training_data)


# In[ ]:


nltk.classify.accuracy(sklearn_classifier , testing_data)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier()
sklearn_classifier_new = SklearnClassifier(rfc)


# In[ ]:


sklearn_classifier_new.train(training_data)


# In[ ]:


nltk.classify.accuracy(sklearn_classifier_new , testing_data)


# In[ ]:


document_input = input()


# In[ ]:


new_document = word_tokenize(document_input)


# In[ ]:


cleanwords = [w for w in text if not w in stop]


# In[ ]:


final_word_list_text = [clean_documents(text) for text in cleanwords]


# In[ ]:


testing_data_new_document = [(get_features_dict(doc) , category) for doc , category in clean_documents_test]


# In[ ]:


classifier.classify(new_document) # to predict the type of documents from the above goiven new document

