#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from os.path import isfile, join
import json
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string 
import pickle

import os


DATA = "NLP"


bib_text_path_1 = "/home1/tirthankar/journal/"+DATA+"/train/pos_bib"
bib_text_path_2 = "/home1/tirthankar/journal/"+DATA+"/train/neg_bib"



# In[4]:


def preprocess(sentence):
    words = []  
    w = word_tokenize(sentence)
    w = [word.lower() for word in w]
    w = [word for word in w if word.isalpha()]
    stop_words = stopwords.words('english')
    w = [word for word in w if not word in stop_words]
    words.extend(w)
    return words

    
def bagofwords(sentence, words):
    bag = np.zeros(len(words))
    for sw in sentence:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] = 1                
    return np.array(bag)


# In[7]:


def get_unigrams(x,path,paper_title,prefix):

    for file in x:
        key = prefix+file
        
        with open(join(path,file)) as input_file:
            data = json.load(input_file)
            if data['metadata']['title'] is not None:
                paper_title[key] = data['metadata']['title']
            else:
                paper_title[key] = "None"
            
            paper_title[key] = preprocess(paper_title[key])
           


# In[8]:


#for  jnca get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_1)
prefix = 'pos_'
paper_title = {}
get_unigrams(x,bib_text_path_1,paper_title,prefix)
#for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_2)
prefix = 'neg_'
get_unigrams(x,bib_text_path_2,paper_title,prefix)


# In[9]:


def generate_vocabulary(vocabulary_name,document_unigram,prefix,x):
    for file in x:
        vocabulary_name = vocabulary_name + document_unigram[prefix+file]
    return vocabulary_name


# In[10]:


title_vocabulary = []

x = listdir(bib_text_path_1)
title_vocabulary = generate_vocabulary(title_vocabulary,paper_title,'pos_',x)

x = listdir(bib_text_path_2)
title_vocabulary = generate_vocabulary(title_vocabulary,paper_title,'neg_',x)


# In[11]:



title_vocabulary1 = sorted(list(set(title_vocabulary)))


# In[12]:


c = 0
for i in set(title_vocabulary1):
    c = c+1
    if c % 100 == 0:
        print(c)
    if title_vocabulary.count(i) < 2:
        title_vocabulary1.remove(i)
        


# In[13]:


print(len(title_vocabulary1))


# In[14]:



print(len(title_vocabulary))
print(len(title_vocabulary1))


# In[15]:



pickle.dump(title_vocabulary1, open(DATA+"_title_vocabulary1.p", "wb"))  # save it into a file named save.p



# In[16]:


# feature vector = title + bib_title+bib_venue

def get_feature_vector(x,paper_title,prefix_value,feature_vector):
    c = 0
    for file in x:
        prefix = prefix_value+file
        c = c + 1
        if c%100 ==0 : 
            print(c)
        feature_vector[prefix] = bagofwords(paper_title[prefix],title_vocabulary1)


# In[17]:


feature_vector = {}
x = listdir(bib_text_path_1)
get_feature_vector(x,paper_title,'pos_',feature_vector)
x = listdir(bib_text_path_2)
get_feature_vector(x,paper_title,'neg_',feature_vector)



# In[ ]:



pickle.dump(feature_vector, open(DATA+"_train_title_feature_vector.p", "wb"))  # save it into a file named save.p


# In[ ]:



bib_text_path_1 = "/home1/tirthankar/journal/"+DATA+"/test/pos_bib"
bib_text_path_2 = "/home1/tirthankar/journal/"+DATA+"/test/neg_bib"



#for  jnca get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_1)
prefix = 'pos_'
paper_title = {}
get_unigrams(x,bib_text_path_1,paper_title,prefix)
#for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_2)
prefix = 'neg_'
get_unigrams(x,bib_text_path_2,paper_title,prefix)


# In[ ]:


feature_vector = {}
x = listdir(bib_text_path_1)
get_feature_vector(x,paper_title,'pos_',feature_vector)
x = listdir(bib_text_path_2)
get_feature_vector(x,paper_title,'neg_',feature_vector)

pickle.dump(feature_vector, open(DATA+"_test_title_feature_vector.p", "wb"))  # save it into a file named save.p



