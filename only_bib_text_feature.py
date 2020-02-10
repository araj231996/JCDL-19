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
import numpy as np
import itertools as it
import tensorflow as tf
import tensorflow_hub as hub


DATA = "E3"


bib_text_path_1 = "/home1/tirthankar/Exp3/train/pos_bib"
bib_text_path_2 = "/home1/tirthankar/Exp3/train/neg_bib"


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


# In[6]:


def get_unigrams(x,path,bib_title,bib_venue,prefix):

    for file in x:
        key = prefix+file
        bib_title[key] = ""
        bib_venue[key] = ""
        with open(join(path,file)) as input_file:
            data = json.load(input_file)
            if data['metadata']['references'] is not None:
                for ref in data['metadata']['references']:
                    if ref['title'] is not None:
                        bib_title[key] = bib_title[key]+ref['title']+" "
                    else:
                        bib_title[key] = "None"
                    if ref['venue'] is not None:
                        bib_venue[key] = bib_venue[key]+ref['venue']+" "
                    else:
                        bib_venue[key] = "None"
            bib_title[key] = preprocess(bib_title[key])
            bib_venue[key] = preprocess(bib_venue[key])



# In[7]:


#for  jnca get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_1)
prefix = 'pos_'
bib_title = {}
bib_venue = {}
get_unigrams(x,bib_text_path_1,bib_title,bib_venue,prefix)
#for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_2)
prefix = 'neg_'
get_unigrams(x,bib_text_path_2,bib_title,bib_venue,prefix)


# In[8]:


def generate_vocabulary(vocabulary_name,document_unigram,prefix,x):
    for file in x:
        vocabulary_name = vocabulary_name + document_unigram[prefix+file]
    return vocabulary_name


# In[9]:


bib_title_vocabulary = []
bib_venue_vocabulary = []

x = listdir(bib_text_path_1)
bib_title_vocabulary = generate_vocabulary(bib_title_vocabulary,bib_title,'pos_',x)
bib_venue_vocabulary = generate_vocabulary(bib_venue_vocabulary,bib_venue,'pos_',x)

x = listdir(bib_text_path_2)
bib_title_vocabulary = generate_vocabulary(bib_title_vocabulary,bib_title,'neg_',x)
bib_venue_vocabulary = generate_vocabulary(bib_venue_vocabulary,bib_venue,'neg_',x)



# In[10]:



bib_title_vocabulary1 = sorted(list(set(bib_title_vocabulary)))
bib_venue_vocabulary1 = sorted(list(set(bib_venue_vocabulary)))


# In[11]:


c = 0
for i in set(bib_title_vocabulary1):
    c = c+1
    if c % 100 == 0:
        print(c)
    if bib_title_vocabulary.count(i) < 6:
        bib_title_vocabulary1.remove(i)
        


# In[12]:


c = 0
for i in set(bib_venue_vocabulary1):
    c = c+1
    if c % 100 == 0:
        print(c)
    if bib_venue_vocabulary.count(i) <2:
        bib_venue_vocabulary1.remove(i)
        


# In[13]:


print(len(bib_venue_vocabulary))
print(len(bib_venue_vocabulary1))
print(len(bib_title_vocabulary))
print(len(bib_title_vocabulary1))


# In[14]:



pickle.dump(bib_venue_vocabulary1, open(DATA+"_bib_venue_vocabulary1.p", "wb"))  # save it into a file named save.p

pickle.dump(bib_title_vocabulary1, open(DATA+"_bib_title_vocabulary1.p", "wb"))  # save it into a file named save.p



# In[15]:


# feature vector = title + bib_title+bib_venue

def get_feature_vector(x,bib_title,bib_venue,prefix_value,feature_vector):
    c = 0
    for file in x:
        prefix = prefix_value+file
        c = c + 1
        if c%100 ==0 : 
            print(c)
        feature_vector[prefix] = np.concatenate(
                                (bagofwords(bib_title[prefix],bib_title_vocabulary1),
                                bagofwords(bib_venue[prefix],bib_venue_vocabulary1)))  


# In[16]:


feature_vector = {}
x = listdir(bib_text_path_1)
get_feature_vector(x,bib_title,bib_venue,'pos_',feature_vector)
x = listdir(bib_text_path_2)
get_feature_vector(x,bib_title,bib_venue,'neg_',feature_vector)



# In[17]:



pickle.dump(feature_vector, open(DATA+"_train_only_bib_feature_vector.p", "wb"))  # save it into a file named save.p



bib_text_path_1 = "/home1/tirthankar/Exp3/test/pos_bib"
bib_text_path_2 = "/home1/tirthankar/Exp3/test/neg_bib"




# In[19]:


#for  jnca get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_1)
prefix = 'pos_'
bib_title = {}
bib_venue = {}
get_unigrams(x,bib_text_path_1,bib_title,bib_venue,prefix)
#for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_2)
prefix = 'neg_'
get_unigrams(x,bib_text_path_2,bib_title,bib_venue,prefix)


# In[20]:


feature_vector = {}
x = listdir(bib_text_path_1)
get_feature_vector(x,bib_title,bib_venue,'pos_',feature_vector)
x = listdir(bib_text_path_2)
get_feature_vector(x,bib_title,bib_venue,'neg_',feature_vector)



pickle.dump(feature_vector, open(DATA+"_0_test_only_bib_feature_vector.p", "wb"))  # save it into a file named save.p



bib_text_path_1 = "/home1/tirthankar/Exp3/test/pos1_bib"
bib_text_path_2 = "/home1/tirthankar/Exp3/test/neg_bib"




# In[19]:


#for  jnca get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_1)
prefix = 'pos_'
bib_title = {}
bib_venue = {}
get_unigrams(x,bib_text_path_1,bib_title,bib_venue,prefix)
#for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
x = listdir(bib_text_path_2)
prefix = 'neg_'
get_unigrams(x,bib_text_path_2,bib_title,bib_venue,prefix)


# In[20]:


feature_vector = {}
x = listdir(bib_text_path_1)
get_feature_vector(x,bib_title,bib_venue,'pos_',feature_vector)
x = listdir(bib_text_path_2)
get_feature_vector(x,bib_title,bib_venue,'neg_',feature_vector)



pickle.dump(feature_vector, open(DATA+"_1_test_only_bib_feature_vector.p", "wb"))  # save it into a file named save.p



