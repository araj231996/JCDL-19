
# coding: utf-8

# In[2]:


from os import listdir
from os.path import isfile, join
import json
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import string 
import pickle

import numpy as np
import itertools as it
import tensorflow as tf
import tensorflow_hub as hub
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[1]:


DATA = "CSI"


# In[11]:


bib_text_path_1 = "/home1/tirthankar/btpfinal/dataset/"+DATA+"/train/positive_bib_part/"
bib_text_path_2 = "/home1/tirthankar/btpfinal/dataset/"+DATA+"/train/negative_bib_part/"


# In[12]:


def get_sentences(x,path,word_document_list,file_name,prefix):
    c = 0
    for file in x:
        c = c+1
        if c%100 == 0:
            print(c)
        key = prefix+file
        file_name.append(key)
        with open(join(path,file),encoding="utf8") as input_file:
            data = json.load(input_file)
            abstract = []
            if data['metadata']['abstractText'] is not None:
                abstract = nltk.sent_tokenize(data['metadata']['abstractText'].lower())
                if len(abstract) > 25:
                    abstract = abstract[:25]
            word_document_list.append(abstract)


# In[13]:


def run(sentence):
    with tf.Graph().as_default():
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddings = session.run(output, feed_dict={messages: sentence})
    return embeddings


# embed the input x and pad appropriately
def embed(x):
    lens = list(map(lambda i:len(i), x))
    x = list(it.chain.from_iterable(x))
    print('Total sentences to be embedded: {}'.format(len(x)))
    emb = run(x)
    print("embedding done")
    embedded = []
    zero = [0]*512
    ir = iter(emb)
    for i, l in enumerate(lens):
        if i % 500 == 0:
            print(i)
        z = []
        while len(z)<l:
            z.append(next(ir).tolist())
        embedded.append(z)
    print("Adding zeros")
    embedded = np.array(list(zip(*list(it.zip_longest(*embedded, fillvalue = zero)))))
   # print(embedded.shape)
    return embedded


# In[ ]:


#for  jnca get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
file_name_final  =[]
use_vector = []
for i in listdir(bib_text_path_1):
    x = listdir(bib_text_path_1+i)
    prefix = 'pos_'
    file_name = []
    word_document_list = []
    get_sentences(x,bib_text_path_1+i,word_document_list,file_name,prefix)
    y = embed(word_document_list)
    pickle.dump(file_name, open(DATA+"_train_abstract_filename_"+i+".p", "wb"))  # save it into a file named save.p
    pickle.dump(y, open(DATA+"_train_abstract_use_vector_"+i+".p", "wb"))  # save it into a file named save.p

    
    print(i) 
    
    #for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
    
    
for i in listdir(bib_text_path_2):
    x = listdir(bib_text_path_2+i)
    prefix = 'neg_'
    file_name = []
    word_document_list = []
    get_sentences(x,bib_text_path_2+i,word_document_list,file_name,prefix)
    y = embed(word_document_list)
    pickle.dump(file_name, open(DATA+"_train_abstract_filename_"+i+".p", "wb"))  # save it into a file named save.p
    pickle.dump(y, open(DATA+"_train_abstract_use_vector_"+i+".p", "wb"))  # save it into a file named save.p

    print(i)
    #for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed


# In[ ]:


bib_text_path_1 = "/home/tirthankar/Ashish/btpfinal/dataset/"+DATA+"/test/positive_bib_part/"
bib_text_path_2 = "/home/tirthankar/Ashish/btpfinal/dataset/"+DATA+"/test/negative_bib_part/"


# In[ ]:


file_name_final  =[]
use_vector = []
for i in listdir(bib_text_path_1):
    x = listdir(bib_text_path_1+i)
    prefix = 'pos_'
    file_name = []
    word_document_list = []
    get_sentences(x,bib_text_path_1+i,word_document_list,file_name,prefix)
    y = embed(word_document_list)
    
    pickle.dump(file_name, open(DATA+"_test_abstract_filename_"+i+".p", "wb"))  # save it into a file named save.p
    pickle.dump(y, open(DATA+"_test_abstract_use_vector_"+i+".p", "wb"))  # save it into a file named save.p

    print(i) 
    
    #for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed
    
    
for i in listdir(bib_text_path_2):
    x = listdir(bib_text_path_2+i)
    prefix = 'neg_'
    file_name = []
    word_document_list = []
    get_sentences(x,bib_text_path_2+i,word_document_list,file_name,prefix)
    y = embed(word_document_list)
    
    pickle.dump(file_name, open(DATA+"_test_abstract_filename_"+i+".p", "wb"))  # save it into a file named save.p
    pickle.dump(y, open(DATA+"_test_abstract_use_vector_"+i+".p", "wb"))  # save it into a file named save.p
    print(i)
    
    #for comnet get unigrams or each element is dictionary of prefix_filename and corresponding unigram words preprocessed



