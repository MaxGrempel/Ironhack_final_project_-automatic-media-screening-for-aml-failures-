#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ### Automation of Merchant Media Screening for AML purposes

# In[173]:


# importing relevant libraries

import pandas as pd
import numpy as np
import requests
from serpapi import GoogleSearch
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
import webbrowser

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')


# etc


# In[174]:


# loading pickles (vectorizer and model)

with open('vect.pkl', 'rb') as file:
    vect = pickle.load(file)
    
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)


# In[175]:


# settings for script modification
# pd.set_option('display.max_colwidth', None)


# In[176]:


# SerpApi api key
api_key = 'cf07369d96714784e62c191673c17918e8d7274bb4037d525ee1ed50d2b795b1'


# In[177]:


# defining the query
print("Please, enter the legal entity name of the company you want to screen.")
screening_object = input()
query = '"' + screening_object + '"'+ ' ' + 'AND (convicted OR charged OR money laundering OR tax evasion OR embezzle OR investigation OR trialed OR sentenced OR corruption OR fraud OR fine OR penalty OR terrorist)'


# In[178]:


print('The search query is: ' + query)


# In[ ]:





# In[179]:


# define parameters for the api request
params = {
  "engine": "google",
  "q": query, 
    'num': '20',
  "api_key": api_key
}


# In[180]:


# api request
search = GoogleSearch(params)
results = search.get_dict()
# news_results = results["news_results"]
organic_results = results["organic_results"]


# In[184]:


# transforming results to dataframe
results = pd.DataFrame(organic_results)

# minor cleaning and dropping irrelevant columns
# results.drop(['position', 'thumbnail'], axis=1, inplace=True)
results = results[['title', 'link', 'snippet']]
results['snippet'] = results['snippet'].str.replace('\n', '')
results['snippet'] = results['snippet'].str.lower()
results['title'] = results['title'].str.lower()


# In[185]:


# wordnet POS function

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, 
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# In[186]:


# tokenizing and removing punctuation function

# title
def tokenizer_depunctuation_title(row):
  tokens = word_tokenize(row['title'])
  return [word.lower() for word in tokens if word.isalpha()]

results['title_tokenized'] = results.apply(tokenizer_depunctuation_title,axis=1)

# snippet
def tokenizer_depunctuation_snippet(row):
  tokens = word_tokenize(row['snippet'])
  return [word.lower() for word in tokens if word.isalpha()]

results['snippet_tokenized'] = results.apply(tokenizer_depunctuation_snippet,axis=1)


# In[187]:


# lemmatizing function

lemmatizer = WordNetLemmatizer() 

# title
def lemmatizer_with_pos_title(row):
  return [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in row['title_tokenized']]

results['title_lemmatized'] = results.apply(lemmatizer_with_pos_title,axis=1)

# snippet
def lemmatizer_with_pos_snippet(row):
  return [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in row['snippet_tokenized']]

results['snippet_lemmatized'] = results.apply(lemmatizer_with_pos_snippet,axis=1)


# In[188]:


# removing stopwords

#title
def remove_sw_title(row):
  return list(set(row['title_lemmatized']).difference(stopwords.words()))

results['title_no_stopwords'] = results.apply(remove_sw_title,axis=1)

# snippet
def remove_sw_snippet(row):
  return list(set(row['snippet_lemmatized']).difference(stopwords.words()))

results['snippet_no_stopwords'] = results.apply(remove_sw_snippet,axis=1)


# In[189]:


# restructuring function

# title
def re_structure_title(row):
  return " ".join(row['title_no_stopwords'])

results['title_clean'] = results.apply(re_structure_title,axis=1)

# snippet
def re_structure_snippet(row):
  return " ".join(row['snippet_no_stopwords'])

results['snippet_clean'] = results.apply(re_structure_snippet,axis=1)


# In[190]:


# defining variable for prediction

X_title = vect.transform(results['title_clean']).toarray()
 
X_snippet = vect.transform(results['snippet_clean']).toarray()


# In[191]:


# making classification
results['label_title'] = svm_model.predict(X_title)
results['label_snippet'] = svm_model.predict(X_snippet)


# In[192]:


# creating final results table
final_results = results[['title', 'label_title', 'snippet', 'label_snippet', 'link']]


# In[193]:


# excluding irrelevant results
final_results = final_results[(final_results['label_title'] == 'relevant') | (final_results['label_snippet'] == 'relevant')]


# In[195]:


results_count = len(final_results)


# In[196]:


# saving file and/or concluding the script
if results_count > 0:
    
    final_results.to_csv('media_' + screening_object + '.csv', index=False)
    
    print("A csv file containing " + str(results_count) + " potentially relevant news has been saved on your machine! Please manually review them for relevance!")
    
    print("Would you like to open the URL(s) of the results? [y/n]")
    
    while True:
    
        open_url = input()
    
        if open_url.lower() in ['y', 'yes']:
        
            urls = list(final_results['link'])
            
            print('The links will be opened automatically.')

            for url in urls:
                webbrowser.open(url)
            break
            
        elif open_url.lower() in ['n', 'no']:
        
            print("The links will not be opened automatically.")
            
            break
        
        else:
    
            print("Invalid answer. Try 'yes' or 'no'.")
    
else:
    
     print("No relevant media was found. To be sure, please check manually!")   


# In[ ]:




