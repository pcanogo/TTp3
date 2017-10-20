import os
import re
import nltk
import math
import string
import random
import operator
import numpy as np

from itertools import islice
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize



def init_stop_words(language):
   #initialize stop words, passing a laguange as a parameter 
   return set(stopwords.words(language))

def collect_texts(dir):
  #Get base directory path
  base_dir = os.path.dirname(os.path.realpath(__file__))
  #Add texts directory to base directory path 
  texts_dir = base_dir + dir
  return texts_dir

def tokenize_words(text):
  return word_tokenize(text.lower())

def remove_stop_words(text):
  return list(filter(lambda word: word not in stop_words, text))

def stem_text(text):
  return list(map(lambda x: SnowballStemmer('english').stem(x),text))

def filter_puntuation(text):
  filtered_text = []
  for term in text:
    if re.search('[a-zA-Z]', term):
        filtered_text.append(term)
  return filtered_text

def count_words(text):
  return Counter(text)

def dictionary_element_select(element, dictionary):
  return [e[element] for e in dictionary]

def clean_texts(texts_dir, stop_words):
  #Initialize list to gather text and title of text
  texts = {}
  #Initialize puntuation remover
  for root, dirs, text_names in os.walk(texts_dir):
      #Make list of all the directories of the texts
      text_path = list(map(lambda x: texts_dir + '/' + x, text_names))

      for text_index, text_title in enumerate(text_path):
        #Open and read file
        text = open(text_title).read().decode('utf-8')
        #Text preparation for operations
        text_prep = tokenize_words(text)
        #Remove stop words
        clean_text = remove_stop_words(text_prep)
        #Stem text
        text_stem = stem_text(clean_text)
        #Filter Puntuation 
        text_filtered = filter_puntuation(text_stem)
        #Count word ocurrence in text
        text_counted = count_words(text_filtered)
        #Add texts and titles to list
        # texts.append({'text': text_names[text_index],'text_vector': text_counted})
        # texts.append({text_names[text_index]: text_counted})
        texts[text_names[text_index]] = text_counted

  return texts

def df_vectorize(texts):
  df = {}
  for text, terms in texts.iteritems():
    for term, frequency in terms.iteritems():
      if terms[term]:
        df.setdefault(term, 0)
        df[term] += 1
  return df

def calculate_weight(term, terms, df_vector, number_documents):
  return terms[term] * math.log(number_documents/df_vector[term])

  
def tfidf_vectorize (df_vector, texts, size):
  tf_idf = {}
  tf_idf_sliced = {}
  for text, terms in texts.iteritems():
    for term, frequency in terms.iteritems():
      if terms[term]:
        tf_idf.setdefault(term, 0)
        weight = calculate_weight(term, terms, df_vector, len(texts))
        tf_idf[term] = max(tf_idf[term], weight)

  #Sort list by weight descending the select first N elements 
  for term, frequency in sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)[:size]:
    tf_idf_sliced[term]= tf_idf[term]

  return tf_idf_sliced



def cos_distance(vector1, vector2):
  keys_v1 = set(vector1.keys())
  keys_v2 = set(vector2.keys())
  intersection = keys_v1 & keys_v2
  magnitude_v1 = math.sqrt(sum([vector1[x]**2 for x in vector1.keys()]))
  magnitude_v2 = math.sqrt(sum([vector2[x]**2 for x in vector2.keys()]))
  dot_product = sum([vector1[x] * vector2[x] for x in intersection])
  # dot_product = np.dot(vector1, vector2)
  cross_product = magnitude_v1 * magnitude_v2

  if cross_product != 0:
    return 1 - (float(dot_product)/cross_product)
  else:
    return 0


def find_nearest_centroid(centroids, vector):
  distances = []
  for centroid_index, centroid in enumerate(centroids):
    distances.append(cos_distance(vector, centroid))

  return distances.index(min(distances))
  # print'That was the centroid'
  # print distances

# def calculate_medium(texts, cluster):
#   vectors_to_add=[]

#   for name in cluster:
#     vectors_to_add.append(texts[name])




def kmeans(k, tf_idf, texts):
  #Random init of centroids using the text vectors as an example
  centroids = random.sample(texts.values(), 2)
  # print centroids
  #Empty init of clusters
  clusters = [[] for centroid in centroids]
  #Emty init of old centroids for convergence
  old_centroids = centroids
  # while old_centroids != centroids:
  for text, vector in texts.iteritems():
    nearest_centroid = find_nearest_centroid(centroids, vector)
    clusters[nearest_centroid].append(text)

  # print clusters

  # for cluster_index, cluster in enumerate(clusters):
  #   old_centroids[index] = centroids[cluster_index]
  #   centroids[cluster_index] = calculate_medium(texts, cluster)
    





  # print clusters
  # print centroids

if __name__ == '__main__':

  #Create list of stop words
  stop_words = init_stop_words('english')
  #Gather texts
  texts_dir = collect_texts('/test3')
  #Clean and optimize texts for functionality
  texts = clean_texts(texts_dir, stop_words)
  #Create vector of document freuqency for terms
  df_vector = df_vectorize(texts)
  #Create tf-idf vector with determined size 
  tf_idf_size = 20
  tf_idf = tfidf_vectorize(df_vector, texts, tf_idf_size)

  kmeans(2, tf_idf, texts)
  