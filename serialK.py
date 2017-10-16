import os
import nltk
import math
import string
import random
import numpy as np

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize



def init_stop_words(language):
   #initialize stop words, passing a laguange as a parameter 
   return set(stopwords.words(language))

def collect_texts():
  #Get base directory path
  base_dir = os.path.dirname(os.path.realpath(__file__))
  #Add texts directory to base directory path 
  texts_dir = base_dir + '/test2'
  return texts_dir

def tokenize_words(text):
  return word_tokenize(text.lower())

def remove_stop_words(text):
  return list(filter(lambda word: word not in stop_words, text))

def stem_text(text):
  return list(map(lambda x: SnowballStemmer('english').stem(x),text))

def count_words(text):
  return Counter(text)

def dictionary_element_select(element, dictionary):
  return [e[element] for e in dictionary]

def clean_texts(texts_dir, stop_words):
  #Initialize list to gather text and title of text
  texts = []
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
        clean_text = remove_stop_words(text_prep )
        #Stem text
        text_stem = stem_text(clean_text)
        #Count word ocurrence in text
        text_counted = count_words(text_stem)
        #Add texts and titles to list
        texts.append({'text': text_names[text_index],'text_vector': text_counted})

  return texts

def cos_distance(vector1, vector2):
  keys_v1 = set(vector1.keys())
  keys_v2 = set(vector2.keys())
  intersection = keys_v1 & keys_v2
  magnitude_v1 = math.sqrt(sum([vector1[x]**2 for x in vector1.keys()]))
  magnitude_v2 = math.sqrt(sum([vector2[x]**2 for x in vector2.keys()]))
  dot_product = sum([vector1[x] * vector2[x] for x in intersection])
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

def calculate_medium(texts_names, vectors, cluster):
  vectors_to_add=[]
  for index, name in enumerate(cluster):
    vectors_to_add.append(vectors[texts_names.index(name)])


def kmeans(k, texts):
  # print texts['File5.txt']['text_vector']
  #Extract the vectors genereated from the texts
  vectors = dictionary_element_select('text_vector', texts)
  text_names = dictionary_element_select('text', texts)
  #Random init of centroids using the text vectors as an example
  centroids = random.sample(vectors, 2)
  #Empty init of clusters
  clusters = [[] for centroid in centroids]
  #Emty init of old centroids for convergence
  old_centroids = clusters
  # while old_centroids != centroids:
  for index, vector in enumerate(vectors):
    nearest_centroid = find_nearest_centroid(centroids, vector)
    clusters[nearest_centroid].append(text_names[index])

  for cluster_index, cluster in enumerate(clusters):
    calculate_medium(text_names, vectors, cluster)
    # old_centroids[index] = centroid[index]





  print clusters
  # print centroids

if __name__ == '__main__':
  stop_words = init_stop_words('english')
  texts_dir = collect_texts()
  texts = clean_texts(texts_dir, stop_words)
  kmeans(2, texts)
  