import os
import re
import nltk
import math
import string
import random
import operator
import numpy as np
import pandas as pd

from time import time
from mpi4py import MPI
from collections import Counter
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from pandas.util.testing import assert_frame_equal

def init_stop_words(language):
   #initialize stop words, passing a laguange as a parameter 
   return set(stopwords.words(language))

def tokenize_words(text):
  return word_tokenize(text.lower())

def remove_stop_words(text, stop_words):
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


def get_text_dir(dir):
  #Get base directory path
  base_dir = os.path.dirname(os.path.realpath(__file__))
  #Add texts directory to base directory path 
  texts_dir = base_dir + dir
  return texts_dir

def collect_texts(texts_dir):
  texts_path = {}
  #Initialize puntuation remover
  for root, dirs, text_names in os.walk(texts_dir):
    for text_name in text_names:
      texts_path[text_name] = texts_dir + '/' + text_name
  return texts_path

def normalize_texts(text_title, text_path, stop_words):
  #Initialize list to gather text and title of text
  terms = []

# print 'Collecting...'
  #Open and read file
  text = open(text_path).read().decode('latin1')
  #Text preparation for operations
  text_prep = tokenize_words(text)
  #Remove stop words
  clean_text = remove_stop_words(text_prep, stop_words)
  #Stem text
  text_stem = stem_text(clean_text)
  #Filter Puntuation 
  text_filtered = filter_puntuation(text_stem)
  #Count word ocurrence in text
  text_counted = count_words(text_filtered)
  #Add texts and titles to dictionary
  terms = [text_title, text_counted]

  return terms

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
  tf_idf_sliced = []
  for text, terms in texts.iteritems():
    for term, frequency in terms.iteritems():
      if terms[term]:
        tf_idf.setdefault(term, 0)
        weight = calculate_weight(term, terms, df_vector, len(texts))
        tf_idf[term] = max(tf_idf[term], weight)

  #Sort list by weight descending the select first N elements 
  for term, frequency in sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)[:size]:
    tf_idf_sliced.append(term)

  return tf_idf_sliced

def fill_matrix(tf_idf, texts, term_matrix):
  for text, vector in texts.iteritems():
    print 'Vectorizing...'
    for  term in tf_idf:
      vector.setdefault(term, 0)
      term_matrix.at[term, text] = vector[term]


def cos_distance(vector1, vector2):
  magnitude_v1 = np.linalg.norm(vector1)
  magnitude_v2 = np.linalg.norm(vector2)
  dot_product = np.dot(vector1, vector2)
  cross_product = magnitude_v1 * magnitude_v2
  if cross_product != 0:
    return 1 - (float(dot_product)/cross_product)
  else:
    return 0


def find_nearest_centroid(centroids, vector):
  distances = {}
  for name, centroid in centroids.iteritems():
    distances[name] =  cos_distance(vector, centroid)

  return min(distances.iteritems(), key=operator.itemgetter(1))[0]

def calculate_mean(term_matrix, centroid, cluster): 
  vectors_to_add = pd.DataFrame(index = term_matrix.index, columns = cluster)
  for name in cluster:
    vectors_to_add[name] = term_matrix[name]
  mean = vectors_to_add.mean(axis=1)
  if mean.isnull().values.any() or np.sum(mean) == 0:
    return centroid
  else:
    return mean

def kmeans(k, max_iteration, term_matrix,):
  iterations = 0
  #Random init of centroids using the text vectors as an example
  centroids = term_matrix.sample(k, axis=1)
  centroids_sum = centroids.sum(axis=1)
  #Empty init of clusters
  clusters = defaultdict(list)
  #Emty init of old centroids for convergence
  old_centroids = pd.DataFrame(0, index = centroids.index, columns = centroids.columns)
  old_centroids_sum = old_centroids.sum(axis=1)

  while iterations < max_iteration and not np.array_equal(old_centroids_sum, centroids_sum):
    old_centroids_sum = centroids.sum(axis=1)
    #Empty init of clusters
    clusters.clear()
    for name, vector in term_matrix.iteritems():
      nearest_centroid = find_nearest_centroid(centroids, vector)
      clusters[nearest_centroid].append(name)

    for name, centroid in centroids.iteritems():
      centroids[name] = calculate_mean(term_matrix, centroids[name], clusters[name])

    print 'Running...'
    centroids_sum = centroids.sum(axis=1)
    iterations+=1

  print clusters

if __name__ == '__main__':

  comm = MPI.COMM_WORLD  
  rank = comm.rank
  size = comm.size                
  status = MPI.Status()

  start_time = 0

  tag = {'READY': 1, 'DONE': 2, 'EXIT': 3, 'START': 4} 

  if rank == 0:  
    texts = {}
    texts_norm = {}

    start_time = time()
    task_index = 0
    workers_done = 0
    workers = size - 1

    #Gather texts
    texts_dir = get_text_dir('/tests/test3')
    #Collect all texts
    texts = collect_texts(texts_dir)

    text_list = texts.items()

    while workers_done < workers:

      #Master recibe el estado actual de los workers
      message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
      task_tag = status.Get_tag()
      source = status.Get_source()

      if task_tag == tag['READY']:
        if task_index < len(text_list):
          comm.send(text_list[task_index], dest=source, tag=tag['START'])
          task_index+=1
        else:
          comm.send(None, dest=source, tag=tag['EXIT'])
      elif task_tag == tag['DONE']:
        texts_norm[message[0]] = message[1]    
      elif task_tag == tag['EXIT']:
        workers_done += 1

  else: 
        #Create list of stop words
    stop_words = init_stop_words('english')
    comm.send(None, dest=0, tag=tag['READY'])
    task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
    task_tag = status.Get_tag()
    while task_tag!=tag['EXIT']:
      if task_tag == tag['START']:
        #Clean and optimize texts for functionality
        message = normalize_texts(task[0], task[1], stop_words)
        comm.send(message, dest=0, tag=tag['DONE'])

      comm.send(None, dest=0, tag=tag['READY'])
      task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
      task_tag = status.Get_tag()

    comm.send(None, dest=0, tag=tag['EXIT'])

  if rank == 0:
    #Create vector of document freuqency for terms
    df_vector = df_vectorize(texts_norm)
    #Create tf-idf vector with determined size 
    tf_idf_size = 100
    tf_idf = tfidf_vectorize(df_vector, texts_norm, tf_idf_size)
    #Create term matrix to store vectore using pandas
    term_matrix = pd.DataFrame(index=tf_idf, columns=texts_norm)
    #Fill the matrix
    fill_matrix(tf_idf, texts_norm, term_matrix)

    kmeans(2, 30, term_matrix)
  print "PROGRAMA EJECUTO POR", time()-start_time,"SEGUNDOS"
  