import regex as re
import numpy as np
import pythainlp
from pythainlp.corpus.common import thai_words
from typing import Collection, Callable, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from joblib import dump, load

#from configurable sources
from encoders import registered_encoders
from preprocessing_rules import rules_before_tokenization, rules_after_tokenization


## If you want to use multiprocessing techniques, I would recommend to use get_preprocessed_words() instead of preprocessing()
class Text_processing():
  def __init__(self,
               max_len: '(int) max number of tokens per sample',
               min_len: '(int) min number of tokens per sample',
               min_len_character: '(int) min number of characters per token' = 1, #recommend > 0 or 1 to automatically clear white spaces and error from the tokenizer 
              #  do_padding: '(bool) use "-PAD-" to pad the sentenses until their length is equal to max_len' = False,
              #  return_mask: '(bool) if True also return list of booleans indicating where the -PAD- is (True for real tokens and False for -PAD- token)' = False,
              #  rules_before_tokenization: '(Collection[function(str)]) Collection of functions taking sentence-level input string' = None,
              #  rules_after_tokenization: '(Collection[function(list[str])]) Collection of functions taking list of tokens' = None,
              #  stopwords: '(set[string]) set of stopwords' = {},
               engine: '(str) engine used to tokenize sentences see: https://thainlp.org/pythainlp/docs/2.0/api/tokenize.html' = 'newmm',
               verbose: '(bool) if True print some comparisons of before and after processing texts' = False
              #  additional_words: '(Collection[str]) Collection of words to "add" into the dictionary **ducplicated words will be eliminated automatically**' = {},
              #  unwanted_words: '(Collection[str]) Collection of words to "remove" into the dictionary **ducplicated words will be eliminated automatically**' = {}
               ):
    # Define rules_before_tokenization and rules_after_tokenization carefully (the order is important!!)
    self.max_len = max_len
    self.min_len = min_len
    self.min_len_character = min_len_character
    self.rules_after_tokenization = rules_after_tokenization
    self.rules_before_tokenization = rules_before_tokenization
    self.tfxidf_obj = None # to make it we need to call visualize_important_words() once
    # self.stopwords = stopwords
    self.engine = engine
    # self.do_padding = do_padding
    self.verbose = verbose
    # self.return_mask = return_mask

    #you can freely define additional words, unwanted words and stopwords using 1 word per line in each corresponding file

    additional_words = set()
    with open('./word_configs/additional_words.txt', 'r', encoding='utf8') as f:
      for line in f:
        line = line.strip()
        if line != '':
          additional_words.add(line)
    #print(f'additional_words: {additional_words}')
    unwanted_words = set()
    with open('./word_configs/unwanted_words.txt', 'r', encoding='utf8') as f:
      for line in f:
        line = line.strip()
        if line != '':
          unwanted_words.add(line)
    #print(f'unwanted_words: {unwanted_words}')
    self.stopwords = set()
    with open('./word_configs/stopwords.txt', 'r', encoding='utf8') as f:
      for line in f:
        line = line.strip()
        if line != '':
          self.stopwords.add(line)
    #print(f'self.stopwords: {self.stopwords}')
    self.dictionary = pythainlp.tokenize.dict_trie(set(thai_words()).union(set(additional_words)).difference(set(unwanted_words)))
  
  def apply_rules_before(self,
                         text: '(str) string to apply rules',
                         rules: '(Collection[tuple[pattern to be replaced, word to replace]]) Collection of function to process texts in sentence-level'
                         ) -> '(str)':
    text = text.strip()
    if text == '':
      return ''
    #######################
    ### apply the rules ###
    for func in rules:
      text = func(text)
    return text


  def apply_rules_after(self,
                        texts: '(Collection[str]) Collection of tokens to apply rules',
                        rules: '(Collection[tuple[pattern to be replaced, word to replace]]) Collection of function to process texts in token-level texts'
                        ) -> '(list[str])':
    for func in rules:
      texts = func(texts)
    return texts

  def preprocessing(self,
                    texts: '(Collection[str]): list of input strings',
                    labels: '(Collection[str] or None): list of labels',
                    label_dict: '(dict[str: int]) the key is for label names, the values is for int ranging from 0 to num unique label - 1',
                    do_padding: '(bool) use padding or not' = False,
                    return_mask: '(bool) use mask for padding tokens or not' = False
                    )-> 'if return_mask is True list[tuple(list[str], list[bool])] else list[list[str]]':
    text_out = []
    mask_out = []
    try:
      labels = list(labels)
    except:
      pass
    if labels:
      label_out = []
    ind_drop = []
    for ind, text in enumerate(texts):

      if self.verbose:
        if ind % 5 == 0:
          print(f'text before preprocessing: {text}')
      text = self.get_preprocessed_words(text)
      
      if len(text) < self.min_len:
        ind_drop.append(ind)
        continue
      elif len(text) > self.max_len:
        text = text[:self.max_len]
        mask = [True] * self.max_len
      else:
        if do_padding:
          mask = [True] * len(text) + [False] * (self.max_len - len(text))
          for _ in range(len(text), self.max_len):
            text.append('-PAD-')
      

      if self.verbose:
        if ind % 5 == 0:
          print(f'text after preprocessing: {text}')
          print('----------------------------------')
      
      text_out.append(text)
      mask_out.append(mask)
      if labels:
        label_out.append(label_dict[labels[ind]])
    if labels:
      label_out = keras.utils.to_categorical(label_out, num_classes=len(label_dict))
      if return_mask:
        return ((text_out, mask_out), label_out, ind_drop)
      else:
        return (text_out, label_out, ind_drop)
    else:
      if return_mask:
        return (text_out, mask_out, ind_drop)
      else:
        return text_out, ind_drop
  
  def preprocessing2(
    self,
    texts: '(Collection[str]): list of input sentence-level string ==> already tokenized and use "|" as the delimiter, use "/" as the separator separating between a word and its label. Ex. ฉัน/Noun|กิน/Verb|...',
    do_padding: '(bool) use padding or not. In case of True, masks are also returned (False for padding token. Otherwise is True)' = False
    ):
    tokens = []
    labels = []
    masks = []
    all_labels = {}
    count_ulabels = {}
    for i in range(len(texts)):
      texts[i] = texts[i].strip()
      if texts[i] != '':
        tokens.append([])
        labels.append([])
        for j in texts[i].split('|'):
          tmp = j.strip().split('/')
          if len(tmp[0]) < self.min_len_character:
            continue
          else:
            tokens[-1].append(tmp[0])
            labels[-1].append(tmp[1])
            if tmp[1] not in all_labels:
              all_labels[tmp[1]] = len(all_labels)
              count_ulabels[tmp[1]] = 1
            else:
              count_ulabels[tmp[1]] += 1
    
    for i in range(len(tokens)):
      if len(tokens[i]) < self.min_len:
        continue
      elif len(tokens[i]) >= self.max_len:
        tokens[i] = tokens[i][:self.max_len]
        labels[i] = labels[i][:self.max_len]
        masks.append([True]*self.max_len)
      else:
        if do_padding:
          masks.append([True]*(len(tokens[i])) + [False]*(self.max_len - len(tokens[i])))
          tokens[i] = tokens[i] + ['-PAD-']*(self.max_len - len(tokens[i]))
          labels[i] = labels[i][:self.max_len] + ['-PAD-']*(self.max_len - len(labels[i]))
        else:
          continue
    if do_padding:
      return ((tokens, masks), labels), all_labels, count_ulabels
    else:
      return (tokens, labels), all_labels, count_ulabels



  def tokenize(self,
               text: '(str) sentence-level string to be tokenized'
               ) -> '(list[str]) only tokenize doesnt apply any preprocessing step':
    
    #################################
    ### request to elastic search ###
    words = []
    for word in pythainlp.tokenize.word_tokenize(text, custom_dict=self.dictionary, engine=self.engine):
      word = word.strip()
      if len(word) < self.min_len_character or word in self.stopwords:
        continue
      words.append(word)
    ################################
    return words

  def get_preprocessed_words(self,
                             text: '(str) raw sentence-level string'
                             ) -> '(list[str]) convert plain text to a list of tokens':
    text = self.apply_rules_before(text, self.rules_before_tokenization)
    text = self.tokenize(text)
    return self.apply_rules_after(text, self.rules_after_tokenization)

  def visualize_important_words(self,
                                texts: '(Collection[str]) Collection of raw sentence-level texts',
                                labels: '(Collection[str]) Collection of labels ==> dont need to turn into enumerated or one-hot vectors',
                                n_gram_range: '(tuple[int, int]) tuple of (min_n_gram_size, max_n_gram_size)' = (1,2),
                                min_df: '(int or float) we consider only the word that existes for min_df doccuments and percent from 0-1 for float' = 1,
                                tfxidf_path: '(str) path to save the tf-idf model'= 'tfxidf_encoder'
                                ) -> '(list[pd.DataFrame]) ==> length equals to number of unique labels':
    ## this method uses score ranking from tf-idf feature to see usefulness of each words ##
    
    vectorizer = TfidfVectorizer(tokenizer=self.tokenize, 
                                 ngram_range= n_gram_range,
                                 sublinear_tf=True,
                                 min_df=min_df)
    X = vectorizer.fit_transform(texts)
    dump(vectorizer, os.path.join('trained_models\\', tfxidf_path + '.joblib'))
    #visualize texts
    #from visualize import top_feats_all, plot_top_feats
    #code from pythainlp
    features = vectorizer.get_feature_names()
    ts = self.top_feats_all(X.toarray(), labels, features)
    self.tfxidf_obj = vectorizer
    return ts

  def tfxidf_encode(
    self,
    texts: '(list[str]) list of sentence-level strings'
    ):
    return self.tfxidf_obj.transform(texts)

  def top_feats_label(self,
                      X: np.ndarray, features: Collection[str], label_idx: Collection[bool] = None,
                      min_val: float = 0.1, agg_func: Callable = np.mean)->pd.DataFrame:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    rank features of each label by their encoded values (CountVectorizer, TfidfVectorizer, etc.)
    aggregated with `agg_func`
    :param X np.ndarray: document-value matrix
    :param features Collection[str]: feature names
    :param label_idx Collection[int]: position of rows with specified label
    :param min_val float: minimum value to take into account for each feature
    :param agg_func Callable: how to aggregate features such as `np.mean` or `np.sum`
    :return: a dataframe with `feature`, `score` and `ngram`
    '''
    res = X[label_idx] if label_idx is not None else X
    res[res < min_val] = 0
    res_agg = agg_func(res, axis=0)
    df = pd.DataFrame([(features[i], res_agg[i]) for i in np.argsort(res_agg)[::-1]])
    df.columns = ['feature','score']
    df['ngram'] = df.feature.map(lambda x: len(set(x.split(' '))))
    return df

  def top_feats_all(self,
                    X: np.ndarray, y: np.ndarray, features: Collection[str], min_val: float = 0.1, 
                    agg_func: Callable = np.mean)->Collection[pd.DataFrame]:
    '''
    original code (Thomas Buhrman)[from https://buhrmann.github.io/tfidf-analysis.html]
    for all labels, rank features of each label by their encoded values (CountVectorizer, TfidfVectorizer, etc.)
    aggregated with `agg_func`
    :param X np.ndarray: document-value matrix
    :param y np.ndarray: labels
    :param features Collection[str]: feature names
    :param min_val float: minimum value to take into account for each feature
    :param agg_func Callable: how to aggregate features such as `np.mean` or `np.sum`
    :return: a list of dataframes with `rank` (rank within label), `feature`, `score`, `ngram` and `label`
    '''
    labels = np.unique(y)
    dfs = []
    for l in labels:
        label_idx = (y==l)
        df = self.top_feats_label(X,features,label_idx,min_val,agg_func).reset_index()
        df['label'] = l
        df.columns = ['rank','feature','score','ngram','label']
        dfs.append(df)
    return dfs

  def get_dictionary(self):
    return set(self.dictionary)


class Word_Embedder():
  def __init__(self,
               engine: '(str) embedder name ==> For now, only Thai fasttext is supported',
               max_len: '(int) number of tokens',
               is_sequence_prediction: '(bool) True for sequence prediction (Ex: NER), False for sentence prediction (Ex: Sentiment Analysis)' = False,
               unique_labels: '(dict[str: int])' = None
               ):
    self.model = registered_encoders[engine]['load_model']()
    self.encode_function = registered_encoders[engine]['encode']
    self.encode_size = registered_encoders[engine]['encode_size'] # 300 for Thai fasttext
    self.max_len = max_len
    self.is_sequence_prediction = is_sequence_prediction
    self.unique_labels = unique_labels
  def cre_generator(self,
                    texts: '(Collection[Collection[str]]) Collection of collections of tokenized words==> with padding!',
                    masks: '(Collection[Collection[bool]]) Collection of collections of booleans indicating -PAD- ==> True for non-padding token',
                    labels: '(Collection[Collection[int]]) Collection of collections of labels ==> int ranging from 0 to num_unique_label - 1'
                    ):
    if self.is_sequence_prediction:
      for i in range(len(texts)):
        encoded_texts = []
        for j in range(self.max_len):
          encoded_texts.append(self.encode_function(self.model, texts[i][j]))
        # print('labels[i]')
        # print(labels[i])

        yield ({'word_vectors': np.stack(encoded_texts), 'masks': masks[i]}, keras.utils.to_categorical([self.unique_labels[k] for k in labels[i]], num_classes=len(self.unique_labels)))
    else:
      for i in range(len(texts)):
        encoded_texts = []
        for j in range(self.max_len):
          encoded_texts.append(self.encode_function(self.model, texts[i][j]))
        yield ({'word_vectors': np.stack(encoded_texts), 'masks': masks[i]}, labels[i])

  def cre_tf_dataset(self,
                     is_testset: '(bool)',
                     batch_size: '(int)',
                     texts: '(Collection[Collection[str]]) Collection of collections of tokenized words==> with padding!',
                     masks: '(Collection[Collection[bool]]) Collection of collections of booleans indicating -PAD- ==> True for non-padding token',
                     labels: '(Collection[Collection[int]]) Collection of collections of labels ==> int ranging from 0 to num_unique_label - 1'
                     ):
    if self.is_sequence_prediction:
      data = tf.data.Dataset.from_generator(lambda: self.cre_generator(texts, masks, labels), output_types= ({'word_vectors':tf.float32, 'masks':tf.bool}, tf.float32), 
                                            output_shapes=({'word_vectors':tf.TensorShape([self.max_len,self.encode_size]), 'masks': tf.TensorShape([self.max_len])}, tf.TensorShape([self.max_len,len(self.unique_labels)])))
    else:
      data = tf.data.Dataset.from_generator(lambda: self.cre_generator(texts, masks, labels), output_types= ({'word_vectors':tf.float32, 'masks':tf.bool}, tf.float32), 
                                            output_shapes=({'word_vectors':tf.TensorShape([self.max_len,self.encode_size]), 'masks': tf.TensorShape([self.max_len])}, tf.TensorShape([len(self.unique_labels)])))
    
    AUTO = tf.data.experimental.AUTOTUNE
    if is_testset:
      data = data.prefetch(AUTO).batch(batch_size)
    else:
      data = data.shuffle(batch_size*5).prefetch(AUTO).batch(batch_size)
    return data
  
  def encode(
    self,
    words: '(Collection[str]) collection of tokens ==> in case of prediction, padding is required'):
    out = []
    for word in words:
      out.append(self.encode_function(self.model, word))
    return np.stack(out)

  def get_embedding_size(self):
    return self.encode_size
    
