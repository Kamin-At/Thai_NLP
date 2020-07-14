from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
import json
import pandas as pd
import regex as re


from preprocessing import Word_Embedder
from preprocessing import Text_processing

class count_based_model():
  def __init__(
    self,
    filename_dict: '(dict[str: collection[str]]) example: {"Negative class": ["neg.txt", "neg_manual.txt"], "Positive class": ["pos.txt"]}',
    is_sentiment_analysis: '(bool) if True ==> the class should contains "pos" ==> positive and "neg" ==> negative',
    do_load_models: '(bool) True if use pretrained models',
    model_path: '(str) model path'):
    self.class_dictionary = {}
    self.is_sentiment_analysis = is_sentiment_analysis
    self.model_path = model_path
    self.all_cols = []
    if do_load_models:
      self.model, self.scaler = self.load_model()
      self.mapping = {ind: tmp_class for ind, tmp_class in enumerate(self.model.classes_)}
      with open(os.path.join(self.model_path,'count_based_model_config.txt'), 'r', encoding='utf8') as f:
        for line in f:
          line = line.strip()
          if line != '':
            self.all_cols.append(line)
    else:
      self.model = None
      self.scaler = None
      self.mapping = {}
    for tmp_class in filename_dict:
      self.class_dictionary[tmp_class] = set()
      for tmp_file in filename_dict[tmp_class]:
        with open(tmp_file, 'r', encoding='utf8') as f:
          for word in f:
            word = word.strip()
            if word != '':
              word = ''.join(word.split(' '))
              if len(word) > 1:
                self.class_dictionary[tmp_class].add(word)
    self.class_finding = {}
    for tmp_class in self.class_dictionary:
      self.class_finding[tmp_class] = '('+'|'.join(self.class_dictionary[tmp_class]) + ')'
    
  def load_model(self):
      return load(os.path.join(self.model_path, 'SVM_count_based_model.joblib')), load(os.path.join(self.model_path, 'z_normalizer.joblib'))
  def encode(
    self,
    string_in: '(str) sentence-level string'):
    
    tmp_dict = {}
    for tmp_class in self.class_finding:
      tmp_dict[tmp_class] = [len(re.findall(self.class_finding[tmp_class], string_in))]
      # print(f'tmp_class: {tmp_class}, tmp_dict[tmp_class]: {tmp_dict[tmp_class]}')
      tmp_dict['log_' + tmp_class] = [np.log(1 + tmp_dict[tmp_class][0])]
    if self.is_sentiment_analysis:
      tmp_dict['pos-neg'] = [tmp_dict['pos'][0] - tmp_dict['neg'][0]]
    # print(tmp_dict)
    # print(self.all_cols)
    tmp_dict = pd.DataFrame(tmp_dict)
    # print(tmp_dict[self.all_cols].values)
    tmp = self.scaler.transform(tmp_dict[self.all_cols].values)
    # print(tmp)
    # print(tmp.shape)
    return tmp

  def fit(
    self,
    traing_df: '(pd.DataFrame) contains "texts" ==> sentence_level_data, "labels" ==> sentence_level_label columns',
    test_df: '(pd.DataFrame) contains "texts" ==> sentence_level_data, "labels" ==> sentence_level_label columns'):
    self.all_cols = []
    for tmp_class in self.class_finding:
      self.all_cols.extend([tmp_class, 'log_'+tmp_class])
      traing_df[tmp_class] = traing_df['texts'].apply(lambda z: len(re.findall(self.class_finding[tmp_class], z)))
      traing_df['log_' + tmp_class]= np.log(1 + traing_df[tmp_class])
      test_df[tmp_class] = test_df['texts'].apply(lambda z: len(re.findall(self.class_finding[tmp_class], z)))
      test_df['log_' + tmp_class]= np.log(1 + test_df[tmp_class])
      
    if self.is_sentiment_analysis:
      traing_df['pos-neg'] = traing_df['pos'] - traing_df['neg']
      test_df['pos-neg'] = test_df['pos'] - test_df['neg']
      self.all_cols.append('pos-neg')
    print('running grid search and 5-folds cross validation')
    parameters = {'C':[1]}#[0.001, 0.01, 0.1, 1, 5, 10, 100]}
    self.scaler = StandardScaler()
    self.scaler.fit(traing_df[self.all_cols].values)
    dump(self.scaler, os.path.join(self.model_path, 'z_normalizer.joblib'))
    tr = self.scaler.transform(traing_df[self.all_cols].values)
    te = self.scaler.transform(test_df[self.all_cols].values)
    model = SVC(kernel='rbf', class_weight='balanced')
    clf = GridSearchCV(model, parameters, scoring= 'f1_macro', return_train_score= True, n_jobs=-1)
    out = clf.fit(tr, traing_df['labels'])
    best_c = out.best_params_['C']
    self.model = SVC( C=best_c, kernel='rbf', class_weight='balanced')
    out = self.model.fit(tr, traing_df['labels'])
    y_pred = out.predict(te)
    out2 = f1_score(test_df['labels'],y_pred,average=None)
    print(f'best_C: {best_c}, f1-score: {out2}, f1-macro: {np.mean(out2)}')
    dump(self.model, os.path.join(self.model_path, 'SVM_count_based_model.joblib'))

    with open(os.path.join(self.model_path,'count_based_model_config.txt'), 'w', encoding='utf-8') as file:
      for i in self.all_cols:
        file.write(i + '\n')

  def predict(
    self,
    string_in: '(str) sentence-level string'):
    tmp = self.encode(string_in)
    out = self.model.predict(tmp)
    return out[0]

    
def prepare_data_for_text_classification(
    train_dataframe:'(pd.DataFrame) dataframe containing 2 fields ==> "texts" (sentence-level text) and "labels" (string)',
    test_dataframe:'(pd.DataFrame) dataframe containing 2 fields ==> "texts" (sentence-level text) and "labels" (string)',
    max_len: '(int) max length of all sentences (shorter sentences will be padded, longer sentences will be truncated)',
    min_len: '(int) min number of tokens per sample ==> if less than min_len, we drop that sample',
    n_gram_range: '(tuple[int,int]) (min_n_gram, max_n_gram)' = (1,3),
    min_df: '(int) consider only tokens which exist greater of equal to min_df documents'=5,
    word_embedder: '(str) engine to be used for word embedding' = 'fasttext',
    tfxidf_path: '(str) path to save the tf-idf model'= 'tf-idf_encoder',
    engine: '(str) engine used for word tokenization' = 'newmm',
    threshold_tfxidf: '(float) consider words which correspond to each class' = 0.01,
    verbose=False
    ):
    u_label = {}
    for ind, i in enumerate(train_dataframe['labels'].value_counts().index):
        u_label[i] = ind
    print('initializing')
    tp = Text_processing(max_len, min_len,engine=engine,verbose=verbose)
    print('tf-idf train begins')
    train_dataframe['texts_bef'] = train_dataframe['texts'].apply(tp.apply_rules_before)
    test_dataframe['texts_bef'] = test_dataframe['texts'].apply(tp.apply_rules_before)
    d = tp.visualize_important_words(train_dataframe['texts_bef'],train_dataframe['labels'],n_gram_range,min_df, tfxidf_path= tfxidf_path)
#     d = tp.visualize_important_words(,,n_gram_range,min_df, tfxidf_path= tfxidf_path)
    print('finished tf-idf train')
    cur_path = os.getcwd()
    os.chdir('word_configs')
    print('begin text preprocessing')
    out = tp.preprocessing(train_dataframe['texts'], train_dataframe['labels'], u_label, True, True)
    print('finished text preprocessing for train')
    out2 = tp.preprocessing(test_dataframe['texts'], test_dataframe['labels'], u_label, True, True)
    print('finished text preprocessing for test')
    for i in d:
      tmp_class = i.iloc[0]['label']
      i = i[i['score']> threshold_tfxidf]
      with open(tmp_class + '.txt', 'w', encoding='utf8') as f:
        for j in i['feature']:
          f.write(j+'\n')
    os.chdir(cur_path)
    
    train_tf = tp.tfxidf_encode(train_dataframe['texts_bef'])
    test_tf = tp.tfxidf_encode(test_dataframe['texts_bef'])
    WE = Word_Embedder(word_embedder, max_len, unique_labels= u_label)
    print(train_dataframe['labels'].value_counts())
    
    #print(u_label)
    
    # print(out[1][:100])
    tmp_label_train = np.argmax(out[1],axis=1)

    CW = compute_class_weight('balanced', np.unique(tmp_label_train), tmp_label_train)
    CW2 = {}
    for i in range(CW.shape[0]):
      CW2[i] = CW[i]
    print(f'CW2: {CW2}')
    x = WE.cre_tf_dataset(is_testset = False, batch_size=64, texts=out[0][0], masks=out[0][1], labels=out[1])
    x2 = WE.cre_tf_dataset(is_testset = True, batch_size=64, texts=out2[0][0], masks=out2[0][1], labels=out2[1])
    x3 = WE.cre_tf_dataset(is_testset = True, batch_size=64, texts=out[0][0], masks=out[0][1], labels=out[1])
    train_tf = np.delete(train_tf.toarray(), out[2], axis=0)
    test_tf = np.delete(test_tf.toarray(), out2[2], axis=0)
    # print(f'num train sample linear: {train_tf.shape[0]}, num test sample linear: {test_tf.shape[0]}')
    # print(f'num train label linear: {len(train_dataframe["labels"].drop(out[2],axis=0))}, num test label linear: {len(test_dataframe["labels"].drop(out[2],axis=0))}')
    # print(f'num train sample deeplearning: {len(out[0][0])}, num test sample deeplearning: {len(out2[0][0])}')
    # print(f'num train sample deeplearning: {len(out[1])}, num test sample deeplearning: {len(out2[0])}')
    return {
        'unique_label': u_label, 'max_len': max_len, 'embedding_size': WE.get_embedding_size(), 'tfxidf_train': train_tf, 
        'tfxidf_label_train': train_dataframe['labels'].drop(out[2],axis=0), 'tfxidf_test': test_tf, 'tfxidf_label_test': test_dataframe['labels'].drop(out2[2],axis=0),
        'tf_train_dataset': x, 'tf_test_dataset': x2, 'tf_train_dataset2': x3, 'class_weight': CW2, 'min_len': min_len, 'engine': engine, 'word_embedder':word_embedder, 'preprocessed text': out[0][0]
        }
    #tc =Text_classification(u_label,max_len,WE.get_embedding_size(),True,True,False,train_tf,train_dataframe['labels'],test_tf,test_dataframe['labels'],x,x2)

def prepare_data_for_sequence_prediction(
    train_collection:'(tuple[Collection[list(str)], Collection[list(int)], Collection[int(int)], ....]) tuple of collections containing list of tokens, labels for outhers (multi-task is ok)',
    test_collection:'(tuple[Collection[list(str)], Collection[list(int)], Collection[int(int)], ....]) tuple of collections containing list of tokens, labels for outhers (multi-task is ok)',
    max_len: '(int) max length of all sentences (shorter sentences will be padded, longer sentences will be truncated)',
    min_len: '(int) min number of tokens per sample ==> if less than min_len, we drop that sample',
    unique_labels: '(dict[str: int])',
    # cnt_unique_labels: '(dict[str: int])',
    word_embedder: '(str) engine to be used for word embedding' = 'fasttext'
    ):
  WE = Word_Embedder(engine= word_embedder, max_len=max_len, is_sequence_prediction=True, unique_labels=unique_labels)
  out1 = WE.cre_tf_dataset(is_testset=False,batch_size=64,texts=train_collection[0][0][0] ,masks=train_collection[0][0][1],labels=train_collection[0][1])
  out2 = WE.cre_tf_dataset(is_testset=True,batch_size=64,texts=test_collection[0][0][0] ,masks=test_collection[0][0][1],labels=test_collection[0][1])
  out3 = WE.cre_tf_dataset(is_testset=True,batch_size=64,texts=train_collection[0][0][0] ,masks=train_collection[0][0][1],labels=train_collection[0][1])
  # max_num = []
  # for i in cnt_unique_labels:
  #   max_num.append(cnt_unique_labels[i])
  # max_num = max(max_num)
  # class_weights = {unique_labels[class_name] : max_num/tmp_cnt for class_name, tmp_cnt in cnt_unique_labels.items()}

  #################################################
  ############# Not Ready Yet #####################
  return {
    'tf_train_dataset': out1, 'tf_test_dataset':out2, 'tf_train_dataset2':out3, 'max_len': max_len, 'embedding_size': WE.get_embedding_size(), 
    'min_len': min_len, 'word_embedder':word_embedder, 'unique_label': unique_labels}

class Text_classification():
  def __init__(self,
            #    class_dict: '(dict[str: int]) the key is for the class names, the value is for index of the classes ==> int ranging from 0 to num_unique_class -1',
            #    max_len: '(int) max number of tokens per sample',
            #    num_feature_per_word: '(int) encoding size (token-level)',
               prepared_data_dict: '(dict) output of prepare_data_for_text_classification()',
               do_deep_learning: '(bool) if True ==> use bi-GRU to predict'=True,
               do_linear_classifier: '(bool) if True ==> use linear classifier with TF-IDF features'=True,
               is_sequence_prediciton: '(bool) if True ==> Ex. NER, else ==> Ex. Sentiment Analysis'=False,
               model_path: '(str) folder name which will be created in "trained_models" folder' = 'text_classification'
               ):
            #    tfxidf_train: '(np.array) (for do_linear_classifier) from TfidfVectorizer.fit or .fit_transform'=None,
            #    label_train: '(Collection[str]) (for do_linear_classifier) Ex. ["angry", "happy", "angry",....]'=None,
            #    tfxidf_test: '(np.array) (for do_linear_classifier) from TfidfVectorizer.fit or .fit_transform'=None,
            #    label_test: '(Collection[str]) (for do_linear_classifier) Ex. ["angry", "happy", "angry",....]'=None,
            #    tf_train_dataset: '(tf.dataset) from Word_Embedder class'= None,
            #    tf_test_dataset: '(tf.dataset) from Word_Embedder class'= None):

    self.max_len = prepared_data_dict['max_len']
    self.min_len = prepared_data_dict['min_len']
    self.word_embedder = prepared_data_dict['word_embedder']
    self.num_feature_per_word = prepared_data_dict['embedding_size']
    self.class_dict = prepared_data_dict['unique_label']
    self.is_sequence_prediciton = is_sequence_prediciton
    self.tf_train_dataset = prepared_data_dict['tf_train_dataset']
    self.tf_test_dataset = prepared_data_dict['tf_test_dataset']
    self.tf_train_dataset2 = prepared_data_dict['tf_train_dataset2']
    self.dl_model = None
    self.model_path = os.path.join('trained_models',model_path)
    self.do_deep_learning = do_deep_learning
    
    if self.is_sequence_prediciton:
      self.config = {
        'max_len': self.max_len, 
        'embedding_size': self.num_feature_per_word,
        'class_dict': self.class_dict,
        'is_sequence_prediction': self.is_sequence_prediciton,
        'min_len': self.min_len,
        'word_embedder': self.word_embedder
        }
    else:
      self.class_weight = prepared_data_dict['class_weight']
      self.engine = prepared_data_dict['engine']
      self.do_deep_learning = do_deep_learning
      self.do_linear_classifier = do_linear_classifier
      self.tfxidf_train = prepared_data_dict['tfxidf_train']
      self.label_train = prepared_data_dict['tfxidf_label_train']
      self.tfxidf_test = prepared_data_dict['tfxidf_test']
      self.label_test = prepared_data_dict['tfxidf_label_test']
      self.linear_classifier = None
      self.SVM_classifier = None
      self.ensemble_model = None
      self.config = {
        'max_len': self.max_len, 
        'embedding_size': self.num_feature_per_word,
        'class_dict': self.class_dict,
        'is_sequence_prediction': self.is_sequence_prediciton,
        'min_len': self.min_len,
        'word_embedder': self.word_embedder,
        'engine': self.engine
        }
      
    
    os.makedirs(self.model_path, exist_ok =True) 
    print(f'Created the path: {self.model_path}')
    with open(os.path.join(self.model_path,'model_config.json'), 'w', encoding='utf-8') as file:
      json.dump(self.config, file)
    # self.dl_model = keras.models.load_model( os.path.join(self.model_path, 'deeplearning_model.h5'))
    # self.linear_classifier = load(os.path.join(self.model_path, 'logistic_regression_model.joblib'))
    
  def cre_deep_learning_model(
      self,
      dropout_dense=0.4,
      dropout_gru=0.4,
      gru_size=128,
      num_gru_layer=3,
      num_dense_layer=3,
      WR = 5e-3):
    L2 = keras.regularizers.l2(WR)
    inputs = keras.Input(shape=(self.max_len,self.num_feature_per_word), name='word_vectors')
    masks = keras.Input(shape=(self.max_len),dtype=np.dtype(bool), name='masks')
    
    x = keras.layers.Bidirectional(keras.layers.GRU(gru_size,dropout=dropout_gru,recurrent_dropout=0,return_sequences=True, kernel_regularizer= L2, bias_regularizer= L2))(inputs, mask=masks)
    for i in range(num_gru_layer - 1):
      x = keras.layers.Bidirectional(keras.layers.GRU(gru_size,dropout=dropout_gru,recurrent_dropout=0,return_sequences=True, kernel_regularizer= L2, bias_regularizer= L2))(x, mask=masks)
    #x = keras.layers.Dropout(0.2, input_shape=(None,1,128))(x)
    if dropout_dense > 0.:
      x = keras.layers.Dropout(dropout_dense)(x)
    if self.is_sequence_prediciton:
      for i in range(num_dense_layer):
        x = keras.layers.TimeDistributed(keras.layers.Dense(gru_size//(2*(i+1)), kernel_regularizer= L2, bias_regularizer= L2))(x, mask=masks)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(keras.layers.LeakyReLU())(x, mask=masks)
        if dropout_dense > 0.:
          x = keras.layers.Dropout(dropout_dense)(x)
      SEQ_out = keras.layers.TimeDistributed(keras.layers.Dense(len(self.class_dict), activation= 'softmax', kernel_regularizer= L2, bias_regularizer= L2),name='seq_out')(x, mask=masks)
      return keras.Model(inputs= [inputs, masks], outputs = SEQ_out)
    else:
      x = keras.layers.Flatten()(x)
      for i in range(num_dense_layer):
        x = keras.layers.Dense(128//(2**(i)), kernel_regularizer= L2, bias_regularizer= L2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
      SEQ_out = keras.layers.Dense(len(self.class_dict), activation= 'softmax', name='seq_out', kernel_regularizer= L2, bias_regularizer= L2)(x)
      return keras.Model(inputs= [inputs, masks], outputs = SEQ_out)
#   def train_model(self, 
#                   config: "model's hyper parameters ==> must contain 'cre_model_func' key"):
#     model = config['cre_model_func'](config)
#     MC = keras.callbacks.ModelCheckpoint( 'deeplearning_model.h5', monitor="val_loss",verbose=0,save_best_only=True)
#     RP = keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.1,min_lr=1e-6, patience=10)
#     ES = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=20,restore_best_weights=True)
#     hist = model.fit(self.tf_train_dataset, 
#           validation_data = self.tf_test_dataset, 
#           epochs=500, 
#           callbacks=[MC,RP,ES], 
#           use_multiprocessing=False)
    

  def fit_count_based(self):
    if self.is_sequence_prediciton:
      print('count-based model is not available for sequence prediction tasks')
      return None


  def fit_linear_classifier(self):
    if self.is_sequence_prediciton:
      print('linear_classifier is not available for sequence prediction tasks')
      return None
    print('Linear classifier part')
    print('running 5-fold cv')
    parameters = {'C':[0.001, 0.05, 0.01, 0.5, 0.1, 1, 3, 5, 8, 10, 50, 100]}   
    model = LogisticRegression(penalty="l2", solver="liblinear", dual=False, multi_class="ovr", class_weight='balanced')
    clf = GridSearchCV(model, parameters, scoring= 'f1_macro', return_train_score= True, n_jobs=-1)
    out = clf.fit(self.tfxidf_train, self.label_train)
    best_c = out.best_params_['C']
    self.linear_classifier = LogisticRegression(C=best_c, penalty="l2", solver="liblinear", dual=False, multi_class="ovr", class_weight='balanced')
    self.linear_classifier.fit(self.tfxidf_train, self.label_train)
    dump(self.linear_classifier, os.path.join(self.model_path, 'logistic_regression_model.joblib'))
    y_pred = self.linear_classifier.predict(self.tfxidf_test)
    out = f1_score(self.label_test,y_pred,average=None)
    print(f'Best C: {best_c}, Best f1-scores: {out}')

    conf_mat = confusion_matrix(self.label_test,y_pred)
    print(f'Unweighted f1-score: {np.mean(out)}, Weighted f1-score: {f1_score(self.label_test,y_pred,average="weighted")}')
    sns.heatmap(conf_mat, annot=True, fmt="d",
                xticklabels=self.linear_classifier.classes_, yticklabels=self.linear_classifier.classes_)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

  def fit_SVM_classifier(self):
    if self.is_sequence_prediciton:
      print('SVM_classifier is not available for sequence prediction tasks')
      return None
    print('SVM classifier part')
    print('running 5-fold cv')
    parameters = {'C':[ 0.01, 0.1, 1, 10, 100]}   
    model = SVC(kernel='rbf', class_weight='balanced')
    clf = GridSearchCV(model, parameters, scoring= 'f1_macro', return_train_score= True, n_jobs=5)
    out = clf.fit(self.tfxidf_train, self.label_train)
    best_c = out.best_params_['C']
    self.SVM_classifier = SVC(C=best_c, kernel='rbf', class_weight='balanced')
    self.SVM_classifier.fit(self.tfxidf_train, self.label_train)
    dump(self.SVM_classifier, os.path.join(self.model_path, 'SVM_model.joblib'))
    y_pred = self.SVM_classifier.predict(self.tfxidf_test)
    out = f1_score(self.label_test,y_pred,average=None)
    print(f'Best C: {best_c}, Best f1-scores: {out}')

    conf_mat = confusion_matrix(self.label_test,y_pred)
    print(f'Unweighted f1-score: {np.mean(out)}, Weighted f1-score: {f1_score(self.label_test,y_pred,average="weighted")}')
    sns.heatmap(conf_mat, annot=True, fmt="d",
                xticklabels=self.SVM_classifier.classes_, yticklabels=self.SVM_classifier.classes_)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

  def eval_dl_text_classification(self):
    if self.is_sequence_prediciton:
      # conf_mat = entity_level_d(y_true, y_pred, B_tag, I_tag, P_tag, window_size, return_conf_mat=True)
      y_true_all = []
      y_pred_all = []
      for data in self.tf_test_dataset.take(1):
        y_pred = self.dl_model.predict(data[0])
        y_pred_all.append(np.argmax(y_pred, axis=2).flatten())
        y_true_all.append(np.argmax(data[1], axis=2).flatten())
      y_true_all = np.stack(y_true_all).flatten()
      y_pred_all = np.stack(y_pred_all).flatten()
      f1s = f1_score(y_true_all, y_pred_all, average=None)
      W_f1s = f1_score(y_true_all, y_pred_all, average='weighted')
      print(f'f1s: {f1s}')
      print(f'Unweighted f1-score: {np.mean(f1s)}, Weighted f1-score: {W_f1s}')
    else:
      y_pred = []
      y_true = []
      for ind, i in enumerate(self.tf_test_dataset.as_numpy_iterator()):
          # print(i)
          tmp_out = self.dl_model.predict(i[0])
          y_pred += list(np.argmax(tmp_out, axis=1))
          y_true += list(np.argmax(i[1], axis=1))
          if ind == 0:
            count_classes = np.sum(i[1], axis=0)
          else:
            count_classes += np.sum(i[1], axis=0)
      conf_mat = confusion_matrix(y_true, y_pred)
      print(conf_mat)
      r = np.sum(conf_mat,axis=1)
      p = np.sum(conf_mat,axis=0)
      ww = []
      for i in range(conf_mat.shape[0]):
          ww.append(conf_mat[i,i])
      ww= np.array(ww)
      z1 = np.divide(ww, r)
      z2 = np.divide(ww, p)
      f1s = 2*(z1*z2)/(z1+z2)
      print(f1s)
      print(f'Unweighted f1-score: {np.mean(f1s)}, Weighted f1-score: {np.sum(f1s*count_classes)/np.sum(count_classes)}')


  def fit_deep_learning(self, alpha=.25,gamma=2):
    cur_path = os.getcwd()
    if self.do_deep_learning:
      self.dl_model = self.cre_deep_learning_model()
      self.dl_model.summary()
    os.chdir(self.model_path)
    MC = keras.callbacks.ModelCheckpoint( 'deeplearning_model.h5', monitor="val_loss",verbose=0,save_best_only=True)
    RP = keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.1,min_lr=1e-6, patience=10)
    ES = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=20,restore_best_weights=True)
    if self.is_sequence_prediciton:
        self.dl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                              loss=categorical_focal_loss(alpha=alpha, gamma=gamma))
        hist = self.dl_model.fit(
          self.tf_train_dataset, 
          validation_data = self.tf_test_dataset, 
          epochs=500, 
          callbacks=[MC,RP,ES], 
          use_multiprocessing=False)
    else:
        P = keras.metrics.Precision()
        R = keras.metrics.Recall()
        self.dl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                              loss=categorical_focal_loss(alpha=alpha, gamma=gamma), metrics= [P,R])
        hist = self.dl_model.fit(
          self.tf_train_dataset, 
          validation_data = self.tf_test_dataset, 
          epochs=500, 
          callbacks=[MC,RP,ES], 
          use_multiprocessing=True,
          class_weight=self.class_weight)
        
    print('f1-scores')
    self.eval_dl_text_classification()
    plt.plot(hist.history['loss'], label='Loss')
    plt.plot(hist.history['val_loss'], label='Val_loss')
    plt.grid()
    plt.show()
    os.chdir(cur_path)



  # def fit_ensemble(self, WR, dropout_rate):
  #   prob_linear = self.linear_classifier.predict_proba(self.tfxidf_train)
  #   prob_dl = self.dl_model.predict(self.tf_train_dataset2, use_multiprocessing=True)
  #   dl2linear_mapper = []
  #   for i in self.linear_classifier.classes_:
  #     dl2linear_mapper.append(self.class_dict[i])

  #   prob_dl = prob_dl[:,dl2linear_mapper]
  #   y_true = keras.utils.to_categorical([self.class_dict[i] for i in self.label_train])[:,dl2linear_mapper]
  #   # y_true2 = np.argmax(y_true, axis=1)

  #   prob_linear_te = self.linear_classifier.predict_proba(self.tfxidf_test)
  #   prob_dl_te = self.dl_model.predict(self.tf_test_dataset, use_multiprocessing=True)
  #   y_true_te = keras.utils.to_categorical([self.class_dict[i] for i in self.label_test])[:,dl2linear_mapper]
  #   y_true_te2 = np.argmax(y_true_te, axis=1)
  #   # out1 = {}
  #   # uw_f1s = []
  #   # w_f1s = []
  #   # for i in range(0,101):
  #   #   tmp_i = i/100
  #   #   tmp_output =  np.argmax(prob_linear*tmp_i + prob_dl*(1-tmp_i), axis=1)
  #   #   out1[i] = f1_score(y_true2, tmp_output,average='macro')
  #   #   uw_f1s.append(out1[i])
  #   #   w_f1s.append( f1_score(y_true2, tmp_output,average='weighted'))
  #   # plt.plot(uw_f1s, label= 'Unweighted f1')
  #   # plt.plot(w_f1s, label= 'Weighted f1')
  #   # plt.grid()
  #   # plt.plot()
  #   # out2 = sorted(out1.items(), key= lambda x: x[1], reverse=True)

  #   # max_ratio = out2[0][0]/100
  #   # tmp_output =  np.argmax(prob_linear_te*(max_ratio) + prob_dl_te*(1-max_ratio), axis=1)
  #   # print(f"Unweighted f1: {f1_score(y_true_te2, tmp_output,average='macro')}, Weighted f1: {f1_score(y_true_te2, tmp_output,average='weighted')}")
  #   for i in range(20):
  #     print(f'linear_prediction: {prob_linear[i]}')
  #     print(f'dl_prediction: {prob_dl[i]}')
  #     print('-----')
  #   L2 = keras.regularizers.l2(WR)
  #   linear_in = keras.Input(shape=(len(self.class_dict)))
  #   x = keras.layers.Dense(len(self.class_dict)*3, kernel_regularizer= L2, bias_regularizer= L2)(linear_in)
  #   x = keras.layers.BatchNormalization()(x)
  #   x = keras.layers.LeakyReLU()(x)
    
  #   dl_in = keras.Input(shape=(len(self.class_dict)))
  #   y = keras.layers.Dense(len(self.class_dict)*3, kernel_regularizer= L2, bias_regularizer= L2)(dl_in)
  #   y = keras.layers.BatchNormalization()(y)
  #   y = keras.layers.LeakyReLU()(y)
    
  #   w = keras.layers.Add()([x,y])
  #   w = keras.layers.Dropout(dropout_rate)(w)
  #   w = keras.layers.Dense(len(self.class_dict)*2, kernel_regularizer= L2, bias_regularizer= L2)(w)
  #   w = keras.layers.BatchNormalization()(w)
  #   w = keras.layers.LeakyReLU()(w)
  #   # w = keras.layers.Dropout(dropout_rate)(w)
  #   # w = keras.layers.Dense(len(self.class_dict)*2, kernel_regularizer= L2, bias_regularizer= L2)(w)
  #   # w = keras.layers.BatchNormalization()(w)
  #   # w = keras.layers.LeakyReLU()(w)
  #   w = keras.layers.Dropout(dropout_rate)(w)
  #   w = keras.layers.Dense(len(self.class_dict), kernel_regularizer= L2, bias_regularizer= L2,activation='softmax')(w)
  #   self.ensemble_model = keras.Model(inputs = [linear_in, dl_in], outputs=w)
  #   self.ensemble_model.summary()
  #   P = keras.metrics.Precision()
  #   R = keras.metrics.Recall()
  #   self.ensemble_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
  #   loss='CategoricalCrossentropy', metrics= [P,R])
  #   cur_path = os.getcwd()
  #   os.chdir(self.model_path)
  #   MC = keras.callbacks.ModelCheckpoint( 'ensemble_model.h5', monitor="val_loss",verbose=0,save_best_only=True)
  #   RP = keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.1,min_lr=1e-6, patience=7)
  #   ES = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=30,restore_best_weights=True)
  #   tmp_cw = {}
  #   for ind, i in enumerate(dl2linear_mapper):
  #     tmp_cw[ind] = self.class_weight[i]
  #   print(prob_linear.shape)
  #   print(prob_dl.shape)
  #   print(y_true.shape)
  #   print(prob_linear_te.shape)
  #   print(prob_dl_te.shape)
  #   print(y_true_te.shape)
  #   print(tmp_cw)
  #   hist = self.ensemble_model.fit(
  #     x=[prob_linear, prob_dl], 
  #     y=y_true,
  #     batch_size = 1024,
  #     validation_data = ([prob_linear_te, prob_dl_te], y_true_te), 
  #     epochs=500, 
  #     callbacks=[MC,RP,ES], 
  #     class_weight=tmp_cw)
  #   y_pred = np.argmax(self.ensemble_model.predict([prob_linear_te, prob_dl_te]),axis=1)
  #   print(f"Unweighted f1: {f1_score(y_true_te2, y_pred,average='macro')}, Weighted f1: {f1_score(y_true_te2, y_pred,average='weighted')}")
  #   os.chdir(cur_path)

class Text_classification_for_prediction():
  def __init__(
    self,
    path_to_tfxidf: '(str) path to tf-idf model (.joblib extension)',
    model_path: '(str) path to the folder created by "Text_classification". it may contain both logistic and deep learning models or just one of them',
    engine: '(str) now only "deeplearning", "linear_classifier" and "both" are available' = 'all'):
    self.tfxidf_model = load(path_to_tfxidf)
    self.model_path = model_path
    self.logistic_regression_model = None
    self.SVM_model = None
    self.deeplearning_model = None
    self.engine = engine
    cur_path = os.getcwd()
    try:
      os.chdir(model_path)
    except:
      raise Exception(f"Wrong model path: {model_path}")
    Dirs = os.listdir()
    #if len(Dirs) != 4:
      #raise Exception(f"The model path: {model_path} should sotre only 1 folder for deeplearning model and 2 .joblib extension for logistic regression and SVM models and 1 .json extension for the model config")
    for Dir in Dirs:
      if '.json' in Dir:
        continue
      if 'joblib' == Dir.split('.')[-1]:
        if ('logistic' in Dir.lower()) and (self.engine == 'linear_classifier' or self.engine == 'all'):
          print('loading logistic regression model')
          self.logistic_regression_model = load(Dir)
        if ('svm' in Dir.lower()) and (self.engine == 'SVM_classifier' or self.engine == 'all'):
          print('loading SVM model')
          self.SVM_model = load(Dir)

      elif 'h5' == Dir.split('.')[-1]:
        if self.engine == 'deeplearning' or self.engine == 'all':
          print('loading deeplearning model')
          self.deeplearning_model = keras.models.load_model(Dir)
    os.chdir(cur_path)
    with open(os.path.join(self.model_path,'model_config.json'), 'r', encoding='utf-8') as fh:
      self.config = json.load(fh)
    self.class_mapping = {y:x for x,y in self.config['class_dict'].items()}
    self.tp = Text_processing(self.config['max_len'],self.config['min_len'], engine=self.config['engine'])
    if self.deeplearning_model:
      print('loading the embedder and tokenizer')
      self.WE = Word_Embedder(self.config['word_embedder'], self.config['max_len'],is_sequence_prediction=self.config['is_sequence_prediction'])
  
  def predict(
    self,
    text: '(str) raw sentence-level string'):
    outputs = {}
    if self.engine in ['all', 'linear_classifier', 'SVM_classifier']:
      text2 = ' '.join([i.strip() for i in self.tp.apply_rules_before(text).split(' ') if i.strip() != ''])
      vectorized_sentences = self.tfxidf_model.transform([text2])
    if self.engine == 'linear_classifier' or self.engine == 'all':
      if self.logistic_regression_model:
        y_pred = self.logistic_regression_model.predict_proba(vectorized_sentences)
        # print(f'logistic y_pred: {y_pred}')
        probas = {}
        for i in range(y_pred.shape[1]):
          probas[self.logistic_regression_model.classes_[i]] = y_pred[0,i]
        outputs['logistic_regression'] = probas
    if self.engine == 'SVM_classifier' or self.engine == 'all':
      if self.SVM_model:
        y_pred = self.SVM_model.predict(vectorized_sentences)
        # print(f'logistic y_pred: {y_pred}')
        outputs['SVM'] = y_pred
    if self.engine == 'deeplearning' or self.engine == 'all':
      if self.deeplearning_model:
        tokenized_sentence, masks, _ = self.tp.preprocessing([text], None, None, do_padding=True, return_mask=True)
        tokenized_sentence = tokenized_sentence[0]
        masks = masks[0]
        # print(f'tokenized sentence: {tokenized_sentence}')
        vectorized_words = np.reshape(self.WE.encode(tokenized_sentence), (1,self.config['max_len'], self.config['embedding_size']))
        masks = np.reshape(masks, (1, self.config['max_len']))
        
        y_pred = self.deeplearning_model.predict((vectorized_words, masks))
        # print(f'dl y_pred: {y_pred}')
        probas = {}
        for i in range(y_pred.shape[1]):
          probas[self.class_mapping[i]] = y_pred[0,i]
        outputs['deeplearning'] = probas
    return outputs
    

# ######################################
# #### Attention layer for ensemble ####
# class My_attention(keras.layers..Layer):

#   def __init__(self, num_class):
#       super().__init__()
#       self.num_class = num_class

#   def build(self, input_shape):  # Create the state of the layer (weights)
#     w_init = tf.random_normal_initializer()
#     self.w = tf.Variable(
#         initial_value=w_init(shape=(input_shape[-1], self.units),
#                              dtype='float32'),
#         trainable=True)
#     b_init = tf.zeros_initializer()
#     self.b = tf.Variable(
#         initial_value=b_init(shape=(self.units,), dtype='float32'),
#         trainable=True)

#   def call(self, inputs):  # Defines the computation from inputs to outputs
#       return tf.matmul(inputs, self.w) + self.b
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed