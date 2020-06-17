from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import Word_Embedder
from preprocessing import Text_processing

def prepare_data_for_text_classification(
    train_dataframe:'(pd.DataFrame) dataframe containing 2 fields ==> "texts" (sentence-level text) and "labels" (string)',
    test_dataframe:'(pd.DataFrame) dataframe containing 2 fields ==> "texts" (sentence-level text) and "labels" (string)',
    max_len: '(int) max length of all sentences (shorter sentences will be padded, longer sentences will be truncated)',
    min_len: '(int) min number of tokens per sample ==> if less than min_len, we drop that sample',
    n_gram_range: '(tuple[int,int]) (min_n_gram, max_n_gram)' = (1,3),
    min_df: '(int) consider only tokens which exist greater of equal to min_df documents'=5,
    engine: '(str) engine to be used for word embedding' = 'fasttext'
    ):
    d = tp.visualize_important_words(train_dataframe['texts'],train_dataframe['labels'],n_gram_range,min_df)
    train_tf = tp.tfxidf_encode(train_dataframe['texts'])
    test_tf = tp.tfxidf_encode(test_dataframe['texts'])
    WE = Word_Embedder(engine, max_len)
    print(train_dataframe['labels'].value_counts())
    u_label = {}
    for ind, i in enumerate(train_dataframe['labels'].value_counts().index):
        u_label[i] = ind
    #print(u_label)
    tp = Text_processing(max_len, min_len)
    out = tp.preprocessing(train_dataframe['texts'], train_dataframe['labels'], u_label, True, True)
    out2 = tp.preprocessing(test_dataframe['texts'], test_dataframe['labels'], u_label, True, True)
    
    tmp_label_train = np.argmax(out[1],axis=1)
    CW = compute_class_weight('balanced', np.unique(tmp_label_train), tmp_label_train)
    CW2 = {}
    for i in range(CW.shape[0]):
      CW2[i] = CW[i]
    print(f'CW2: {CW2}')
    x = WE.cre_tf_dataset(False, max_len, out[0][0], out[0][1], out[1], len(u_label))
    x2 = WE.cre_tf_dataset(True, max_len, out2[0][0], out2[0][1], out2[1], len(u_label))
    return {
        'unique_label': u_label, 'max_len': max_len, 'embedding_size': WE.get_embedding_size(), 'tfxidf_train': train_tf, 
        'tfxidf_label_train': train_dataframe['labels'], 'tfxidf_test': test_tf, 'tfxidf_label_test': test_dataframe['labels'],
        'tf_train_dataset': x, 'tf_test_dataset': x2, 'class_weight': CW2
        }
    #tc =Text_classification(u_label,max_len,WE.get_embedding_size(),True,True,False,train_tf,train_dataframe['labels'],test_tf,test_dataframe['labels'],x,x2)


class Text_classification():
  def __init__(self,
            #    class_dict: '(dict[str: int]) the key is for the class names, the value is for index of the classes ==> int ranging from 0 to num_unique_class -1',
            #    max_len: '(int) max number of tokens per sample',
            #    num_feature_per_word: '(int) encoding size (token-level)',
               prepared_data_dict: '(dict) output of prepare_data_for_text_classification()',
               do_deep_learning: '(bool) if True ==> use bi-GRU to predict'=True,
               do_linear_classifier: '(bool) if True ==> use linear classifier with TF-IDF features'=True,
               is_sequence_prediciton: '(bool) if True ==> Ex. NER, else ==> Ex. Sentiment Analysis'=False
               ):
            #    tfxidf_train: '(np.array) (for do_linear_classifier) from TfidfVectorizer.fit or .fit_transform'=None,
            #    label_train: '(Collection[str]) (for do_linear_classifier) Ex. ["angry", "happy", "angry",....]'=None,
            #    tfxidf_test: '(np.array) (for do_linear_classifier) from TfidfVectorizer.fit or .fit_transform'=None,
            #    label_test: '(Collection[str]) (for do_linear_classifier) Ex. ["angry", "happy", "angry",....]'=None,
            #    tf_train_dataset: '(tf.dataset) from Word_Embedder class'= None,
            #    tf_test_dataset: '(tf.dataset) from Word_Embedder class'= None):

    self.max_len = prepared_data_dict['max_len']
    self.num_feature_per_word = prepared_data_dict['embedding_size']
    self.class_dict = prepared_data_dict['unique_label']
    self.do_deep_learning = do_deep_learning
    self.do_linear_classifier = do_linear_classifier
    self.is_sequence_prediciton = is_sequence_prediciton
    self.tfxidf_train = prepared_data_dict['tfxidf_train']
    self.label_train = prepared_data_dict['tfxidf_label_train']
    self.tfxidf_test = prepared_data_dict['tfxidf_test']
    self.label_test = prepared_data_dict['tfxidf_label_test']
    self.tf_train_dataset = prepared_data_dict['tf_train_dataset']
    self.tf_test_dataset = prepared_data_dict['tf_test_dataset']
    self.dl_model = None
    self.class_weight = prepared_data_dict['class_weight']

    if do_deep_learning:
      self.dl_model = self.cre_deep_learning_model()
      self.dl_model.summary()
    
  def cre_deep_learning_model(
      self,
      dropout_dense=0.35,
      dropout_gru=0.35,
      gru_size=128,
      num_gru_layer=2,
      num_dense_layer=3,
      WR = 1e-3):
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
      SEQ_out = keras.layers.TimeDistributed(keras.layers.Dense(len(self.class_dict), activation= 'softmax'),name='seq_out', kernel_regularizer= L2, bias_regularizer= L2)(x, mask=masks)
      return keras.Model(inputs= [inputs, masks], outputs = SEQ_out)
    else:
      x = keras.layers.Flatten()(x)
      for i in range(num_dense_layer):
        x = keras.layers.Dense(128//(2**(i)), kernel_regularizer= L2, bias_regularizer= L2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
      SEQ_out = keras.layers.Dense(len(self.class_dict), activation= 'softmax', name='seq_out', kernel_regularizer= L2, bias_regularizer= L2)(x)
      return keras.Model(inputs= [inputs, masks], outputs = SEQ_out)
  
  def fit_linear_classifier(self):
    print('Linear classifier part')
    print('running 5-fold cv')
    parameters = {'C':[0.001, 0.01, 0.5, 0.1, 1, 5, 10, 100]}   
    model = LogisticRegression(penalty="l2", solver="liblinear", dual=False, multi_class="ovr", class_weight='balanced')
    clf = GridSearchCV(model, parameters, scoring= 'f1_macro', return_train_score= True)
    out = clf.fit(self.tfxidf_train.toarray(), self.label_train)
    best_c = out.best_params_['C']
    model = LogisticRegression(C=best_c, penalty="l2", solver="liblinear", dual=False, multi_class="ovr", class_weight='balanced')
    model.fit(self.tfxidf_train.toarray(), self.label_train)
    y_pred = model.predict(self.tfxidf_test.toarray())
    out = f1_score(self.label_test,y_pred,average=None)
    print(f'Best C: {best_c}, Best f1-scores: {out}')

    conf_mat = confusion_matrix(self.label_test,y_pred)
    print(model.score(self.tfxidf_test.toarray(),self.label_test))
    sns.heatmap(conf_mat, annot=True, fmt="d",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

  def eval_dl_text_classification(self):
    y_pred = []
    y_true = []
    for i in self.tf_test_dataset.as_numpy_iterator():
        # print(i)
        tmp_out = self.dl_model.predict(i[0])
        y_pred += list(np.argmax(tmp_out, axis=1))
        y_true += list(np.argmax(i[1], axis=1))

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
    print(f'Macro f1-score: {np.mean(f1s)}')

  def fit_deep_learning(self):
    MC = keras.callbacks.ModelCheckpoint('DL_model', monitor="val_loss",verbose=0,save_best_only=True)
    RP = keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.1,min_lr=1e-6, patience=4)
    ES = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=7,restore_best_weights=True)
    if self.is_sequence_prediciton:
        self.dl_model.compile(optimizer="adam",loss='CategoricalCrossentropy')
    else:
        P = keras.metrics.Precision()
        R = keras.metrics.Recall()
        self.dl_model.compile(optimizer="adam",loss='CategoricalCrossentropy', metrics= [P,R])
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
