from text_classification import prepare_data_for_text_classification
from Classifiers import compare_classifiers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mapping(text):
    return classes.index(text)
if __name__ == '__main__':
    df = pd.read_csv('train_RS_123.csv')
    df2 = pd.read_csv('test_RS_123.csv')
    all_df = pd.concat((df,df2)).sample(frac=1.0).reset_index(drop=True)
    classes = list(df['labels'].unique())

    all_df['texts'] = all_df['texts'].apply(str)
    df2['texts'] = df2['texts'].apply(str)

    out = prepare_data_for_text_classification(train_dataframe=all_df, 
                                         test_dataframe=df2, 
                                         max_len=128, 
                                         min_len=2, 
                                         n_gram_range=(1,3),
                                         min_df=2,
                                         word_embedder='fasttext',
                                         tfxidf_path='tf-idf_encoder2',
                                         engine='newmm',
    #                                      threshold_tfxidf=0.01,
                                         verbose=False)
    print(f'max in feature: {np.max(out["tfxidf_train"])}, min in feature: {np.min(out["tfxidf_train"])}')
    a = compare_classifiers(X=out['tfxidf_train'], 
                    Y=out['tfxidf_label_train'].map(mapping).values, 
                    n_folds=5, 
                    use_pca = 0.999,
                    random_state=123)
    a.fit_models()
    print(classes)