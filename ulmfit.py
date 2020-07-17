from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.callbacks.tracker import TrackerCallback, EarlyStoppingCallback
from fastai.utils.mem import GPUMemTrace

from pythainlp.ulmfit import *
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

class SaveEncoderCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', every:str='improvement', name:str='bestmodel'):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch:int)->None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if isinstance(current, Tensor): current = current.cpu()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save_encoder(f'{self.name}')
                
class ULMfit_for_predict():
    def __init__(self,
                 model_path):
        data_lm = load_data(model_path, "language_model_data.pkl")
        data_lm.sanity_check()
        cur_path = os.getcwd()
        os.chdir(model_path)
        tr = pd.read_csv('train.csv')
        te = pd.read_csv('test.csv')
        os.chdir(cur_path)
        #classification data
        tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
        processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
                    NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=20)]

        data_cls = (ItemLists(model_path,train=TextList.from_df(tr, model_path, cols=["texts"], processor=processor),
                             valid=TextList.from_df(te, model_path, cols=["texts"], processor=processor))
            .label_from_df("labels")
            .databunch(bs=50)
            )
        data_cls.sanity_check()
#         print(len(data_cls.vocab.itos))

        config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
                     output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
        trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

        self.learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
        self.learn.load("bestmodel")
        self.classes = self.learn.data.train_ds.classes
    def predict(self,
                raw_text):
        dictionary = {}
        x = self.learn.predict(raw_text)[2].numpy()
        for ind, tmp_class in enumerate(self.classes):
            dictionary[tmp_class] = x[ind]
        return dictionary
    
class ULMfit_model():
    # to train using ULMfit
    # 1. Fine-tuning the language model is needed
    # 2. Train the classifier
    def __init__(self,
                 data_for_lm, 
                 model_path: '(str) locate to the folder dont specify in any extension'):
        self.model_path = model_path
        self.tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
        self.processor = [TokenizeProcessor(tokenizer=self.tt, chunksize=10000, mark_fields=False),
                    NumericalizeProcessor(vocab=None, max_vocab=60000, min_freq=2)]
        
        self.data_lm = (TextList.from_df(data_for_lm, self.model_path, cols="texts", processor=self.processor)
            .split_by_rand_pct(valid_pct = 0.2, seed = 123)
            .label_for_lm()
            .databunch(bs=48))
        self.data_lm.save('language_model_data.pkl')
        self.data_lm.sanity_check()
        print(f'number of training samples: {len(self.data_lm.train_ds)}')
        print(f'number of test samples: {len(self.data_lm.valid_ds)}')
        self.learn = None
        
    def fit_lm(self):
        config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False, tie_weights=True, out_bias=True,
             output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)

        trn_args = dict(drop_mult=1., clip=0.12, alpha=2, beta=1)

        self.learn = language_model_learner(self.data_lm, AWD_LSTM, config=config, pretrained=False, **trn_args)

        #load pretrained models
        self.learn.load_pretrained(**_THWIKI_LSTM)

        #train unfrozen
        print("training unfrozen")
#         with GPUMemTrace():
        self.learn.unfreeze()

        self.learn.fit_one_cycle(3, 1e-3, moms=(0.8, 0.7),
                                 callbacks=[SaveEncoderCallback(self.learn, every='improvement', monitor='accuracy', name='LM'),
                                            EarlyStoppingCallback(self.learn, min_delta=0.0, patience=5)])
    def fit_classifier(self,
                       tr,
                       te):
        self.processor = [TokenizeProcessor(tokenizer=self.tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=self.data_lm.vocab, max_vocab=60000, min_freq=20)]

        self.data_cls = (ItemLists(self.model_path,train=TextList.from_df(tr, self.model_path, cols=["texts"], processor=self.processor),
                             valid=TextList.from_df(te, self.model_path, cols=["texts"], processor=self.processor))
            .label_from_df("labels")
            .databunch(bs=50)
            )
        tr.to_csv(os.path.join(self.model_path, 'train.csv'), index=False)
        te.to_csv(os.path.join(self.model_path, 'test.csv'), index=False)
#         self.data_cls.save('data_cls.pkl')
        self.data_cls.sanity_check()
        print(f'total vocab size: {len(self.data_cls.vocab.itos)}')

        #model
        config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
                     output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
        trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

        self.learn = text_classifier_learner(self.data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
        #load pretrained finetuned model
        self.learn.load_encoder("./LM")

        #train unfrozen
#         with GPUMemTrace():
        self.learn.freeze_to(-1)
        self.learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
        self.learn.freeze_to(-2)
        self.learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
        self.learn.freeze_to(-3)
        self.learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
        self.learn.unfreeze()
        self.learn.fit_one_cycle(3, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7),
                           callbacks=[SaveModelCallback(self.learn, every='improvement', monitor='accuracy', name='bestmodel'),
                                      EarlyStoppingCallback(self.learn, min_delta=0.0, patience=5)])

        self.learn.load("bestmodel")

        #get predictions
        probs, y_true, loss = self.learn.get_preds(ds_type = DatasetType.Valid, ordered=True, with_loss=True)
        classes = self.learn.data.train_ds.classes
        y_true = np.array([classes[i] for i in y_true.numpy()])
        preds = np.array([classes[i] for i in probs.argmax(1).numpy()])
        prob = probs.numpy()
        loss = loss.numpy()

        to_df = np.concatenate([y_true[:,None],preds[:,None],loss[:,None],prob],1)
        probs_df = pd.DataFrame(to_df)
        probs_df.columns = ["category","preds","loss"] + classes
        probs_df["hit"] = (probs_df.category == probs_df.preds)
        probs_df["texts"] = te.texts
        (y_true==preds).mean()

        conf_mat = confusion_matrix(probs_df.category,probs_df.preds)
        sns.heatmap(conf_mat, annot=True, fmt="d",
                    xticklabels=classes, yticklabels=classes)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()
        
        f1s = f1_score(probs_df.category, probs_df.preds, average=None)
        print(f'f1-scores for each class: {f1s}')
        print(f'Weighted avg f1-score: {f1_score(probs_df.category, probs_df.preds, average="weighted")}, Unweighted avf f1-score: {np.mean(f1s)}')
        