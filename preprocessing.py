import regex as re
import numpy as np
import pythainlp

class Text_processing():
  def __init__(self,
               max_len: '(int) max number of tokens per sample',
               min_len: '(int) min number of tokens per sample',
               min_len_character: '(int) min number of characters per token', #recommend > 0 or 1 to automatically remove white spaces and error from the tokenizer 
               do_padding: '(bool) use "-PAD-" to pad the sentenses until their length is equal to max_len',
               return_mask: '(bool) if True also return list of booleans indicating where the -PAD- is (True for real tokens and False for -PAD- token)',
               rules_before_tokenization: '(list[function(str)]) list of functions taking sentence-level input string',
               rules_after_tokenization: '(list[function(list[str])]) list of functions taking list of tokens',
               stopwords: '(set[string]) set of stopwords',
               engine: '(str) engine used to tokenize sentences see: https://thainlp.org/pythainlp/docs/2.0/api/tokenize.html' = 'newmm',
               verbose: '(bool) if True print some comparisons of before and after processing texts' = False
               ):
    # Define rules_before_tokenization and rules_after_tokenization carefully (the order is important!!)
    self.max_len = max_len
    self.min_len = min_len
    self.min_len_character = min_len_character
    self.rules_after_tokenization = rules_after_tokenization
    self.rules_before_tokenization = rules_before_tokenization
    self.stopwords = stopwords
    self.engine = engine
    self.do_padding = do_padding
    self.verbose = verbose
    self.return_mask = return_mask
  
  def apply_rules_before(self,
                         text: '(str) string to apply rules',
                         rules: '(list[tuple[pattern to be replaced, word to replace]]) list of patters to be replaced using regex.sub()'
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
                        texts: '(list[str]) list of tokens to apply rules',
                        rules: '(list[tuple[pattern to be replaced, word to replace]]) list of patters to be replaced using regex.sub()'
                        ) -> '(str)':
    for func in rules:
      texts = func(texts)
    return texts

  def preprocessing(self,
                    texts: '(list[str]): list of input strings'
                    )-> 'if return_mask is True list[tuple(list[str], list[bool])] else list[list[str]]':
    Data = []
    for ind, text in enumerate(texts):

      if self.verbose:
        if ind % 5 == 0:
          print(f'text before preprocessing: {text}')

      text = self.apply_rules_before(text, self.rules_before_tokenization)
      #################################
      ### request to elastic search ###
      words = []
      for word in pythainlp.tokenize.word_tokenize(text, engine=self.engine):
        word = word.strip()
        if len(word) < self.min_len_character or word in self.stopwords:
          continue
        words.append(word)
      ################################
      text = self.apply_rules_after(words, self.rules_after_tokenization)
      if len(text) > self.max_len:
        text = text[:self.max_len]
        mask = [True] * self.max_len

      else:
        if self.do_padding:
          mask = [True] * len(text) + [False] * (self.max_len - len(text))
          for _ in range(len(text), self.max_len):
            text.append('-PAD-')
      if self.verbose:
        if ind % 5 == 0:
          print(f'text after preprocessing: {text}')
          print('----------------------------------')
      if self.return_mask:
        Data.append((text, mask))
      else:
        Data.append(text)
    return Data
