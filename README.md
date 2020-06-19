## To train a model for text-classification
### Step 1. Inputs
### pandas dataframes which contain fields as following
- 'texts' field is 1 sample (raw sentence-level string) per row ==> no need to do any preprocessing (there is a number of default built-in preprocessing steps). If you want to insert or use some custom preprocessing step, define them in preprocessing_function.py and call them in preprocessing_rules.py. There are 3 main steps for the text-preprocessing.
  - Preprocessing before word tokenization: the defined functions recieve only a plain-text (string) and return a string. Example function: URL removing, repeating word removing.
  - Word tokenization: we mainly use 'newmm' algorithm as the defualt due to its flexibility and time complexity ==> easy to add new words and fast for large scale tokenization. For other tokenizing algorithms, please see the engine options from https://thainlp.org/pythainlp/docs/2.0/api/tokenize.html
  - Preprocessing after word tokenization: the defined functions recieve only a list of string (list of tokens) and return a lis of tokens. Example function: stopword removing, whitespace removing.

- 'labels' field is 1 label (string) per sample

### Step 2. Call prepare_data_for_text_classification function from text_classification.py
- for default setting, the function requires...
  - Training set (pandas DataFrame) (from step 1.)
  - Test set (pandas DataFrame) (from step 1.)
  - Max length (int) of the text_sequence ==> Longer sequences will be truncated. Smaller sequences will be padded.
  - Min length (int) of the text_sequence ==> shorter sequence will be dropped
  ### This function returns a tuple which is everything needed for training models

### Step 3. Construct Text_classification class from text_classification.py
- for default setting, the function requires...
  - the tuple received from step 2.
  - do_deep_learning (bool) ==> use deep learnining model or not
  - do_linear_classifier (bool) ==> use logistic regression or not
  - is_sequence_prediciton (bool) ==> if True, it is for sequence prediction, for example: named-entity recognition (unsupported for now). False is for senetence-level prediciton, for example: sentiment analysis and intent classification. 
  - folder name (string) after training each model, a new folder will be created to store the models and the models' configuration file in the trained_model folder.

### 4. Call fit_linear_classifier and fit_deep_learning functions which belong to Text_classification class.

## Models usage for text-classification (predeiction)
It is recommended to move the folder containing the trained models in "trained_models" to the "models for real deployment" folder to avoid overwriting the trained model with the new one (in case of training new models).

- Step 1. Construct Text_classification_for_prediction class from text_classification.py. To construct it, we need..
  - path_to_tfxidf (str) the path to tf-idf encoder (.joblib extension)
  - model_path (str)  the path to the folder containing the models and its configuration.

\** you dont need to specify anything because all information needed is already stored in configuration.json (same condition in the training process)**

- Step 2. Call predict function which belongs to  Text_classification_for_prediction class. this function requires only a plain text (string) as the input. (Collection of string is not supported)

# Future features
- ### Sequence prediction
- ### Other embedders (now, support only tf-idf and fasttext encoding)
- ### Custom deep learning model