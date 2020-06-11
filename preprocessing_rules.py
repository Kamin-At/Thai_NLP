from preprocessing_functions import *
import pythainlp

#################################################################################################
##### Note: each processing function will be executed in order. So, the order does matter!! #####
#################################################################################################
# if you want to use your custom preprocessing function, define it in preprocessing_functions.py


#   rules_before_tokenization: '(Collection[function(str)]) Collection of functions taking sentence-level input string'        
rules_before_tokenization = [
    replace_url,
    pythainlp.util.normalize,
    spec_add_spaces,
    rm_useless_spaces,
    rm_useless_newlines,
    rm_brackets,
    replace_rep_nonum,
    replace_url
    ]

#  rules_after_tokenization: '(Collection[function(list[str])]) Collection of functions taking list of tokens'
rules_after_tokenization = [
    replace_wrep_post,
    ungroup_emoji, 
    lowercase_all
    ]