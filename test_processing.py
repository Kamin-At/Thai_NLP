from preprocessing_functions import replace_url, spec_add_spaces, rm_useless_spaces, rm_useless_newlines, rm_brackets, replace_rep_nonum, replace_url, replace_wrep_post,ungroup_emoji, lowercase_all
import pythainlp
from preprocessing import Text_processing
import numpy as np

max_len = 15
min_len = 1
min_len_character = 1
do_padding = True
return_mask = True
# :func:`fix_html`,
#         :func:`pythainlp.util.normalize`,
#         :func:`spec_add_spaces`,
#         :func:`rm_useless_spaces`,
#         :func:`rm_useless_newlines`,
#         :func:`rm_brackets`
#         and :func:`replace_rep_nonum`.
#       - The default **post-rules** consists of :func:`ungroup_emoji`,
#         :func:`lowercase_all`,  :func:`replace_wrep_post_nonum`,
#         and :func:`remove_space`.
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
rules_after_tokenization = [replace_wrep_post, ungroup_emoji, lowercase_all]
stopwords = {'ข้าวมันไก่'}
engine = 'newmm'
verbose = True

tokenizer = Text_processing(max_len,min_len,min_len_character,do_padding,return_mask,rules_before_tokenization,rules_after_tokenization,stopwords,engine,verbose)

word_list = [
             'ฉันกินข้าวมันไก่กับปลาต้ม ก ก ก',
             'แมววิ่งหนี่หมา',
             '',
             'ไม่กิน',
             'โคตรรรรรรรรรรรรรรรรรบุฟเฟ่',
             'โคตรรรรรรรรรรรรรรรรรบุฟเฟ่https://colab.research.google.com/drive/1v7qA3U4-oqFtbR7_vXFCgcQDm#scrollTo=tKNWb7-RJUjZเาำพิพะา'
]

out = tokenizer.preprocessing(word_list)

for i in out:
  print(i)