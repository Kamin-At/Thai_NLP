import pythainlp
from preprocessing import Text_processing
import numpy as np
import os

print(os.getcwd())
print(os.listdir())

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

max_len = 15
min_len = 1
min_len_character = 1
do_padding = True
return_mask = True
engine = 'newmm'
verbose = True

tokenizer = Text_processing(max_len,min_len,min_len_character,do_padding,return_mask,engine,verbose)

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

print(tokenizer.stopwords)
#print(tokenizer.get_dictionary())