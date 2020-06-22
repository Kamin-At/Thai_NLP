from PIL import Image, ImageDraw,ImageFont
import PIL as pil
import numpy as np

def highlighted_text(
    texts: '(list[str])',
    labels: '(list[int]) the size must be the same as "text"',
    font: '(Font object)',
    font_size: '(int) font size'
    ):
  txt = Image.new('RGBA', (1500,font_size + 6), (255,255,255,0))
  # get a drawing context
  d = ImageDraw.Draw(txt)
  position = 5
  for ind, word in enumerate(texts):
    word = word.strip()
    if word == '-PAD-':
      break
    tmp_size = font.getsize(word)
    if labels[ind] == 0:
      colour = (255,255,255,255)
    elif labels[ind] == 1:
      colour = (0,255,127,255)
    elif labels[ind] == 2:
      colour = (51,51,255,255)
    else:
      colour = (255,255,255,255)
    d.text((position,3), word, font=font, fill=colour)
    #print(tmp_size)
    position += tmp_size[0] + 3
  display(txt)

def entity_level_d(y_true, y_pred, B_tag, I_tag, P_tag, window_size, return_conf_mat=True):
  end_ind = window_size
  ind = 0
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  while ind < end_ind:
    if y_true[ind] == P_tag:
      break
    if y_true[ind] != B_tag and y_true != I_tag:
      if y_pred[ind] == B_tag or y_pred[ind] == I_tag:
        fp += 1
      elif y_pred[ind] != B_tag and y_pred[ind] != I_tag:
        tn += 1
      else:
        print(f'found exception: y_true: {y_true[ind]}, y_pred: {y_pred[ind]}')
      ind += 1
    elif y_true[ind] == B_tag:
      begin_index = ind
      tmp_ind = ind + 1
      if tmp_ind == window_size:
        if y_pred[begin_index] == B_tag:
          tp += 1
        else:
          fn += 1
      else:
        #print(f'tmp_ind: {tmp_ind}')
        while tmp_ind < window_size and y_true[tmp_ind] == I_tag:
          tmp_ind += 1
        end_ind2 = tmp_ind
        still_correct = True
        #print(f'cond:  begin_index: {begin_index}, end_ind2: {end_ind2}')
        for i in range(begin_index, end_ind2):
          if i == begin_index:
            if y_pred[i] != B_tag:
              still_correct = False
              break
          else:
            if y_pred[i] != I_tag:
              still_correct = False
              break
        if still_correct:
          tp += 1
        else:
          fn += 1
      ind = tmp_ind
    #print(f'ind: {ind}, end_ind: {end_ind}, tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')
  if return_conf_mat:
    return np.array([[tp,fp],[fn,tn]])
  else:
    return tp, tn, fp, fn