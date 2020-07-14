import fasttext

#### Thai + Eng fasttext ####
def fasttext_load():
    model = fasttext.load_model('encoders/cc.th.300.bin')
    return model
def fasttext_encode(model, text):
    return model[text]



FASTTEXT_ENCODER = {
    'load_model': fasttext_load,
    'encode': fasttext_encode,
    'encode_size': 300}

registered_encoders = {
    'fasttext': FASTTEXT_ENCODER}

