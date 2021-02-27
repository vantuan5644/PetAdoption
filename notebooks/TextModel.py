from keras.models import Model
import keras.backend as K
import keras
from nltk.tokenize import TweetTokenizer
import nltk
isascii = lambda s: len(s) == len(s.encode())
tknzr = TweetTokenizer()
import jieba
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")
nltk.download('averaged_perceptron_tagger')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, GRU
import matplotlib.pyplot as plt
import numpy as np

model = None

def custom_tokenizer(text):
#     print(text)
    init_doc = tknzr.tokenize(text)
    retval = []
    for t in init_doc:
        if isascii(t): 
            retval.append(t)
        else:
            for w in t:
                retval.append(w)
    return retval

def build_emb_matrix(word_dict, emb_dict):
    embed_size = 300
    nb_words = len(word_dict)+1000
    nb_oov = 0
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    for key in tqdm(word_dict):
        word = key
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        nb_oov+=1
        embedding_matrix[word_dict[key]] = unknown_vector                    
    return embedding_matrix, nb_words, nb_oov

def load_text_feature_model(path_model='/'):
    global model
    model = keras.models.load_model(path_model)

def preprocess(Description):
    english_desc, chinese_desc = [], []
    tokens = set()
    word_dict = {}
    pos_count, word_count = 1, 1 # starts from 1, 0 for padding token
    e_d, c_d, eng_seq, pos_seq = [], [], [], []

    doc = custom_tokenizer(str(Description))
    for token in doc:
        if not isascii(token):
            c_d.append(token)
        else:
            e_d.append(token)
            if token not in word_dict:
                word_dict[token] = word_count
                word_count +=1
    english_desc.append(' '.join(e_d))
    chinese_desc.append(' '.join(c_d))
    descriptions = english_desc
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(descriptions)
    word_index = tokenizer.word_index

#     print('Found %s unique tokens' % len(word_index))
    sequences = tokenizer.texts_to_sequences(descriptions)
    max_len = 100
    max_words = 10000
    training_sample = 8000
    # val_sample = 12000
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
#     print("Shape of data: ", padded_sequences.shape)

    return padded_sequences

def predict_text(desc="abc", feature=False, model_path="/"):
    x = preprocess(desc)
    if model is None:
        load_text_feature_model(model_path)
        
    if feature==False:
        return model.predict(x)
    else:
        test_model = Model(model.input, model.layers[2].output)
        return test_model.predict(x)

