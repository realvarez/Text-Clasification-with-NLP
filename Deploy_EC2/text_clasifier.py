import re, unicodedata, nltk, boto3, json
import pandas as uwu
import numpy as np
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_yaml

app = Flask(__name__)

def remove_stop_words(list_tokens):
    stop = stopwords.words('spanish')
    for token in list_tokens:
        if token.lower() in stop:
            list_tokens.pop(list_tokens.index(token))
    return list_tokens


def remove_void_elements(tokens):
    for token in tokens:
        if token == '' or token == " " or token == "  ":
            tokens.pop(tokens.index(token))
    return tokens


def normalize_text(tokens):
    for token in tokens:
        pos = tokens.index(token)
        if token.isdigit() or token.isalpha() == False:
            tokens.pop(pos)
        else:
            token = re.sub(r'[^\w\s]', ' ', token)
            if token != ' ' or token != '':
                tokens[pos] = unicodedata.normalize('NFKD', token.lower()).encode('ascii', 'ignore').decode('utf-8','ignore')
            else:
                tokens.pop(pos)
    return tokens


def create_lemma_dict():
    lemmaDiccionario = {}
    with open('./lemma.txt', 'rb') as fichero:
        datos = (fichero.read().decode('utf8').replace(u'\r', u'').split(u'\n'))
        datos = ([avance.split(u'\t') for avance in datos])
    for avance in datos:
        if len(avance) > 1:
            lemmaDiccionario[avance[1]] = avance[0]
    return lemmaDiccionario

def lemmatize(lemmaDiccionario, word):
    return lemmaDiccionario.get(word, word + u'')

def lemmatize_words(words):
    lemmaDiccionario = create_lemma_dict()
    new_words = []
    for palabra in words:
        new_word = lemmatize(lemmaDiccionario, palabra)
        new_words.append(new_word)
    return new_words

def wordsToNumbers(tokens, vocabulary):
    number_array = np.zeros(10, dtype=int)
    for i in range(len(tokens)):
        if i < 10 and tokens[i] in vocabulary:
            number_array[i] = vocabulary.index(i) + 1
    return number_array

def processText(Text):
    list_text = nltk.word_tokenize(Text)
    list_text = remove_stop_words(list_text)
    list_text = remove_void_elements(list_text)
    list_text = normalize_text(list_text)
    list_text = lemmatize_words(list_text)
    if len(list_text) > 10:
        return list_text[:10]
    else:
        return list_text

def get_dimension_of_text(text, area):
    if not isinstance(text, str) or len(text) < 5:
        return {
            'statusCode': 200,
            'body': json.dumps('Texto a clasificar mal ingresado o muy corto (minimo 5 palabras)')
        }
    if area not in ['Gobierno Corporativo', 'Medio Ambiente', 'Social Externo', 'Social Interno']:
        return {
            'statusCode': 200,
            'body': json.dumps('El area ingresada no coincide, puede ser Gobierno Corporativo, Medio Ambiente, Social Externo o Social Interno')
        }
    list_words = processText(text)
    vocab = list(uwu.read_csv('./vocabulary.csv'))
    list_words_numeric = wordsToNumbers(list_words, vocab)
    if area == 'Gobierno Corporativo':
        yaml_file = open('./GOB/modelGOB.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("./GOB/modelGOB.h5")
        
        return RNNmodel.predict_classes(np.array([list_words_numeric]))

    elif area == 'Medio Ambiente':
        yaml_file = open('./MA/modelMA.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("./MA/modelMA.h5")
        
        return RNNmodel.predict_classes(np.array([list_words_numeric]))

    elif area == 'Social Externo':
        yaml_file = open('./SEXT/modelSEXT.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("./SEXT/modelSEXT.h5")
        return RNNmodel.predict_classes(np.array([list_words_numeric]))

    elif area == 'Social Interno':
        yaml_file = open('./SINT/modelSINT.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("./SINT/modelSINT.h5")
        return RNNmodel.predict_classes(np.array([list_words_numeric]))


@app.route('/classify/v1', methods=["POST"])
def classify_text():
    body = request.get_json()
    area = body['area']
    text = body['text']
    prediction = get_dimension_of_text(text, area)
    print(prediction[0])
    return {
        'statusCode': 200,
        'body': json.dumps(int(prediction[0]))
    }

# app.run(port=5000, debug =False)
