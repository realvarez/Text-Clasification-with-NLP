import re, unicodedata
import nltk
import boto3
import pandas as uwu
import json
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.models import model_from_yaml


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
    with open('/tmp/lemma.txt', 'rb') as fichero:
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
    number_array = []
    for i in tokens:
        number_array.append(vocabulary.index(i) + 1)
    return np.asarray(number_array)


# def word_to_vec(array_text, model):
#     model = Word2Vec.load('/tmp/modelWord2vec.bin')
#     array_vectors = []
#     for word in array_text:
#         array_vectors.append(model.wv['word'])
#     return array_vectors

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


def downloadFiles(area):
    s3 = boto3.client('s3')
    #downloadingFiles
    s3.download_file('titleproyectbucket2019', 'vocabulary.csv', '/tmp/vocabulary.csv')
    s3.download_file('titleproyectbucket2019', 'Herramientas/lemma.txt', '/tmp/lemma.txt')
    #downloadingModels
    if area=='Medio Ambiente':
        s3.download_file('titleproyectbucket2019', '/Models/MA/modelMA.yaml', '/tmp/modelMA.yaml')
        s3.download_file('titleproyectbucket2019', '/Models/MA/modelMA.h5', '/tmp/modelMA.h5')
    elif area=='Gobierno Corporativso':
        s3.download_file('titleproyectbucket2019', '/Models/GOB/modelGOB.yaml', '/tmp/modelGOB.yaml')
        s3.download_file('titleproyectbucket2019', '/Models/GOB/modelGOB.h5', '/tmp/modelGOB.h5')
    elif area=='Social Externo':
        s3.download_file('titleproyectbucket2019', '/Models/SEXT/modelSEXT.yaml', '/tmp/modelSEXT.yaml')
        s3.download_file('titleproyectbucket2019', '/Models/SEXT/modelSEXT.h5', '/tmp/modelSEXT.h5')
    elif area == 'Social Interno':
        s3.download_file('titleproyectbucket2019', '/Models/SINT/modelSINT.yaml', '/tmp/modelSINT.yaml')
        s3.download_file('titleproyectbucket2019', '/Models/SINT/modelSINT.h5', '/tmp/modelSINT.h5')


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
    downloadFiles(area)
    vocab = set(uwu.readcsv('/tmp/vocabulary.csv'))
    list_words_numeric = wordsToNumbers(list_words, vocab)
    list_words_numeric = pad_sequences(list_words_numeric, maxlen=10, dtype='object', padding='post', value=0)
    if area == 'Gobierno Corporativo':
        yaml_file = open('/tmp/modelGOB.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("/tmp/modelGOB.h5")
        return RNNmodel.predict(list_words_numeric)

    elif area == 'Medio Ambiente':
        yaml_file = open('/tmp/modelMA.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("/tmp/modelMA.h5")
        return RNNmodel.predict(list_words_numeric)

    elif area == 'Social Externo':
        yaml_file = open('/tmp/modelSEXT.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("/tmp/modelSEXT.h5")
        return RNNmodel.predict(list_words_numeric)

    elif area == 'Social Interno':
        yaml_file = open('/tmp/modelSINT.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        RNNmodel = model_from_yaml(loaded_model_yaml)
        RNNmodel.load_weights("/tmp/modelSINT.h5")
        return RNNmodel.predict(list_words_numeric)



def lambda_handler_make_process(event, context):
    body = json.loads(event['body'])
    area = body['area']
    text = body['text']
    prediction = get_dimension_of_text(text, area)
    return {
        'statusCode': 200,
        'body': json.dumps('La predicción es '+prediction)
    }