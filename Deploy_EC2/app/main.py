import re, unicodedata, nltk, json, ssl
import pandas as uwu
import numpy as np
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_yaml
import tensorflow as tf
from flask import jsonify

app = Flask(__name__)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
print('Instalando librerias de NLTK')
nltk.download('punkt')
nltk.download('stopwords')

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
    with open('/app/Herramientas/lemma.txt', 'rb') as fichero:
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
        return 'Texto a clasificar mal ingresado o muy corto (minimo 5 palabras)', 200
    if area not in ['Gobierno Corporativo', 'Medio Ambiente', 'Social Externo', 'Social Interno']:
        return 'El area ingresada no coincide, puede ser Gobierno Corporativo, Medio Ambiente, Social Externo o Social Interno', 200
    list_words = processText(text)
    vocab = list(uwu.read_csv('./Herramientas/vocabulary.csv'))
    list_words_numeric = wordsToNumbers(list_words, vocab)
    if area == 'Gobierno Corporativo':
        df = uwu.read_csv('./Herramientas/dims_gob.csv', sep=";")
        RNNmodel = tf.keras.models.load_model('model_gob.h5')
    elif area == 'Medio Ambiente':
        df = uwu.read_csv('./Herramientas/dims_amb.csv', sep=";")
        RNNmodel = tf.keras.models.load_model('./model_amb.h5')
    elif area == 'Social Externo':
        df = uwu.read_csv('./Herramientas/dims_soce.csv', sep=";")
        RNNmodel = tf.keras.models.load_model('./model_soce.h5')
    elif area == 'Social Interno':
        df = uwu.read_csv('./Herramientas/dims_soci.csv', sep=";")
        RNNmodel = tf.keras.models.load_model('./model_soci.h5')
    
    print("\n\n\nello, como estas")

    dim = RNNmodel.predict_classes(np.array([list_words_numeric]))[0]
    print("\n\n\nello, como estas")
    
    dim_str = df.Dimension[dim]
    return dim, dim_str, list_words

@app.route('/classify/v1', methods=["POST"])
def classify_text():
    body = request.get_json()
    if 'area' in body:
        area = body['area']
    else:
        return 'Area no encontrada', 200    
    if 'text' in body:
        text = body['text']
    else:
        return 'Texto no encontrado', 200
    dimension, dimension_string, palabras = get_dimension_of_text(text, area)
    response = {
            'dim_cod': int(dimension),
            'dim_str': dimension_string,
            'palabras': palabras
            }
    return response, 200

@app.route('/')
def presentation():
    return '''
    Desarrollo de Proyecto de tesis de Ricardo Alvarez Zambrano<br>
    Consiste en un clasificador de texto de dimensiones ESG para ESG Compass<br>
    Para clasificar texto se debe hacer una request "POST" a la ruta < /classify/v1 > y en el body incluir un json los siguientes campos<br>
    {<br>
        'area':'Gobierno Corporativo',<br>
        'text':'El texto que se desea clasificar'<br>
    }<br>
    Las areas posibles que acepta el modelo son:<br>
    - Gobierno Corporativo<br>
    - Social Externo<br>
    - Social Interno<br>
    - Medio Ambiente<br>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=80)
