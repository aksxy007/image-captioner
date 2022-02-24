import os
from flask import Flask, request, jsonify,Response, send_file
import numpy as np
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.models import load_model
app = Flask(__name__)
# CORS(app)
static_dir = 'images/'


def featureExtract(image_path):
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img = x
    img_fea_vec = model_new.predict(img)
    img_fea_vec = np.reshape(img_fea_vec, img_fea_vec.shape[1])
    return img_fea_vec


def generateDesc(photo, model, word_to_index, index_to_word, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    desc = in_text.split()
    desc = desc[1:-1]
    desc = ' '.join(desc)
    return desc


def generate_captions(imagePath):
    model = load_model('caption_generator.h5')
    word_to_index = load(open("./wordToIndex.pkl", "rb"))
    index_to_word = load(open("./indexToWord.pkl", "rb"))
    max_length = 40
    photo = featureExtract(imagePath)
    photo = photo.reshape((1, 2048))
    description = generateDesc(photo, model, word_to_index, index_to_word, max_length)
    print(description)
    return description


@app.route('/api', methods=['GET', 'POST'])
def apiHome():
    r = request.method
    if (r == "GET"):
        return 'Machine Learning Inference'
    elif (r == 'POST'):
        if 'file' not in request.files:
            return jsonify("file not found")
        file = request.files.get('file')
        if not file:
            return jsonify("file not uploaded")

        filename = 'sample.jpg'
        file.save(os.path.join(static_dir, filename))
        photo = static_dir + filename
        captions = generate_captions(photo)
        return jsonify(captions)

@app.route('/result')
def sendImage():
    return send_file(static_dir + 'sample.jpg', mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True)
