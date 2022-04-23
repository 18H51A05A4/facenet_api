from flask import Flask,jsonify,request,render_template
from PIL import Image 
import numpy as np 
import keras.models as kerasmodels
from flask_cors import CORS,cross_origin
from io import BytesIO
import re
import base64
import os
from numpy import save,load

app = Flask(__name__,template_folder="template")

CORS(app)

model = kerasmodels.load_model('facenet_keras.h5')

face_embedding = None

# def extract_face(filename, required_size=(160, 160)):
#     # load image from file
#     image = Image.open(filename)
#     # convert to RGB, if needed
#     image = image.convert('RGB')
#     image = image.resize(required_size)
#     face_array = np.asarray(image)
#     print(face_array)
#     return face_array

def get_embedding(face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]

def findCosineDistance(a, b):
    x = np.dot(np.transpose(a),b)
    y = np.dot(np.transpose(a),a)
    z = np.dot(np.transpose(b),b)
    return (1 - (x / (np.sqrt(y) * np.sqrt(z))))



@app.route('/')
def working():
    return jsonify("api working")


@app.route('/verify',methods=["POST"])
def verify():
    res = request.form['image']
    image_data = re.sub('^data:image/.+;base64,', '', res)
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert('RGB')
    image = image.resize((160, 160))
    image = np.asarray(image)
    print(image)
    image_embedding = get_embedding(image)
    face_embedding=load('data.npy')
    print(image_embedding.shape,image_embedding)
    print(face_embedding.shape,face_embedding)
    res = findCosineDistance(image_embedding,face_embedding)
    if(res<0.5):
        print("verified")
        return jsonify("person verification successful " + str(res) )
    else:
        print('not verified')
        return jsonify("person is not verified " + str(res))
    # return jsonify("done")


@app.route('/get_embeddings',methods=["POST"])
def get_embeddings():
    res = request.form['image']
    image_data = re.sub('^data:image/.+;base64,', '', res)
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert('RGB')
    image = image.resize((160, 160))
    image = np.asarray(image)
    print(image)
    global face_embedding 
    face_embedding= get_embedding(image)
    save('data.npy',face_embedding)
    print(type(face_embedding))
    print('embeddings stored')
    return jsonify("face mapping stored")
    

@app.route('/check',methods=["POST"])
def method():
    res = request.form['image']
    print(res)
    image_data = re.sub('^data:image/.+;base64,', '', res)
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert('RGB')
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    print(face_array)
    return jsonify("post done")


if __name__ == "__main__":
    app.run(debug=True,port=5000)