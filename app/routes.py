from app import app
#from app.ular import *

from flask import url_for, render_template, request

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.metrics import categorical_accuracy
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


nama_ular = {0: 'Air Albino',
 1: 'Blue Insularis',
 2: 'Cincin Emas',
 3: 'Cincin Perak',
 4: 'Crocodylus',
 5: 'Gavial',
 6: 'Gekkonidae',
 7: 'Gonyosoma',
 8: 'Green Tree',
 9: 'Jali',
 10: 'Sanca Albertisi',
 11: 'Sanca Burma',
 12: 'Sanca Kembang',
 13: 'Testudines'}

#Fourth Model
classifier = Sequential()

classifier.add(Conv2D(64, (5, 5), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(96, (5, 5), input_shape = (30, 30, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (5, 5), input_shape = (13, 13, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 14, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [categorical_accuracy])
classifier.load_weights('ular_ep_23.h5')
classifier._make_predict_function()


def ini_apa_sih(path):    
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    result = classifier.predict_classes(test_image)
    
    return nama_ular[result[0]]

@app.before_request
def clear_trailing():
    from flask import redirect

    rp = request.path 
    if rp != '/' and rp.endswith('/'):
        return redirect(rp[:-1])

@app.route('/')
def ini_apa():
    return render_template('upload_ini.html')
   
@app.route('/ini_apa_upload', methods = ['GET', 'POST'])
def ini_apa_upload():
   if request.method == 'POST':
      f = request.files['file']
      path = './app/static/images' + url_for('ini_apa') + '_' + f.filename
      path_ini = 'static/images' + url_for('ini_apa') + '_' + f.filename
      f.save(path)
      #return ini_apa(path)
      nama_ini = ini_apa_sih(path)
      
      return render_template(
        'hasil_prediksi.html',
        ini = nama_ini,
        posisi = path_ini
      )
'''
with app.test_request_context():
    print(url_for('hello'))
    print(url_for('show_post', post_id=3))
    print(url_for('static', filename='style.css'))

'''
