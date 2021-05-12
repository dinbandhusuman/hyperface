from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

json_file = open('Gendermodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("Gendermodel.h5")
print("Loaded model from disk")

def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (64, 64))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] == 1:
        prediction = 'male'
    else:
        prediction = 'female'
    print(prediction,img_name)


import os
def testing():
    path = 'Gender/Testing'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
       for file in f:
         if '.jpg' in file:
           files.append(os.path.join(r, file))

    for f in files:
       classify(f)
       print('\n')