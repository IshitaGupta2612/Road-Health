import os
import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from os import listdir
from tensorflow.keras import backend as K

# importing Flask and other modules
from flask import Flask, request, render_template

K.clear_session()

app = Flask(__name__)

@app.route('/')  
def home ():  
    return render_template("upload.html")  

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
im = ''
result = '...'
percentage = '...'
i = 0
imageName = ''
solution = ''
@app.route("/uploadMain")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    global im, result, percentage , i , imageName , solution
    target = os.path.join(APP_ROOT, 'static\\')
    print(f'Target : {target}')
 
    for file in request.files.getlist("file"):
        print(f'File : {file}')
        i += 1
        imageName = str(i) + '.JPG'
        filename = file.filename
        destination = "/".join([target, imageName])
        print(f'Destination : {destination}')
        file.save(destination)
        print('analysing Image')
        try:
            image = os.listdir('static')
            im = destination
            print(f'Analysing Image : {im}')
        except Exception as e:
            print(e)
        result = "Failed to Analyse"
        percentage = "0 %"
        try:
            detect()
        except Exception as e:
            print(f'Error While Loading : {e}')  
    return render_template('complete.html', name=result, accuracy=percentage , img = imageName , soln = solution)


def detect():
    global im, result, percentage
    global size
    # OG size = 300
    size = 300
    model = Sequential()
    model = load_model('full_model.h5')


    ## load Testing data : non-pothole 
    nonPotholeTestImages = glob.glob('static/1.JPG')
    print(im)
    test2 = [cv2.imread(img,0) for img in nonPotholeTestImages]
    # train2[train2 != np.array(None)]
    for i in range(0,len(test2)):
        test2[i] = cv2.resize(test2[i],(size,size))
    temp4 = np.asarray(test2)

    '''
    ## load Testing data : potholes 
    potholeTestImages = glob.glob("C:/Users/admin/Downloads/road health/My Dataset/test/Pothole/*.jpg")
    test1 = [cv2.imread(img,0) for img in potholeTestImages]
    # train2[train2 != np.array(None)]
    for i in range(0,len(test1)):
        test1[i] = cv2.resize(test1[i],(size,size))
    temp3 = np.asarray(test1)
    '''


    X_test = []
    #X_test.extend(temp3)
    X_test.extend(temp4)
    X_test = np.asarray(X_test)

    X_test = X_test.reshape(X_test.shape[0], size, size, 1)



    #y_test1 = np.ones([temp3.shape[0]],dtype = int)
    y_test2 = np.zeros([temp4.shape[0]],dtype = int)

    y_test = []
    #y_test.extend(y_test1)
    y_test.extend(y_test2)
    y_test = np.asarray(y_test)

    y_test = np_utils.to_categorical(y_test)


    print("")
    X_test = X_test/255
    tests = model.predict(X_test)
    for i in range(len(X_test)):
	    print(">>> Predicted %d = %s" % (i,tests[i]))

    
    result = tests[i]  
    percentage = float("{0:.2f}".format(result[1] * 100))
    if result[1] > 0.80:
        result="Yes"  
    
        
if __name__ == "__main__":
    app.run(port=4555, debug=True)
