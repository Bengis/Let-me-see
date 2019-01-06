import numpy as np
import keras.applications.vgg16 as vgg16
import keras.applications.inception_v3 as inceptionv3
from keras.preprocessing import image
from keras.models import Model
from pickle import dump
import progressbar as pb

def encode_process(tModel, model, path):
    if tModel==0:
        prediction=encode_process_VGG16(model, path)
    if tModel==1:
        prediction=encode_process_InceptionV3(model, path)
    return prediction

def encode_process_VGG16(model, path):
    processed_img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(processed_img)
    x = np.expand_dims(x, axis=0)
    x = vgg16.preprocess_input(x)
    image_final = np.asarray(x)
    prediction = model.predict(image_final)
    return prediction

def encode_process_InceptionV3(model, path):
    processed_img = image.load_img(path, target_size=(299,299))
    x = image.img_to_array(processed_img)
    x = np.expand_dims(x, axis=0)
    x = inceptionv3.preprocess_input(x)
    image_final = np.asarray(x)
    prediction = model.predict(image_final)
    return prediction

def encode_image(tModel):
    if tModel==0:
        model=encoding_images_using_VGG16()
    if tModel==1:
        model=encoding_images_using_InceptionV3()
    return model

def encoding_images_using_VGG16():
    vgg = vgg16.VGG16()
    vgg.layers.pop()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)
    return vgg

def encoding_images_using_InceptionV3():
    iv3 = inceptionv3.InceptionV3()
    iv3.layers.pop()
    iv3 = Model(inputs=iv3.inputs, outputs=iv3.layers[-1].output)
    return iv3

def extract_features(tModel, model):    
    image_features=dict()
    
    file_train = open("../data/Flickr8k_text/Flickr_8k.trainImages.txt")
    train_id=file_train.read().split('\n')[:-1]
    print("Train length:" + str(len(train_id)))
    
    print("Dev begin")
    file_dev = open("../data/Flickr8k_text/Flickr_8k.devImages.txt")
    dev_id=file_dev.read().split('\n')[:-1]
    file_dev.close()
    print("Dev length:" + str(len(dev_id)))
    
    print("Dev end")
    file_test = open("../data/Flickr8k_text/Flickr_8k.testImages.txt")
    test_id=file_test.read().split('\n')[:-1]
    file_test.close()
    print("Test length:" + str(len(test_id)))
    
    images = []
    images.extend(train_id)
    images.extend(dev_id)
    images.extend(test_id)
    print("Images length:" + str(len(images)))
    
    b = pb.ProgressBar(maxval=len(images),widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    b.start()
    i=1
    print("Encoding images")
    for img in images:
        path = "../data/Flickr8k_Dataset/"+str(img)
        image_features[img] = encode_process(tModel, model, path)
        b.update(i)
        i += 1  
    b.finish() 
    return image_features

tModel=1 #0:VGG,1:InceptionV3
model= encode_image(tModel)
features=extract_features(tModel,model)
print('Extracted Features: %d' % len(features))
if tModel==0:
    dump(features, open('../result/image_encodings_vgg.p', 'wb'))
if tModel==1:
    dump(features, open('../result/image_encodings_inception.p', 'wb'))

    
