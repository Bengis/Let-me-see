import telepot
import decoder
import keras
from keras.models import load_model
import keras.applications.vgg16 as vgg16
from keras.models import Model
from keras.preprocessing import image
import re
import time
import numpy as np

print("Loading data...")
keras.backend.clear_session()
start = time.time()
dc = decoder.decoder()       
filename = '../result/model-epoch01-train_loss4.63-val_loss4.09-best.h5'
model = load_model(filename)
model._make_predict_function()
vgg = vgg16.VGG16()
vgg.layers.pop()
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)
vgg._make_predict_function()
end = time.time()
print("Data loaded!.\nTime: %0.2f seconds." % (end - start))

def extract_features(filename):
    processed_img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(processed_img)
    x = np.expand_dims(x, axis=0)
    x = vgg16.preprocess_input(x)
    image_final = np.asarray(x)
    feature = vgg.predict(image_final)
    return feature

def predict_model(model, decoder, img):
    out = decoder.sequence_evaluation(model, img)
    out = re.sub('(\s*)startcap(\s*)','',out)
    out = re.sub('(\s*)endcap(\s*)','',out)
    return out

def handle(msg):
    folder="../data/telegram/"
    content_type, chat_type, chat_id = telepot.glance(msg)
    line= str(msg['from']['username']) + ";" + str(msg['from']['first_name'])  + ";" + str(msg['from']['language_code']) + ";" + str(msg['date'])  
    if content_type != 'photo':
        bot.sendMessage(chat_id, "Please send me a photo. I make captions!")
    if content_type == 'photo':
        start = time.time()
        bot.sendMessage(chat_id, "Pretty photo. Let me see. Give me a second, please...")
        photo_id=msg['photo'][2]['file_id']
        filename=folder + photo_id + ".jpg"
        bot.download_file(photo_id, filename)
        img=extract_features(filename)
        out=predict_model(model,dc,img)
        end = time.time()
        line=line+";" + str(out) + ";" + str(end-start)
        bot.sendMessage(chat_id, "I see this: **" + str(out) + "** " + str(round(end-start,2)) + " seconds")
    f=open("../data/telegram/log.txt", "a+")
    f.write(line+"\n")
    f.close
    
token="YOUR TOKEN HERE"
bot = telepot.Bot(token)
bot.message_loop(handle)
print ('Listening ...')
while 1:
    time.sleep(10)