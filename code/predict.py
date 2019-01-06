import decoder
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import time
import pandas as pd

def predict_model(model, decoder, img):
    out = decoder.sequence_evaluation(model, decoder.image_encodings[img])
    out = re.sub('(\s*)startcap(\s*)','',out)
    out = re.sub('(\s*)endcap(\s*)','',out)
    return out

print("Loading data...")
start = time.time()
filename = '../result/model-epoch01-train_loss3.27-val_loss3.79-best.h5'
model = load_model(filename)
dc = decoder.decoder()
end = time.time()
print("Data loaded!.\nTime: %0.2f seconds." % (end - start))

images=list()
file='../result/testImages.txt'
test_files= pd.read_csv(file, delimiter='\t')
for i in range(int(len(test_files))/10):
    if test_files.iloc[i][0] not in images:
        images.append(test_files.iloc[i][0])
        
for image in images:
    start = time.time()
    out=predict_model(model,dc,image)
    out=out.replace(' ', '-')
    img=mpimg.imread("../data/Flickr8k_Dataset/" + image)
    mpimg.imsave("../result/images/"+ out + "-" + image, img, ) 
    imgplot = plt.imshow(img)
    plt.show()
    end = time.time()
    print("Time: %0.2f seconds.\nCaption: %s." % (end - start,out))