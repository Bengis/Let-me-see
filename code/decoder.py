import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, Embedding, Dense, Dropout, Input
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(5050)

class decoder():
    
    def __init__(self):
        self.vocab_size = None
        self.max_length = None
        self.num_samples = None
        self.steps=None
        self.image_encodings = load(open("../result/image_encodings.p", "rb" ))
        self.train_captions = None
        self.dev_captions = None
        self.tokenizer=None
        self.train_captions=self.load_data('../result/trainImages.txt', self.train_captions)
        self.dev_captions=self.load_data('../result/devImages.txt', self.dev_captions)
        self.test_captions=None
        self.prepare_data()
       
    def load_data(self, file, captions):
        dataset = pd.read_csv(file, delimiter='\t')
        captions = dict()
        for i in range(int(len(dataset))):
            if dataset.iloc[i][0] not in captions:
                captions[dataset.iloc[i][0]]=list()
            captions[dataset.iloc[i][0]].append(dataset.iloc[i][1])
        return captions
    
    def load_test_data(self):
        self.test_captions=self.load_data('../result/testImages.txt', self.test_captions)

    def prepare_data(self):
        self.steps=len(self.train_captions)
        self.num_samples=0
        
        caption_length=[]
        for key, caption_list in self.train_captions.items():
            for caption in caption_list:
                 self.num_samples+=len(caption.split())-1
                 caption_length.append(len(caption.split())) 
     
        self.max_length = max(caption_length)
        self.tokenizer=self.create_tokenizer()
        self.vocab_size = len(self.tokenizer.word_index)+1
      
        print("Length captions = " + str(len(self.train_captions)))
        print("Max length = " + str(self.max_length))
        print("Vocab size = " + str(self.vocab_size))
                
    def convert_to_lines(self,captions):
        caption_list = list()
        for key in captions.keys():
            [caption_list.append(d) for d in captions[key]]
        return caption_list

    def create_tokenizer(self):
        lines = self.convert_to_lines(self.train_captions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer    

    def word_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def sequence_all(self, descriptions):
        in_1, in_2, out = list(), list(), list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                seq = self.tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    in_1.append(self.image_encodings[key][0])
                    in_2.append(in_seq)
                    out.append(out_seq)
        return np.array(in_1), np.array(in_2), np.array(out)
    
    def sequence_one(self, desc_list, img):
        in_1, in_2, out = list(), list(), list()
        for desc in desc_list:
            seq = self.tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                in_1.append(img)
                in_2.append(in_seq)
                out.append(out_seq)
        return np.array(in_1), np.array(in_2), np.array(out)
    
    def sequence_evaluation(self, model, img):
        in_text = 'startcap'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            out = model.predict([img,sequence], verbose=0)
            out = np.argmax(out)
            word = self.word_id(out, self.tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endcap':
                break
        return in_text
    
    def data_generator(self, captions):
        while 1:
            for key, desc_list in captions.items():
               img = self.image_encodings[key][0]
               in_1, in_2, out = self.sequence_one(captions[key], img)
               yield [[in_1, in_2], out]
    
    def model(self, lr, dr, lstm):
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(dr)(inputs1)
        fe2 = Dense(lstm, activation='relu')(fe1)
                
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, lstm, mask_zero=True)(inputs2)
        se2 = Dropout(dr)(se1)
        se3 = LSTM(lstm)(se2)
        
        decoder1 = add([fe2, se3])
        decoder2 = Dense(lstm, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
        return model
    
    def fit_no_generator(self, model,epochs):
        filepath = '../result/model-epoch{epoch:02d}-train_loss{loss:.2f}-val_loss{val_loss:.2f}.h5'
        #callbacks = [EarlyStopping(monitor='val_loss',patience=0), 
        #             ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
        callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
        model.fit_generator(self.data_generator(self.train_captions), epochs=epochs, steps_per_epoch=self.steps,callbacks=callbacks, validation_data=self.data_generator(self.dev_captions), validation_steps=len(self.dev_captions),shuffle=False)
        return model
    
    def fit_generator(self, model):
        filepath = '../result/model-epoch{epoch:02d}-train_loss{loss:.2f}-val_loss{val_loss:.2f}.h5'
        in1_train, in2_train, out_train = self.sequence_all(self.train_captions)
        in1_dev, in2_dev, out_dev = self.sequence_all(self.dev_captions)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        model.fit([in1_train, in2_train], out_train, epochs=1, verbose=2, callbacks=[checkpoint], validation_data=([in1_dev, in2_dev], out_dev),shuffle=False)
        return model