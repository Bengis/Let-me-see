import decoder
import numpy as np
import metrics
import matplotlib.pyplot as plt


def ht_lr():    
    bleu=list()
    lr_dist = np.linspace(0.00050, 0.00054, 10, dtype = float)
    for i in range(10):
        lr = lr_dist[i]
        dc=None
        model=None
        dc = decoder.decoder()
        model = dc.model(lr)
        model=dc.fit_generator(model)   
        bleu1=metrics.evaluate_model(model,dc)
        bleu.append(bleu1)
    
    plt.plot(lr_dist, bleu, marker='o');
    plt.xlabel('Learning rate', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylabel('BLEU-1', fontsize=12)
    
def ht_dr():    
    bleu=list()
    lr=0.00051
    dr_dist = np.linspace(0.25, 0.75, 10, dtype = float)
    for i in range(10):
        dr = dr_dist[i]
        dc=None
        model=None
        dc = decoder.decoder()
        model = dc.model(lr, dr)
        model=dc.fit_generator(model)   
        bleu1=metrics.evaluate_model(model,dc)
        bleu.append(bleu1)
    
    plt.plot(dr_dist, bleu, marker='o');
    plt.xlabel('Dropout', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylabel('BLEU-1', fontsize=12)

def ht_lstm():    
    bleu=list()
    lr=0.00051
    dr=0.35
    lstm_dist = np.linspace(100, 500, 10, dtype = int)
    for i in range(10):
        lstm = lstm_dist[i]
        dc=None
        model=None
        dc = decoder.decoder()
        model = dc.model(lr, dr, lstm)
        model=dc.fit_generator(model)   
        bleu1=metrics.evaluate_model(model,dc)
        bleu.append(bleu1)
    
    plt.plot(lstm_dist, bleu, marker='o');
    plt.xlabel('LSTM unit size', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylabel('BLEU-1', fontsize=12)
    
def ht_epochs():    
    bleu=list()
    lr=0.00051
    dr=0.35
    lstm=400
    epochs_dist = np.linspace(1, 60, 20, dtype = int)
    for i in range(20):
        epochs=epochs_dist[i]
        dc=None
        model=None
        dc = decoder.decoder()
        bleu1=metrics.evaluate_model(model,dc)
        bleu.append(bleu1)
        model = dc.model(lr, dr, lstm)
        for j in range(epochs):
            model=dc.fit_generator(model)   
    
    plt.plot(epochs_dist, bleu, marker='o');
    plt.xlabel('Epochs', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylabel('BLEU-1', fontsize=12)

def train():    
    lr=0.00051
    dr=0.35
    lstm=400
    epochs=20
    for i in range(epochs):
        dc = decoder.decoder()
        model = dc.model(lr, dr, lstm)
        model=dc.fit_generator(model,epochs)
        metrics.evaluate_model(model,dc)
    
train()