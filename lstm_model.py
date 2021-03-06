from keras.layers import Dense,Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential, load_model ,save_model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras import regularizers
import time


class lstm_model():
    def __init__(self,nb_categories,tot_vals,cep_num):
        self.nb_categories=nb_categories
        self.sequence_shape=(tot_vals,cep_num)
        self.model= Sequential()
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.add(LSTM(1024,return_sequences=True,input_shape=self.sequence_shape,dropout=0.1))
        self.model.add(LSTM(1024,return_sequences=False,dropout=0.1))
        self.model.add(Dense(nb_categories, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        print(self.model.summary())
    def getmodel(self):
        return self.model


if __name__ == "__main__":
    model = lstm_model(5,40,2048)
