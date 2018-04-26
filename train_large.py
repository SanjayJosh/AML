from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from lstm_model import lstm_model
from Data import Data
import time
import os.path
from utils import mac_remove_file
import tensorflow as tf
def train():
    mac_remove_file()
    datamodel = Data(False)
    datamodel.make_path_lists()
    nb_categories=datamodel.class_num
    print(nb_categories)
    tb = TensorBoard(log_dir=os.path.join('logs'))
    checkpoint= ModelCheckpoint(filepath=os.path.join('checkpoints','lstm-best.hdf5'),verbose=1,save_best_only=True)
    early_stopper = EarlyStopping(patience=20)
    cep_num= 26
    tot_vals=1290
    batchsize=10
    epochs= 1000
    dnn_model = lstm_model(nb_categories,tot_vals,cep_num)
    model= dnn_model.getmodel()
    X_test,y_test = datamodel.load_all_in_memory(datamodel.testlist)
    X,y = datamodel.load_all_in_memory(datamodel.trainlist)
    starttime=time.time()
    model.fit(X,y,validation_data=(X_test, y_test),batch_size=batchsize,verbose=1,epochs=epochs,callbacks=[tb,checkpoint,early_stopper],shuffle=True)
    print("Training-Validation time is",time.time() - starttime)

train();
