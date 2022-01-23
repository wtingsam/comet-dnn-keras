'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''
from __future__ import print_function
import keras
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from numpy import *
from glob import glob
import sys
import struct
import zipfile
import math
import time
#from model_unet import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from keras.callbacks import ModelCheckpoint

from comet_data import comet_data 
from model_cnn import *
from model_unet import *
from comet_parameters import *

import argparse

def main():
    args = sys.argv
    if len(args)!=2:
        print("./train_load_to_memory.py <parameter.dat>")
    par_path = args[1]
    comet_pars = comet_parameters(par_path)

    comet_pars.ls()
    input_zip_path = comet_pars.get_parameter("input_path")
    output_dir = comet_pars.get_parameter("output_dir")
    dir_name , tmp_name = os.path.split(input_zip_path)
    tmp_name = tmp_name.split('.')[:-1]
    name_prefix = ""
    for name in tmp_name:
        name_prefix+=(name+".")
    output_model_path=output_dir+name_prefix+"h5"
    output_result_summary=output_dir+name_prefix+"npz"

    verbose        = int(comet_pars.get_parameter("verbose"))
    load_evts      = int(comet_pars.get_parameter("n_events"))
    num_classes    = int(comet_pars.get_parameter("num_classes"))
    n_filters      = int(comet_pars.get_parameter("n_filters"))
    dropout        = float(comet_pars.get_parameter("dropout"))
    min_lr         = float(comet_pars.get_parameter("min_lr"))
    min_delta      = float(comet_pars.get_parameter("min_delta"))
    factor_lr      = float(comet_pars.get_parameter("factor_lr"))
    learning_rate  = float(comet_pars.get_parameter("learning_rate"))
    loss_func      = comet_pars.get_parameter("loss_func")
    batch_size     = int(comet_pars.get_parameter("batch_size"))
    epochs         = int(comet_pars.get_parameter("epochs"))
    shuffle        = bool(comet_pars.get_parameter("shuffle"))
    
    data           = comet_data(input_zip_path,verbose=verbose)
    batch_imgs, batch_lbls, batch_evts_lbls = data.load_event(load_evts)
    batch_imgs, batch_lbls = data.adjust_data(batch_imgs,batch_lbls)
    train_imgs, train_lbls, train_evts_lbls, test_imgs, test_lbls, test_evts_lbls = data.split_data(batch_imgs,batch_lbls,batch_evts_lbls,int(batch_imgs.shape[0]*0.1))
    # Model building
    h=int(data.header["image_height"])
    w=int(data.header["image_width"])
    c=int(data.header["max_colors"])
    wl=int(data.header["max_w_lbls"])
    nEntries=int(data.header["nEntries"])
    input_tensor = Input([w,h,c])
    model = get_unet(input_tensor,
                     n_filters=n_filters,
                     dropout=dropout,
                     batchnorm=True,
                     num_classes=num_classes,
                     mode=loss_func,
                     lr=learning_rate)
    model.summary()
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss',
                          factor=factor_lr,
                          patience=5,
                          min_lr=min_lr,
                          min_delta=min_delta,
                          verbose=verbose,
                          mode='min'),
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, verbose=verbose),
    ]
    
    results = model.fit(train_imgs,train_lbls,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(test_imgs, test_lbls),
                        verbose=verbose,
                        shuffle=shuffle,
                        callbacks=callbacks)
    
    train_loss = results.history['loss']
    val_loss   = results.history['val_loss']
    train_acc  = results.history[loss_func]
    val_acc    = results.history['val_'+loss_func]
    prediction = model.predict(test_imgs, batch_size=None, verbose=verbose, steps=None)
    
    # for i in range(5):
    #     print("----------------")
    #     a_test=prediction[i][:,:,0]
    #     print(a_test[a_test>0])
    # print("Finished prediction!!")
    
    np.savez(output_result_summary,
             test_imgs, test_lbls, test_evts_lbls, prediction,
             train_loss,val_loss,train_acc,val_acc)
    
if __name__ == '__main__':
    main()
