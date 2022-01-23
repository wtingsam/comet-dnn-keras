'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import os
import numpy as np
from numpy import *
from glob import glob
import sys
import struct
import zipfile
from matplotlib import pyplot
#from matplotlib as plt
import math
#from model_unet import *
from model_cnn import *
from model_unet import *
from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint

#from models import *

DEBUG=0
DRAW=1

n_batch_requested_per_file = 64
batch_size = 32

normalise_q = 25000
normalise_t = 1500

max_pixels = 512
max_layer = 18
max_cell = 300
max_colors = 2
max_labels = 1

# reshape_size_h = 18 #83
# reshape_size_w = 249#54
reshape_size_h = 96
reshape_size_w = 64
# reshape_size_h = 18
# reshape_size_w = 310

def split_data(batch_imgs, batch_lbls, testing_number=128):
    nbatches = batch_imgs.shape[0]
    train_set_x = batch_imgs[0:nbatches-testing_number]
    train_set_y = batch_lbls[0:nbatches-testing_number]
    test_set_x = batch_imgs[nbatches-testing_number:]
    test_set_y = batch_lbls[nbatches-testing_number:]
    print("-----------------------------")
    print ("Training set:")
    print (train_set_x.shape)
    print (train_set_y.shape)
    print ("Validation set:")
    print (test_set_x.shape)
    print (test_set_y.shape)
    print("-----------------------------")
    return train_set_x,train_set_y,test_set_x,test_set_y
    
def _signal_background_label_normalise(lbl,q,t):
    """
    Normalise the background and signal for the model
    return a value from [0,1]
    """
    # Labels
    if lbl>0.5:
        lbl=1
    elif lbl < 0:
        lbl=0
    else:
        lbl=0.50
    # Charge
    tmp_q=q
    # translate and then normalise
    q_norm = 32*700 # maximum ADC sum in RECBE
    if tmp_q>=0:
        q += 100 # minumium value of q is -199 so I want a gap for background and no hits
    else:
        q = 0
    q/=q_norm
    # Drift time
    t_norm = 1500
    tmp_t=t
    if tmp_t>-150:
        t += 100
    else:
        t = 0
    t/=t_norm    
    return lbl,q,t

def _decode(data,offset):
    """
    data: binary file for whole file
    offset: bytes size offset for each event
    ---------
    return a list of labels[max_labels] 
    and images[max_pixels,max_pixels,max_colors]
    """
    images_t = []
    images_q = []
    labels = []    
    size_of_image = reshape_size_h*reshape_size_w
    for ip in range(size_of_image):
        # Labels
        from_ip = ip*4 + offset
        to_ip = from_ip + 4
        lbl = struct.unpack('f',data[from_ip:to_ip])[0]
        # ADC sum
        from_ip = (ip+size_of_image)*4 + offset
        to_ip = from_ip + 4
        q = struct.unpack('f',data[from_ip:to_ip])[0]
        #images_q.append(q/normalise_q)
        # Drift Time
        from_ip = (ip+size_of_image*2)*4 + offset
        to_ip = from_ip + 4
        t = struct.unpack('f',data[from_ip:to_ip])[0]
        # Normalisation
        lbl,q,t=_signal_background_label_normalise(lbl,q,t)
        # Push back
        labels.append(lbl)
        images_q.append(q)
        images_t.append(t)
    return images_t, images_q, labels

def load_data(input_zip_file):
    # Zip files
    zf = zipfile.ZipFile(input_zip_file)

    nfiles=len(zf.namelist())
    # Initialize batch arrays
    # shape (batch,height,width,colors)
    batch_imgs = np.ndarray((n_batch_requested_per_file*nfiles,reshape_size_h,reshape_size_w,max_colors),dtype=float32) 
    batch_imgs.fill(0)
    # shape (batch,size of images)
    #batch_lbls = np.ndarray((n_batch_requested_per_file*nfiles,reshape_size_h*reshape_size_w*max_labels),dtype=float32) 
    batch_lbls = np.ndarray((n_batch_requested_per_file*nfiles,reshape_size_h,reshape_size_w,max_labels),dtype=float32) 
    batch_lbls.fill(-1)
    # Loop zip files
    i_file = 0
    for filename in zf.namelist():
        f_bin = zf.read(filename)
        chunk_bytes = 4*reshape_size_h*reshape_size_w*3
        print("filename",filename);
        print("Maximum batch you can make :",len(f_bin),chunk_bytes,len(f_bin)/chunk_bytes)
        print("Requested :",n_batch_requested_per_file)
        for i in range(n_batch_requested_per_file):
            offset = chunk_bytes*i
            img_t, img_q, lbl = _decode(f_bin,offset)
            batch_imgs[i+i_file*n_batch_requested_per_file][:,:,0] = np.asarray(img_q,dtype=float32).reshape(reshape_size_h,reshape_size_w)
            batch_imgs[i+i_file*n_batch_requested_per_file][:,:,1] = np.asarray(img_t,dtype=float32).reshape(reshape_size_h,reshape_size_w)
            #batch_lbls[i+i_file*n_batch_requested_per_file] = np.asarray(lbl,dtype=float32).reshape(reshape_size_h*reshape_size_w)
            batch_lbls[i+i_file*n_batch_requested_per_file][:,:,0] = np.asarray(lbl,dtype=float32).reshape(reshape_size_h,reshape_size_w)
        i_file+=1        
    return batch_imgs, batch_lbls, nfiles

num_classes = reshape_size_h*reshape_size_w
epochs = 10
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_comet_trained_model.h5'

# # Load data
#input_name="merged_sig_100K_ana.0801.turnId"
#input_name="merged_ewen"
input_name="merged_ewen_bg1"
batch_images, batch_labels, nfiles = load_data("../outputs/"+input_name+".zip")
# Plot event
if DRAW == 1:
    image_1d_q = batch_images[0][:,:,0].flatten()
    image_1d_t = batch_images[0][:,:,1].flatten()
    label_1d = batch_labels[0]
    pyplot.subplot(341)
    pyplot.hist(image_1d_q)
    pyplot.yscale('log', nonposy='clip')
    pyplot.subplot(332)
    pyplot.hist(image_1d_t)
    pyplot.yscale('log', nonposy='clip')
    iev = 4
    pyplot.subplot(334)
    pyplot.imshow(batch_images[iev][:,:,0], cmap=pyplot.get_cmap('binary'))
    pyplot.subplot(335)
    pyplot.imshow(batch_images[iev][:,:,1], cmap=pyplot.get_cmap('binary'))
    pyplot.subplot(336)
    pyplot.imshow(batch_labels.reshape(n_batch_requested_per_file*nfiles,reshape_size_h,reshape_size_w,max_labels)[iev][:,:,0], cmap=pyplot.get_cmap('binary'))
    pyplot.subplot(337)
    pyplot.imshow(batch_images[iev+1][:,:,0], cmap=pyplot.get_cmap('binary'))
    pyplot.subplot(338)
    pyplot.imshow(batch_images[iev+1][:,:,1], cmap=pyplot.get_cmap('binary'))
    pyplot.subplot(339)
    pyplot.imshow(batch_labels.reshape(n_batch_requested_per_file*nfiles,reshape_size_h,reshape_size_w,max_labels)[iev+1][:,:,0], cmap=pyplot.get_cmap('binary'))
    pyplot.savefig('image_tq.png')
#    pyplot.show()

# Split data
x_train, y_train, x_test, y_test = split_data(batch_images, batch_labels, int(n_batch_requested_per_file*nfiles*0.25))

# #model = model_unet(input_size)
#model = model_cnn(x_train.shape[1:],num_classes=num_classes)
#model = model_mnist_cnn(x_train.shape[1:],num_classes=num_classes)

input_tensor = Input((reshape_size_h,reshape_size_w,max_colors))
#input_tensor = Input((128,128,2))
#print(input_tensor)

model = get_unet(input_tensor, n_filters=10, dropout=0.05, batchnorm=True)
model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
#mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
results = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[es]
                    )
# print('Not using data augmentation.')


train_loss = results.history['loss']
val_loss   = results.history['val_loss']
train_acc  = results.history['acc']
val_acc    = results.history['val_acc']

prediction = model.predict(x_test, batch_size=None, verbose=0, steps=None)

print(prediction.shape)
# First image, and then prediction and then targets
binwrite = open('output.gpu1.bin','wb')
for i in range(x_test.shape[0]):
    x_test[i][:,:,0].tofile(binwrite)
    x_test[i][:,:,1].tofile(binwrite)
    prediction[i][:,:,0].tofile(binwrite)
    y_test[i][:,:,0].tofile(binwrite)
        
with open("result.gpu1.txt", 'w') as f:
    f.write('train_loss train_acc test_loss test_acc\n')
    for a,b,c,d  in zip(train_loss,train_acc,val_loss,val_acc):
        f.write(str(a)+" "+str(b)+" "+str(c)+" "+str(d)+'\n')

# pyplot.figure(figsize=(8, 8))
# pyplot.title("Learning curve")
# pyplot.plot(results.history["loss"], label="loss")
# pyplot.plot(results.history["val_loss"], label="val_loss")
# pyplot.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
# pyplot.xlabel("Epochs")
# pyplot.ylabel("log_loss")
# pyplot.legend();

# pyplot.savefig("learning_curve.png")
