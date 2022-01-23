import os
import numpy as np
from numpy import *
from glob import glob
import sys
import struct
import zipfile
from collections import OrderedDict
from collections import defaultdict
import time
import keras
from keras.models import Sequential
from keras.utils import np_utils

class ImageSizeDontMatchException(Exception):
    pass

class NoSuchPathException(Exception):
    pass

class TooManyBinaryFilesException(Exception):
    pass

class comet_data(keras.utils.Sequence):
    """Custom generator"""

    def __init__(self, data_path,
                 batch_size=1, num_of_class=3, verbose=0):
        """construction   

        :param data_path: Paths of the zip file
        :param batch_size: Batch size  
        :param width: Image width  
        :param height: Image height  
        :param ch: Num of image channels  
        :param num_of_class: Num of classes  
        """
        width=128
        height=128
        ch=2
        self.data_path = data_path
        self.verbose = verbose
        """
        All information related to the input data should be changed here
        TODO: Or use add/remove using pop/push function of the class
        """
        ### Define dictionaries
        self.header = OrderedDict()
        self.header["image_height"] = 0
        self.header["image_width"] = 0
        self.header["max_colors"] = 0
        self.header["max_w_lbls"] = 0
        self.header["max_e_lbls"] = 0
        self.header["nEntries"] = 0
        self.norm = OrderedDict()
        self.norm["integratedADC"] = 10
        self.norm["driftTime"] = 1500
        self.a_image = OrderedDict()
        self.a_image["integratedADC"] = []
        self.a_image["driftTime"] = []
        self.a_image["isSignal"] = []
        self.a_image["turnId"] = []
        self.a_evt_lbls = OrderedDict()
        self.a_evt_lbls["max_layer_id"] = 0
        self.a_evt_lbls["max_turn_id"] = 0
        self.a_evt_lbls["pt"] = 0
        self.a_evt_lbls["pz"] = 0
        self.a_evt_lbls["ix"] = 0
        self.a_evt_lbls["iy"] = 0
        self.a_evt_lbls["iz"] = 0
        self.a_evt_lbls["trigger_number"] = 0
        
        ## Define class members
        self.zf = None
        self.data_archive = None
        self.end_of_evt = True
        # Variables related to sizes
        self.nEntries_train = 0
        self.nEntries_test = 0        
        self.total_pixels = 0
        # Variables related to bytes
        self.size_float32 = 4
        self.header_bytes = self.size_float32*len(self.header)
        self.evt_lbl_size = self.size_float32*len(self.a_evt_lbls)
        self.evt_bytes = 0
        self.all_evt_bytes = 0        
        self.c_lbl_offset = 0
        if self.verbose >= 1:
            self._print_data_defintion()
        self.count_loaded = 0
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.ch = ch
        self.num_of_class = num_of_class # Signal background and empty
        self.num_batches_per_epoch = 0

        # Some initial actions of the class
        self._load_header()
        self.on_epoch_end()
        
    def __getitem__(self, idx):
        """Get batch data   

        :param idx: Index of batch  

        :return imgs: numpy array of images 
        :return labels: numpy array of label  
        """
        # If this is already end of events, load again the data archive        
        if self.end_of_evt is True:
            self._close_file()
            self._open_file()
            self.data_archive.read(self.header_bytes)
            self.end_of_evt = False
            
        nEvents=int(self.header["nEntries"])
        # Make sure it does not exceed the total events
        load_evts = self.batch_size
        if self.count_loaded > nEvents:
            self.end_of_evt = True
            load_evts = nEvents - self.batch_size*(idx-1)
            print("## WARNING end of the events, loaded:",load_evts)
        # Initialize the images and masks
        # Load data from
            
        batch_imgs, batch_lbls, batch_evts_lbls = self.load_event(load_evts)
        s=batch_lbls.shape
        Y = batch_lbls[:,:,:,0].reshape(s[0],s[1],s[2],1)
        self.count_loaded+=load_evts
        return batch_imgs, batch_lbls

    def __len__(self):
        """Batch length"""
        return self.num_batches_per_epoch


    def on_epoch_end(self):
        """Task when end of epoch"""
        pass

    def split_data(self,batch_imgs, batch_lbls, batch_evts_lbls, testing_number=128):
        nbatches = batch_imgs.shape[0]
        train_set_x = batch_imgs[0:nbatches-testing_number]
        train_set_y = batch_lbls[0:nbatches-testing_number]
        train_set_evts_y = batch_evts_lbls[0:nbatches-testing_number]        
        test_set_x = batch_imgs[nbatches-testing_number:]
        test_set_y = batch_lbls[nbatches-testing_number:]
        test_set_evts_y = batch_evts_lbls[nbatches-testing_number:]
        
        print("-----------------------------")
        print ("Training set:")
        print (train_set_x.shape)
        print (train_set_y.shape)
        print ("Validation set:")
        print (test_set_x.shape)
        print (test_set_y.shape)
        print("-----------------------------")
        return train_set_x,train_set_y,train_set_evts_y,test_set_x,test_set_y,test_set_evts_y
    
    def _print_data_defintion(self):
        print("""
        ----------------------------------------------------------------
        | This is the defintiion of binary                             |
        | defined by                                                   |
        | app/root2binary_samroot.cxx                                  |
        |                                                              |
        | Headers:                                                     |
        |                                                              |
        """)
        for idx, key in enumerate(self.header):
            print("""            %d %s"""%(idx,key))
        print("""
        | Body:                                                        |
        | IMAGE and PIXEL LABELS:                                      |
        | Current order:                                               |
        """)
        for idx, key in enumerate(self.a_image):
            print("""            %d %s"""%(idx,key))
        print("""
        | integratedADC | DriftTime | IsSignal | turnId                |
        |                                                              |
        |                                                              |
        | - image_height*image_width*(max_colors+max_wire_lbls)        |
        | - def: c000: c:color, 000 = 1st_color pixel_x pixel_y        |
        | - def: l000: l:label, 000 = 1st_label pixel_x pixel_y        |
        |                                                              |
        | - e.g. [c000][c100][l000][l100][c010][c110][l010][l110]...   |
        |        [c001][c101][l001][l101][c011][c111][l011][l111]...   |
        |        :                      :                       :      |
        |        :                      :                       :      |
        |        :                      :                       :      |
        |                                                              |
        |                                                              |
        | EVENT LABELS:                                                |
        | The initial is at the first hit on                           |
        | Cylindricial Drift Chamber                                   |
        """)
        for idx, key in enumerate(self.a_evt_lbls):
            print("""            %d %s"""%(idx,key))
        print("""
        |                                                              |
        |                                                              |
        ----------------------------------------------------------------

        """)
    def _open_file(self):
        path=self.data_path
        if os.path.exists(path) is False:
            raise NoSuchPathException("## Error: Cannot find the path")

        self.zf = zipfile.ZipFile(path)
        nfiles=len(self.zf.namelist())
        if nfiles > 1:
            raise TooManyBinaryFilesException("## Error: Currently not supporting multiple binary files")
        # Load binary
        self.data_archive = self.zf.open(self.zf.namelist()[0])
        
    def _close_file(self):
        # Remove data_archive for good
        self.data_archive = None
        self.zf.close()

    def _load_header(self):
        # Open file
        self._open_file()
        
        # Load pre-set headers 
        data = self.data_archive.read(self.header_bytes)
        for idx, key in enumerate(self.header):
            f_i = idx*self.size_float32
            t_i = (idx+1)*self.size_float32
            self.header[key] = struct.unpack('f',data[f_i:t_i])[0]
            print(diff,idx,key,self.header[key])
        print("## Header bytes : %d bytes"%(self.header_bytes))
        h=int(self.header["image_height"])
        w=int(self.header["image_width"])
        c=int(self.header["max_colors"])
        wl=int(self.header["max_w_lbls"])
        el=int(self.header["max_e_lbls"])
        ne=int(self.header["nEntries"])
        self.num_batches_per_epoch = int(ne/self.batch_size)
        self.total_pixels = int(h*w)
        self.c_lbl_offset = int((c+wl)*self.size_float32)
        self.evt_bytes = (h*w*(c+wl)+el)*self.size_float32
        print("## Max. Events : %d "%(ne))
        print("## Event bytes : %d bytes"%(self.evt_bytes))
        print("## Total size : %d "%(ne*self.evt_bytes))
        self.all_evt_bytes = ne*self.evt_bytes
        
    def _decode(self,data):
        """
        A private function for decoding
        """
        # Clear list of data
        for idx, key in enumerate(self.a_image):
            self.a_image[key] = []
        for idx, key in enumerate(self.a_evt_lbls):
            self.a_evt_lbls[key] = -199
        # Load event level
        start_byte = 0
        if self.verbose >= 2:
            print("--------------------\n event level",len(self.a_evt_lbls),self.a_evt_lbls)
        for idx, key in enumerate(self.a_evt_lbls):
            f_i = start_byte + self.size_float32*idx
            t_i = start_byte + self.size_float32*(1+idx)
            val = struct.unpack('f',data[f_i:t_i])[0]
            self.a_evt_lbls[key]=val

            # Debug message
            if self.verbose >= 2:
                print("iBytes[%d,%d] %s %f"%(f_i,t_i,key,val))
        
        start_byte = len(self.a_evt_lbls)*self.size_float32
        # counter = 0
        # Loop over all pixels
        for ip in range(self.total_pixels):
            # Load image level
            l_ip = len(self.a_image)
            l_ip_b = l_ip*self.size_float32            
            offset = l_ip_b*ip
            for idx, key in enumerate(self.a_image):
                f_i = start_byte + idx*l_ip + offset
                t_i = start_byte + (idx+1)*l_ip + offset
                val = struct.unpack('f',data[f_i:t_i])[0]
                # This is poorly implemented...
                # charge
                if idx == 0 and val>0:
                    val=np.log(val)/self.norm[key]
                # drift time
                if idx == 1 and val!=-199:
                    val/=self.norm[key]
                if idx == 2 or idx ==3:
                    val+=1
                    
                self.a_image[key].append(val)
                if self.verbose >= 3 :
                    print("ip",ip,"offset",offset,key,"data[",f_i,t_i,"]=",val)
        # DEBUG
        if self.verbose >= 2:
            for idx, key in enumerate(self.a_image):
                length=len(self.a_image[key])
                print("image key: ",key," || pixels",length)
                w = self.header["image_width"]
                h = self.header["image_height"]
                if len(self.a_image[key])/w != h:
                    raise ImageSizeDontMatchException("## Error: Image size",len(self.a_image[key])/w,h)        


    def adjust_data(self,imgs, lbls, nClass=3):
        lbls[lbls<0] = 0
        lbls=np_utils.to_categorical(lbls, nClass)
        return imgs, lbls
                
    def load_event(self, n_load_events=32):
        """
        Return images and masks
        """
        if n_load_events < 0:
            n_load_events = self.header["nEntries"]
        if self.verbose > 1:
            print("## Warning, you are going to load",
                  n_load_events*self.evt_bytes,"kB of data")
        # Initialize batch data
        h=int(self.header["image_height"])
        w=int(self.header["image_width"])
        c=int(self.header["max_colors"])
        wl=int(self.header["max_w_lbls"])
        el=int(self.header["max_e_lbls"])
        batch_imgs = np.ndarray((n_load_events,h,w,c),dtype=float32)
        batch_imgs.fill(-199)
        batch_lbls = np.ndarray((n_load_events,h,w,wl),dtype=float32)
        batch_lbls.fill(-1)
        batch_evts_lbls = np.ndarray((n_load_events,el),dtype=float32)
        batch_evts_lbls.fill(-199)

        # Decode
        for iev in range(int(n_load_events)):
            # Events
            self._decode(self.data_archive.read(self.evt_bytes))
            # # Read image
            for idx, key in enumerate(self.a_image):
                if idx<c:
                    batch_imgs[iev][:,:,idx] = np.asarray(self.a_image[key],dtype).reshape(h,w)
                else:
                    ii = idx - c
                    batch_lbls[iev][:,:,ii] = np.asarray(self.a_image[key],dtype).reshape(h,w)
            # Read labels
            for idx, key in enumerate(self.a_evt_lbls):
                batch_evts_lbls[iev][idx] = np.asarray(self.a_evt_lbls[key],dtype)
        # end = time.time()
        # print("Time for loading batch images",end-start,"s")
        # print("Counter ",counter)
        s = batch_lbls.shape
        Y = batch_lbls[:,:,:,0].reshape(s[0],s[1],s[2],1)
        
        if self.verbose >= 3:
            # print(batch_imgs)
            # print(batch_lbls)
            print(batch_evts_lbls)
        return batch_imgs, Y, batch_evts_lbls
    #return batch_imgs, batch_lbls, batch_evts_lbls
        
