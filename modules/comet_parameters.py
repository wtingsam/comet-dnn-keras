import os
import numpy as np
from numpy import *
import sys
from collections import OrderedDict

class NoSuchPathException(Exception):
    pass
class StrangeParameterList(Exception):
    pass

class comet_parameters():

    def __init__(self, path, verbose=0):
        self.input_path = path
        self.verbose = verbose
        self.par_list = OrderedDict()
        self.load_path()
        
    def load_path(self):
        if self.input_path.lower().endswith(('.txt', '.par', '.dat')):
            print("%% Loaded: ",self.input_path)
            with open(self.input_path) as f:
                lines=f.readlines()
                for line in lines:
                    if self.verbose >0:
                        print(line)
                    data=line.split()
                    if len(data)==2 and '#' not in line:
                        self.par_list[data[0]] = data[1]
        else:
            raise NoSuchPathException("## Error: Extension must be [.txt,.par,.dat]",self.path)

    def ls(self):
        for i in self.par_list:
            print(i,self.par_list[i])

    def get_parameter(self,par_name):
        return self.par_list[par_name]
        
    def get_parlist(self):
        return self.par_list
