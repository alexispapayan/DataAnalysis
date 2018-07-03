
# coding: utf-8

# In[ ]:

import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



B_INIT = -0.2 # use a bad bias constant initializer


class Net(nn.Module):
    
    def __init__(self,in_features_dimension,out_features_dimension,nb_of_hidden_layers,nb_of_hidden_nodes,
                 batch_normalization=False):
        
        super(Net,self).__init__()
        
        self.nb_hidden_layers=nb_of_hidden_layers
        self.do_bn=batch_normalization
        self.fcs=[]
        self.bns=[]
        self.bn_input=nn.BatchNorm1d(in_features_dimension,momentum=0.5) #for input data
        
        for i in range(nb_of_hidden_layers):                              # build hidden layers and BN layers
            
            input_size=in_features_dimension if i==0 else nb_of_hidden_nodes
            fc=nn.Linear(input_size,nb_of_hidden_nodes)
            setattr(self, 'fc%i' % i, fc)       # IMPORTANT set layer to the Module
            self._set_init(fc)                  # parameters initialization
            self.fcs.append(fc)
            
            if self.do_bn:
                bn = nn.BatchNorm1d(nb_of_hidden_nodes, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)                         # IMPORTANT set layer to the Module
                self.bns.append(bn)
    
            self.predict = nn.Linear(nb_of_hidden_nodes,out_features_dimension)         # output layer
            self._set_init(self.predict)                                              # parameters initialization
    
    
    def _set_init(self, layer):
            init.normal(layer.weight, mean=0., std=.1)
            init.constant(layer.bias, B_INIT)

    
    
    
    def forward(self, x):
        ACTIVATION=F.relu
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)     # input batch normalization
        layer_input = [x]
        for i in range(self.nb_hidden_layers):
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn: x = self.bns[i](x)   # batch normalization
            x = ACTIVATION(x)
            layer_input.append(x)
        out = self.predict(x)
        return out

