from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
model_path = '/root/Desktop/ckpt'
data_path = '/root/Desktop/data/mnist/mnist.tfrecords'
'''
import tensorflow as tf
from tensorboard.plugins.inference.model_prediction import Inference

class train:
  def __init__(
      self,
      model_path = None,
      data_path = None,
      batch_size = None,
      model_type = None):
    self.model_path = model_path
    self.data_path = data_path
    self.batch_size = batch_size
    self.model_type = model_type

  def start(self):
    self.classification(self.model_path,self.model_type,self.data_path,self.batch_size)

  def classification(self,modelpath,modeltype,datapath,batchsize):
    model = Inference(modelpath,modeltype)
    #model.predict(datapath,batchsize)
    model.feature(datapath,batchsize)

  def regression(self,mp,dp):
    return "regression is offline"
