from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
model_path = '/root/Desktop/ckpt'
data_path = '/root/Desktop/data/mnist/mnist.tfrecords'
'''
import tensorflow as tf
from tensorboard.plugins.inference.model_prediction import Predict

class train:
  def __init__(
      self,
      model_path = None,
      data_path = None,
      batchsize = None,
      model_type = None):
    self.model_path = model_path
    self.data_path = data_path
    self.batchsize = batchsize
    self.model_type = model_type

  def start(self):
    if self.model_type=='c':
      self.classification(self.model_path,self.data_path)
    if self.model_type=='r':
      self.regression(self.model_path,self.data_path)

  def classification(self):
    model = Predict(self.model_path)
    model.predict(self.data_path,self.batchsize)
    #try:
    #  model.predict(self.data_path,self.batchsize)
   # except:
    #  return "fail!"
   # return "success!"
  def regression(self,mp,dp):
    return "regression is offline"
