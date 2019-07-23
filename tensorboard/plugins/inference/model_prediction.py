import tensorflow as tf
import numpy as np
import math
from tensorboard.plugins.inference.model import Network
from tensorboard.plugins.inference.ReadTFRecord import read_and_decode
from tensorboard.plugins.inference.refresh_board import pred_refresh, fea_refresh
import matplotlib.pyplot as plt
import os

class Inference(object):
    
  def __init__(self,
               model_path = None,
               model_type = None):
    tf.reset_default_graph() 
    self.model_path = model_path
    self.model_type = model_type
    self.net = Network()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
    self.restore(self.model_path,self.model_type)
    self.ifDone = False
    print('load susess')
    
  def restore(self,model_dir,model_type_name):
    saver = tf.train.Saver()
    if(model_type_name == 'ckpt'):
      ckpt = tf.train.get_checkpoint_state(model_dir)
      print(ckpt.model_checkpoint_path)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(self.sess,ckpt.model_checkpoint_path)
      else:
        raise FileNotFoundError('dir error')

  def each_label_acc(self,label,pred):
    total_amount = [0]*10
    correct_amount = [0]*10
    for i in range(len(label)):
      total_amount[label[i]]+=1
      if(label[i]==pred[i]):
        correct_amount[label[i]]+=1        
    acc = np.true_divide(np.array(correct_amount),np.array(total_amount))
    return acc.tolist()

  def concact_features(self, conv_output):
    num_or_size_splits = int(math.sqrt(conv_output.shape[0]))
    img_out_list = []
    for j in range(num_or_size_splits):
      img_temp = conv_output[j*4]
      for i in range(num_or_size_splits-1):
        img_temp = np.concatenate((img_temp,conv_output[i+1+4*j]),axis=1)
      img_out_list.append(img_temp)
    img_out = img_out_list[0]
    for k in range(len(img_out_list)-1):
      img_out = np.concatenate((img_out,img_out_list[k+1]))
    return img_out

  def generate_tensor(self,conv):
    g = tf.Graph()     
    with tf.Session(graph=g) as sess:
      conv_transpose = sess.run(tf.transpose(conv, [3, 2, 1, 0]))
    with tf.Session(graph=g) as sess:
      conv_concact = sess.run(tf.transpose(self.concact_features(conv_transpose), [2, 1, 0])) 
    tensor_conv = tf.convert_to_tensor(conv_concact)[:, :, :, np.newaxis]
    return tensor_conv

  def predict(self,file_path,batchsize_s):
    batchsize = int(batchsize_s)
    filename_queue = tf.train.string_input_producer([file_path],num_epochs=None)
    img,label = read_and_decode(filename_queue,True,batchsize)
    #threads stop problem
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      test_x, test_label=sess.run([img,label])
    acc = self.sess.run(self.net.accuracy,feed_dict = {self.net.init_x:test_x,self.net.label:test_label})
    y = self.sess.run(self.net.y,feed_dict = {self.net.init_x:test_x})
    y_label = []
    y_pred = []
    for i in range(batchsize):
      y_label.append(np.argmax(test_label[i]))
      y_pred.append(np.argmax(y[i]))
    eachlabelacc = self.each_label_acc(y_label,y_pred)
    print(eachlabelacc)
    print("准确率: %.3f，共测试了%d张图片 " % (acc, len(test_label)))
    plt.bar(range(len(eachlabelacc)), eachlabelacc)
    try:
      plt.savefig("/root/tensorboard/tensorboard/plugins/inference/cache/cache_each_label_acc.png")
    except:
      print("failed to generate the figure")
    #os.system("python /root/tensorbsoard/tensorboard/plugins/inference/refresh_board.py --log_dir /tmp/mnist")
    pred_refresh("/tmp/mnist/prediction")
    self.ifDone = True

  def feature(self,file_path,batchsize_s):
    batchsize = int(batchsize_s)
    filename_queue = tf.train.string_input_producer([file_path],num_epochs=None)
    img,label = read_and_decode(filename_queue,True,batchsize)
    #threads stop problem
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      test_x, test_label=sess.run([img,label])
#    
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      conv1_16 = self.sess.run(self.net.hl.h_conv1, feed_dict={self.net.init_x:test_x})
      pool1_16 = self.sess.run(self.net.hl.h_pool1, feed_dict={self.net.init_x:test_x})
      conv2_32 = self.sess.run(self.net.hl.h_conv2, feed_dict={self.net.init_x:test_x})
      pool2_32 = self.sess.run(self.net.hl.h_pool2, feed_dict={self.net.init_x:test_x})
    tensor_conv1 = self.generate_tensor(conv1_16)
    tensor_pool1 = self.generate_tensor(pool1_16)
    tensor_conv2 = self.generate_tensor(conv2_32)
    tensor_pool2 = self.generate_tensor(pool2_32)
    fea_refresh("/tmp/mnist/feature", tensor_conv1,tensor_pool1,tensor_conv2,tensor_pool2)
    #fea_refresh("/tmp/mnist/feature", tensor_conv1)



'''
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      conv1_16 = self.sess.run(self.net.hl.h_conv1, feed_dict={self.net.init_x:test_x})     # [1, 28, 28 ,16] 
    with tf.Session(graph=g) as sess:
      conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 2, 1, 0]))
    with tf.Session(graph=g) as sess:
      conv1_concact = sess.run(tf.transpose(self.concact_features(conv1_transpose), [2, 1, 0])) 
    tensor_conv1 = tf.convert_to_tensor(conv1_concact)[:, :, :, np.newaxis]
    print(tensor_conv1.get_shape())
'''






