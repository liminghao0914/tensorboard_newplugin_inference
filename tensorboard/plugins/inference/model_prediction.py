import tensorflow as tf
import numpy as np
from tensorboard.plugins.inference.model import Network
from tensorboard.plugins.inference.ReadTFRecord import read_and_decode
from tensorboard.plugins.inference.refresh_board import run
import matplotlib.pyplot as plt
import os

class Predict(object):
    
    def __init__(self,model_path = None):
        tf.reset_default_graph() 
        self.model_path = model_path
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.restore(self.model_path)
        self.ifDone = False
        print('load susess')
    
    def restore(self,CKPT):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess,ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('未保存模型')

    def each_label_acc(self,label,pred):
        total_amount = [0]*10
        correct_amount = [0]*10
        for i in range(len(label)):
            total_amount[label[i]]+=1
            if(label[i]==pred[i]):
                correct_amount[label[i]]+=1        
        acc = np.true_divide(np.array(correct_amount),np.array(total_amount))
        return acc.tolist()

    def predict(self,file_path,batchsize_s):
        batchsize = int(batchsize_s)
        filename_queue = tf.train.string_input_producer([file_path],num_epochs=None)
        img,label = read_and_decode(filename_queue,True,batchsize)
        #threads stop problem
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            test_x, test_label=sess.run([img,label])
        acc = self.sess.run(self.net.accuracy,feed_dict = {self.net.x:test_x,self.net.label:test_label})
        y = self.sess.run(self.net.y,feed_dict = {self.net.x:test_x})
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
            print("dsd")
        #os.system("python /root/tensorboard/tensorboard/plugins/inference/refresh_board.py --log_dir /tmp/mnist")
        run("/tmp/mnist")
        self.ifDone = True
