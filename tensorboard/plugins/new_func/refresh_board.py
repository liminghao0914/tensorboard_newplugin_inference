from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

FLAGS = None

def refresh_board():
  file = open('/root/tensorboard/tensorboard/plugins/new_func/cache/cache_each_label_acc.png', 'rb')
  data = file.read()
  file.close()
  print("load the file")
  image = tf.image.decode_png(data, channels=4)
  image = tf.expand_dims(image, 0)

  sess = tf.Session()
  writer = tf.summary.FileWriter(FLAGS.log_dir)
  summary_op = tf.summary.image("each_label_acc", image)

  summary = sess.run(summary_op)
  writer.add_summary(summary)

  writer.close()
  sess.close()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  with tf.Graph().as_default():
    refresh_board()
    print("done!")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
