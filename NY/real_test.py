import time
import os
import argparse
import sys
import numpy as np

from pygsp import graphs

import tensorflow as tf

from graph_utils.laplacian import initialize_laplacian_tensor
from state_prediction.models import fir_tv_fc_fn, cheb_fc_fn, fc_fn
#from synthetic_data.data_generation import generate_wave_samples
from uberGraph import gen_data, split_seq


import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd



FLAGS = None
FILEDIR = os.path.dirname(os.path.realpath(__file__))
TEMPDIR = os.path.realpath(os.path.join(FILEDIR, "../experiments"))


class DataLoader(data.Dataset):
    def __init__(self,graph_data,randList,transform=None,train_num=12,test_num = 12):
        self.graph_data = graph_data
        self.length     = len(randList)
        self.randList   = randList
        self.transform  = transform
        self.train_num  = train_num
        self.test_num   = test_num
    
    def __len__(self):
        return self.length
        
    def __getitem__(self,index):

        x = self.graph_data.loc[self.randList[index]][:self.train_num]
        targets = self.graph_data.loc[self.randList[index]][self.test_num:self.test_num+1]

        
        #x = torch.tensor(x.values)
        x = x.values
        targets = targets.values
        #targets = torch.tensor(targets.values)

        if self.transform:
            x = self.transform.transform(x)
            targets = self.transform.transform(targets)
            
        #permutations ?
        #print(x.shape)
            
        x = np.transpose(x, (1, 0))

        x = np.expand_dims(x,2)
        targets = np.squeeze(targets,0) 

        sample = {'x': x,'targets':targets}

        return sample


def _fill_feed_dict(sample, x, y, dropout):

    inputs = sample['x']
    targets = sample['targets']

    #print(inputs.shape)

    feed_dict = {x: inputs, y: targets, dropout: 0.5}

    return feed_dict

def _fill_feed_dict_test(sample, x, y, dropout):

    inputs = sample['x']
    targets = sample['targets']

    #print(inputs.shape)

    feed_dict = {x: inputs, y: targets, dropout: 1}

    return feed_dict


def run_training(train_source, L, test_source):
    """Performs training and evaluation."""

    # Create data placeholders
    num_vertices = 250

    x = tf.placeholder(tf.float32, [None, num_vertices, FLAGS.num_frames,1])
    y_ = tf.placeholder(tf.float32, [None, num_vertices])
    # Initialize model
    prediction, dropout = fir_tv_fc_fn(x, L, num_vertices, FLAGS.time_filter_order, FLAGS.filter_order, FLAGS.num_filters)
    #prediction, dropout = cheb_fc_fn(x, L, num_vertices, FLAGS.filter_order, FLAGS.num_filters)
    #prediction, dropout = fc_fn(x, num_vertices)

    # Define loss
    with tf.name_scope("loss"):
        mse = tf.losses.mean_squared_error(y_, prediction)
        loss = tf.reduce_mean(mse)
        tf.summary.scalar('mse', loss)

    #check there is not something wrong with the metrics    
    with tf.name_scope("metric"):
        metric, metric_opt = tf.metrics.root_mean_squared_error(y_, prediction)

    # Select optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    opt_train = optimizer.minimize(loss, global_step=global_step)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Run session
    with tf.Session() as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

#        MAX_STEPS = FLAGS.num_epochs * FLAGS.num_train // FLAGS.batch_size

#        test_feed_dict = {x: np.expand_dims(test_data[:, :, :-1],3), y_: test_data[:, :, -1], dropout: 1}

        # Start training loop
        epoch_count = 0
        for e in range(FLAGS.num_epochs):

            start_time = time.time()

            for i,sample in enumerate(train_source):

                feed_dict = _fill_feed_dict(sample, x, y_, dropout)
                # Perform one training iteration
                sess.run([opt_train, metric_opt], feed_dict=feed_dict)

                    # Write the summaries and print an overview fairly often.
                if i % 20 == 0:
                    # Print status to stdout.
                    duration = time.time() - start_time
                    print('Step %d: rmse = %.2f (%.3f sec)' % (i, metric.eval(), duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

            

            print("Epoch %d" % e)
            print("--------------------")

            

            # Save a checkpoint and evaluate the model periodically.
            if e%1 == 0 or e == FLAGS.num_epochs-1:

                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=e)

                accuracy = 0
                counts = 0

                for sample in test_source:

                    test_feed_dict = _fill_feed_dict_test(sample, x, y_, dropout)
                    sess.run(metric_opt, feed_dict=test_feed_dict)
                    accuracy += metric.eval()
                    counts   += 10



                print("--------------------")
                print('Test rmse = %.2f' % accuracy/counts)
                print("====================")


def main(_):
    # Initialize tempdir
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)


    x = gen_data(interval=5,future=7,graphDataPath = FLAGS.graphPath)


    sparsed = pd.read_csv('NY/data/'+FLAGS.graphPath,index_col=0)
    W = sparsed.values - np.identity(len(sparsed))
    G = graphs.Graph(W)

    G.compute_laplacian("normalized")
    L = initialize_laplacian_tensor(G.W)
    W = (G.W).astype(np.float32)

    print('{} nodes, {} edges'.format(G.N, G.Ne))


    scaler = StandardScaler()
    scaler.fit(x)
    

    seqs = np.array(list(split_seq(x.index.values,19)))
    np.random.shuffle(seqs)
    print('{} total samples'.format(len(seqs)))

    split = int(len(seqs)*FLAGS.train_ratio)

    train_ds = DataLoader(x,seqs[:split],transform = scaler)
    train_dl = data.DataLoader(train_ds,batch_size = 10,shuffle=True,num_workers = 4)

    test_ds = DataLoader(x,seqs[split:],transform = scaler)
    test_dl = data.DataLoader(test_ds,batch_size = 10,shuffle=True,num_workers = 4)

    # Run training and evaluation loop

    run_training(train_dl, L, test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Number of training samples.'

    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Minibatch size in samples.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(TEMPDIR, "tensorflow/tv_graph_cnn/logs/cheb_net"),
        help='Logging directory'
    )

    parser.add_argument(
        '--num_frames',
        type=int,
        default=12,
        help='Number of temporal frames.'
    )
    parser.add_argument(
        '--filter_order',
        type=int,
        default=5,
        help='Convolution vertex order.'
    )
    parser.add_argument(
        '--time_filter_order',
        type=int,
        default=5,
        help='Convolution time order.'
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        default=32,
        help='Number of parallel convolutional filters.'
    )
    parser.add_argument(
        '--graphPath',
        type=str,
        default='sparsedDist.csv',
        help='Graph data path'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) 