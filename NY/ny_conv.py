import time
import os
import argparse
import sys
import numpy as np
import json

from pygsp import graphs

import tensorflow as tf

from graph_utils.laplacian import initialize_laplacian_tensor
from state_prediction.models import deep_fir_tv_conv, fir_tv_fc_fn, deep_cheb_fc_fn, cheb_fc_fn, deep_sep_fir_fc_fn, fc_fn,deep_fir_tv_full_conv,deep_fir_tv_zero
#from synthetic_data.data_generation import generate_wave_samples
from uberGraph import gen_data, split_seq
#from graph_utils.coarsening import coarsen, perm_data, keep_pooling_laplacians


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
    def __init__(self,graph_data,randList,transform=None,train_num=12,test_num = 1):
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
        targets = self.graph_data.loc[self.randList[index]][self.train_num:self.train_num+self.test_num]

        
        #x = torch.tensor(x.values)
        x = x.values
        targets = targets.values
        #targets = torch.tensor(targets.values)

        if self.transform:
            x = self.transform.transform(x)
            targets = self.transform.transform(targets)


        #permutations ?
            
        x = np.transpose(x, (1, 0))
        x = np.expand_dims(x,2)

        targets = np.transpose(targets, (1,0))

        #targets = np.squeeze(targets,0) 

        sample = {'x': x,'targets':targets}

        return sample


def _fill_feed_dict(sample, x, y, dropout,phase, is_training):

    inputs = sample['x']
    targets = sample['targets'] if is_training else sample['targets']*0

    #print(inputs.shape)

    feed_dict = {x: inputs, y: targets, dropout: 0.5 if is_training else 1, phase: is_training}

    return feed_dict


def setup(L):

    # Create data placeholders
    num_vertices = int(L.shape[0])
    print(L.shape[0])

    x = tf.placeholder(tf.float32, [None, num_vertices, FLAGS.num_frames,1],name="x")
    y = tf.placeholder(tf.float32, [None, num_vertices, FLAGS.num_frames_test],name="y")



     # Initialize model
    if FLAGS.model_type == "deep_fir":
        print("Training deep FIR-TV model...")
        prediction, phase = deep_fir_tv_conv(x=x,
                                          L=L,
                                          output_units=num_vertices*FLAGS.num_frames_test,
                                          time_filter_orders=FLAGS.time_filter_orders,
                                          vertex_filter_orders=FLAGS.vertex_filter_orders,
                                          num_filters=FLAGS.num_filters,
                                          time_poolings=FLAGS.time_poolings,
                                          vertex_poolings=FLAGS.vertex_poolings,
                                          shot_noise=FLAGS.shot_noise)
        dropout = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
        print('\n\nModel initialized')

    elif FLAGS.model_type == "deep_fir_conv":
        print("Training deep FIR-TV model...")
        prediction, phase = deep_fir_tv_full_conv(x=x,
                                          L=L,
                                          output_units=FLAGS.num_frames_test,
                                          time_filter_orders=FLAGS.time_filter_orders,
                                          vertex_filter_orders=FLAGS.vertex_filter_orders,
                                          num_filters=FLAGS.num_filters,
                                          time_poolings=FLAGS.time_poolings,
                                          vertex_poolings=FLAGS.vertex_poolings,
                                          shot_noise=FLAGS.shot_noise)
        dropout = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
        print('\n\nModel initialized')

    elif FLAGS.model_type == "deep_cheb":
        print("Training deep Chebyshev time invariant model...")
        xt = tf.transpose(x, perm=[0, 1, 3, 2])
        prediction, phase = deep_cheb_fc_fn(x=xt,
                                        L=L,
                                        output_units=num_vertices,
                                        vertex_filter_orders=FLAGS.vertex_filter_orders,
                                        num_filters=FLAGS.num_filters,
                                        vertex_poolings=FLAGS.vertex_poolings,
                                        shot_noise=FLAGS.shot_noise)
        dropout = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
    elif FLAGS.model_type == "deep_sep":
        print("Training deep separable FIR model...")
        prediction, phase = deep_sep_fir_fc_fn(x=x,
                                           L=L,
                                           output_units=num_vertices,
                                           time_filter_orders=FLAGS.time_filter_orders,
                                           vertex_filter_orders=FLAGS.vertex_filter_orders,
                                           num_filters=FLAGS.num_filters,
                                           time_poolings=FLAGS.time_poolings,
                                           vertex_poolings=FLAGS.vertex_poolings)
        dropout = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
    elif FLAGS.model_type == "fc":
        print("Training linear classifier model...")
        prediction = fc_fn(x, num_vertices)
        dropout = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
        phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    elif FLAGS.model_type == "fir_tv":
        print("Training fir-tv model...")
        prediction, dropout = fir_tv_fc_fn(x, L, num_vertices, FLAGS.time_filter_order, FLAGS.filter_order, FLAGS.num_filter)
        phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    else:
        raise ValueError("model_type not valid.")


    # Initialize model
    #prediction, dropout = fir_tv_fc_fn(x, L, num_vertices, FLAGS.time_filter_order, FLAGS.filter_order, FLAGS.num_filters)
    #prediction, dropout = cheb_fc_fn(x, L, num_vertices, FLAGS.filter_order, FLAGS.num_filters)
    #prediction, dropout = fc_fn(x, num_vertices)

    # Define loss
    with tf.name_scope("loss"):
        mse = tf.losses.mean_squared_error(y, prediction)
        loss = tf.reduce_mean(mse)
        tf.summary.scalar('mse', loss)

    #check there is not something wrong with the metrics    
    with tf.name_scope("metric"):
        metric, metric_opt = tf.metrics.root_mean_squared_error(y, prediction)

    return x,y,prediction,phase,dropout,metric,metric_opt,loss



def run_training(train_source, L, test_source):
    """Performs training and evaluation."""
    
    x,y,prediction,phase,dropout,metric,metric_opt,loss = setup(L)
    # Select optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    opt_train = optimizer.minimize(loss, global_step=global_step)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Run session
    with tf.compat.v1.Session() as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #MAX_STEPS = FLAGS.num_epochs * FLAGS.num_train // FLAGS.batch_size

        #test_feed_dict = {x: np.expand_dims(test_data[:, :, :-1],3), y_: test_data[:, :, -1], dropout: 1}

        # Start training loop
        epoch_count = 0
        for e in range(FLAGS.num_epochs):

            start_time = time.time()

            for i,sample in enumerate(train_source):

                feed_dict = _fill_feed_dict(sample, x, y, dropout, phase, is_training = True)
                # Perform one training iteration
                sess.run([opt_train, metric_opt], feed_dict=feed_dict)

                    # Write the summaries and print an overview fairly often.
                if i % 10 == 0:
                    # Print status to stdout.
                    duration = time.time() - start_time
                    print('Step %d: mse = %.2f (%.3f sec)' % (i, metric.eval()**2, duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

            

            print("Epoch %d" % e)
            print("--------------------")

            

            # Save a checkpoint and evaluate the model periodically.
            if e%1 == 0:

                checkpoint_file = os.path.join(FLAGS.log_dir, 'model')
                saver.save(sess, checkpoint_file, global_step=e)

                #for sample in test_source:
                #
                #   test_feed_dict = _fill_feed_dict(sample, x, y_, dropout,is_training = False)
                #    sess.run(metric_opt, feed_dict=test_feed_dict)

                _eval_metric(sess, dropout, phase, x, y, test_source,metric, metric_opt)

                print("--------------------")
                print('Test mse = %.2f' % metric.eval()**2)

                print("====================")

            # Save a checkpoint and evaluate the model periodically.
            if  e == FLAGS.num_epochs-1:

                checkpoint_file = os.path.join(FLAGS.log_dir, 'model')
                saver.save(sess, checkpoint_file, global_step=e)

                #for sample in test_source:
                #
                #   test_feed_dict = _fill_feed_dict(sample, x, y_, dropout,is_training = False)
                #    sess.run(metric_opt, feed_dict=test_feed_dict)

                _eval_metric(sess, dropout, phase, x, y, test_source,metric, metric_opt,prediction,last=True)

                print("--------------------")
                print('Test mse = %.2f' % metric.eval()**2)
                print("====================")



def _eval_metric(sess, dropout, phase, x, y, test_source, metric,metric_opt, predictions=None, last=False):

    accuracies = []
 
    #note: only works if all batches have the same size (even the last one)
    for sample in test_source:

        test_feed_dict = _fill_feed_dict(sample, x, y, dropout, phase, is_training = False)
        sess.run(metric_opt, feed_dict=test_feed_dict)
        accuracies.append(metric.eval())


    if last:

        results = []
        truths = []

        assert predictions is not None


        for sample in test_source:

            test_feed_dict = _fill_feed_dict(sample, x, y, dropout, phase, is_training = False)
            #predictions = y.eval(feed_dict ={x: sample['x'], dropout: 1})
            preds =sess.run(predictions,feed_dict=test_feed_dict)

            #predictions = sess.run([inference_step], feed_dict={x: sample['x']})

            #print(sample['x'].shape)


            real = np.concatenate((sample['x'].squeeze(3),sample['targets']),axis=2)


            if len(results) == 0:
                results = preds
                truths = real 
            else:
                results = np.concatenate((results,preds),axis=0)
                truths  = np.concatenate((truths,real),axis=0)


        print('Test dataset evaluated')
        np.save('results',results)
        np.save('truths',truths)

    return np.mean(accuracies)

def _last_checkpoint(log_dir):
    checkpoints = []
    if not os.path.exists(log_dir):
        raise IOError("No such file or directory:", log_dir)

    for file in os.listdir(log_dir):
        if ".meta" not in file:
            continue
        else:
            checkpoints.append(int(file.split("-")[1].split(".")[0]))

    return max(checkpoints)


def _last_exp(log_dir):
    exp_numbers = []
    if not os.path.exists(log_dir):
        return 0
    for file in os.listdir(log_dir):
        if "exp" not in file:
            continue
        else:
            exp_numbers.append(int(file.split("_")[1]))
    return max(exp_numbers) if len(exp_numbers) > 0 else 0


def run_eval(test_source):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(FLAGS.log_dir, "model-" + str(_last_checkpoint(FLAGS.log_dir)) + ".meta"))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))
        graph = tf.get_default_graph()

        # Get inputs
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        fc = graph.get_tensor_by_name("fc_layer/final:0")




        #print([n.name for n in tf.get_default_graph().as_graph_def().node])

        dropout = graph.get_tensor_by_name("keep_prob:0")
        phase = graph.get_tensor_by_name("phase:0")

        results = []
        truths = []


        for sample in test_source:

            test_feed_dict = _fill_feed_dict(sample, x, y, dropout, phase, is_training = False)
            #predictions = y.eval(feed_dict ={x: sample['x'], dropout: 1})
            predictions =sess.run(fc,feed_dict=test_feed_dict)

            #predictions = sess.run([inference_step], feed_dict={x: sample['x']})

            #print(sample['x'].shape)


            real = np.concatenate((sample['x'].squeeze(3),sample['targets']),axis=2)


            if len(results) == 0:
                results = predictions
                truths = real 
            else:
                results = np.concatenate((results,predictions),axis=0)
                truths  = np.concatenate((truths,real),axis=0)


        print('Test dataset evaluated')

        print('Graph consistency test:')
        #print('Shape {}'.format(truths.shape))
        print(truths.sum(axis=(0,2)).argmax())

        return results, truths



        #todo:  run loss function
        # Get output

        #print("Evaluation accuracy: %.2f" % _eval_metric(sess, dropout, phase, x, y, test_source,metric, metric_opt))

        #for idx, v in enumerate([v for v in tf.trainable_variables() if "conv" in v.name]):
        #    plot_tf_fir_filter(sess, v, os.path.join(FLAGS.log_dir, "conv_%d" % idx))

def _number_of_trainable_params():
    return np.sum([np.product(x.shape) for x in tf.trainable_variables()])

def main(_):


    # Initialize tempdir
    if FLAGS.action == "eval" and FLAGS.read_dir is not None:
        FLAGS.log_dir = FLAGS.read_dir
    else:
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_type)
        exp_n = _last_exp(FLAGS.log_dir) + 1 if FLAGS.action == "train" else _last_exp(FLAGS.log_dir)
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, "exp_" + str(exp_n))

    print(FLAGS.log_dir)

    # Make graph and fill with data
    x = gen_data(interval=FLAGS.interval,graphDataPath = FLAGS.graphPath)
    x = (x-x.mean(axis=0))/x.std(axis=0)

    print(x.shape)
    
    #Normalize them
    #scaler = StandardScaler()
    #scaler.fit(x)

    #Data loading
    seqs = np.array(list(split_seq(x.index.values,FLAGS.num_frames+FLAGS.num_frames_test)))

    np.random.seed()
    np.random.shuffle(seqs)
    print('{} total samples'.format(len(seqs)))

    split = int(len(seqs)*FLAGS.train_ratio)


    #x.values = scaler.transform(x)
    

    train_ds = DataLoader(x,seqs[:split],train_num=FLAGS.num_frames ,test_num = FLAGS.num_frames_test,transform = None)
    train_dl = data.DataLoader(train_ds,batch_size = FLAGS.batch_size,shuffle=True,num_workers = 4)

    test_ds = DataLoader(x,seqs[split:], train_num=FLAGS.num_frames, test_num = FLAGS.num_frames_test,transform = None)
    test_dl = data.DataLoader(test_ds,batch_size = FLAGS.batch_size,shuffle=False,num_workers = 4)


    if FLAGS.action == "train":
        if tf.io.gfile.exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.io.gfile.makedirs(FLAGS.log_dir)



        sparsed = pd.read_csv('NY/data/'+FLAGS.graphPath,index_col=0)
        W = sparsed.values
        np.fill_diagonal(W,0)
        #W = sparsed.values - np.identity(len(sparsed))
        G = graphs.Graph(W)

        #G = graphs.Community(len(sparsed), seed=0)

        G.compute_laplacian("normalized")
        L = initialize_laplacian_tensor(G.W)
        W = (G.W).astype(np.float32)

        #print(W)

        print('{} nodes, {} edges'.format(G.N, G.Ne))

        #Log to file
        params = vars(FLAGS)
        with open(os.path.join(FLAGS.log_dir, "params.json"), "w") as f:
            json.dump(params, f)

        # Run training and evaluation loop
        print("Training model...")
        run_training(train_dl, L, test_dl)


        #run eval

        results, truths = run_eval(test_dl)

        np.save('results',results)
        np.save('truths',truths)
        
    elif FLAGS.action == "eval":
    
        #x,y,prediction,phase,dropout,metric,metric_opt,loss = setup(L)

        results, truths = run_eval(test_dl)

        np.save('results',results)
        np.save('truths',truths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="deep_fir",
        help="Model type"
    )
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
        default=10,
        help='Minibatch size in samples.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(TEMPDIR, "tensorflow/tv_graph_cnn/logs/"),
        help='Logging directory'
    )

    parser.add_argument(
        '--read_dir',
        type=str,
        default=os.path.join(TEMPDIR, "tensorflow/tv_graph_cnn/logs/deep_fir_conv/exp_38/"),
        help='Reading directory'
    )

    parser.add_argument(
        '--num_frames',
        type=int,
        default=12,
        help='Number of temporal frames.'
    )

    parser.add_argument(
        '--num_frames_test',
        type=int,
        default=7,
        help='Number of temporal frames.'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Aggregation interval.'
    )


    parser.add_argument(
        '--vertex_filter_orders',
        type=int,
        default=[1, 1, 1],
        nargs="+",
        help='Convolution vertex order.'
    )
    parser.add_argument(
        '--time_filter_orders',
        type=int,
        nargs="+",
        default=[3, 3, 3],
        help='Convolution time order.'
    )
    parser.add_argument(
        '--num_filters',
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help='Number of parallel convolutional filters.'
    )
    parser.add_argument(
        '--time_poolings',
        type=int,
        nargs="+",
        default=[2,2,1],
        help='Time pooling sizes.'
    )
    parser.add_argument(
        "--vertex_poolings",
        type=int,
        nargs="+",
        default=[1,1,1],
        help="Vertex pooling sizes"
    )

    parser.add_argument(
        '--graphPath',
        type=str,
        default='sparsedDist.csv',
        help='Graph data path'
    )
    parser.add_argument(
        "--shot_noise",
        type=float,
        default=1,
        help="Probability of missing entry"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        help="Action to perform on the model"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
