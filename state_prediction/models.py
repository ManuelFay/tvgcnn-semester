import tensorflow as tf

from tv_graph_cnn.layers import chebyshev_convolution, fir_tv_filtering_einsum

from tv_graph_cnn.layers import fir_tv_filtering_einsum, chebyshev_convolution, jtv_chebyshev_convolution, \
    fir_tv_filtering_conv1d, sep_fir_filtering,fir_tv_filtering_conv1d_time
#does not exist the matmul ? 


def _weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.5, dtype=tf.float32)
    return tf.Variable(initial)


def _bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.zeros(shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



def _batch_normalization(input, is_training=True, scope=None):
    # Note: is_training is tf.compat.v1.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input, is_training=True, decay=0.8,
                                                        center=False, updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input, is_training=False, decay=0.8,
                                                        updates_collections=None, center=False, scope=scope,
                                                        reuse=True))


def deep_fir_tv_conv(x, L, output_units, time_filter_orders, vertex_filter_orders, num_filters, time_poolings,vertex_poolings, shot_noise=1):
    assert len(time_filter_orders) == len(vertex_filter_orders) == len(num_filters) == len(time_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(time_filter_orders)
    phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    # Convolutional layers
    vpool = tf.nn.dropout(x, shot_noise)
    # vpool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _fir_tv_layer(vpool, L, time_filter_orders[n], vertex_filter_orders[n], num_filters[n])
            conv = _batch_normalization(conv, is_training=phase, scope="conv%d" % n)
            conv = tf.nn.relu(conv, name="conv%d" % n)

        with tf.name_scope("subsampling%d" % n):
            tpool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=(1, time_poolings[n]),
                padding="same",
                strides=(1, time_poolings[n])
            )
        with tf.name_scope("vertex_pooling%d" % n):
            if vertex_poolings[n] > 1:
                vpool = tf.layers.max_pooling2d(
                    inputs=tpool,
                    pool_size=(vertex_poolings[n], 1),
                    padding="same",
                    strides=(vertex_poolings[n], 1)
                )
            else:
                vpool = tpool


    # Last fully connected layer
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(vpool)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True,
        )
        fc = tf.identity(fc, name="fc")


    with tf.name_scope("fc_layer"):
        fc = tf.compat.v1.reshape(fc,[-1,int(L.shape[0]), int(output_units/int(L.shape[0]))], name='final')

    return fc, phase


def deep_fir_tv_full_conv(x, L, output_units, time_filter_orders, vertex_filter_orders, num_filters, time_poolings,vertex_poolings, shot_noise=1):
    assert len(time_filter_orders) == len(vertex_filter_orders) == len(num_filters) == len(time_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(time_filter_orders)
    phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    # Convolutional layers
    tpool = tf.nn.dropout(x, shot_noise)
    # vpool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _fir_tv_layer(tpool, L, time_filter_orders[n], vertex_filter_orders[n], num_filters[n])
            conv = _batch_normalization(conv, is_training=phase, scope="conv%d" % n)
            conv = tf.nn.relu(conv, name="conv%d" % n)

        with tf.name_scope("subsampling%d" % n):
            tpool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=(1, time_poolings[n]),
                padding="valid",   #"same"
                strides=(1, time_poolings[n])
            )


    print(tpool.get_shape())

    B, N, T, C = tpool.get_shape()  # B: number of samples in batch, N: number of nodes, T: temporal length, C: channels


    with tf.name_scope("weights"):
            ht = _weight_variable([int(T), 1])
            hf = _weight_variable([int(C)-output_units+1,1])

            print(ht.get_shape())



    #K, F = hv.get_shape()  # K: Length vertex filter,  F: Number of filters
    M, F = ht.get_shape()  # M: Length time filter, F: Number of filters
    #C, F = hmimo.get_shape()  # M: Length time filter, F: Number of filters

    tpool = tf.transpose(tpool, perm=[0, 1, 3, 2])  # BxNxCxT
    tpool = tf.expand_dims(tpool, axis=4)  # BxNxCxTx1
    tpool = tf.reshape(tpool, shape=[-1, T, 1])  # BNCxTx1


    with tf.name_scope("final_time_conv"):
        x_convt = tf.nn.conv1d(tpool, tf.expand_dims(ht, axis=1), stride=1, padding="VALID", data_format="NHWC")  # BNCxTxF

        x_convt = tf.reshape(x_convt, shape=[-1, N, C, 1, F])  # BxNxCxTxF
        x_convt = tf.transpose(x_convt, perm=[0, 1, 3, 2, 4])


    #x_convt = tf.transpose(x_convt, perm=[0, 1, 3, 2])  # BxNxTxCxF
    x_convt = tf.reshape(x_convt,shape=[-1,C,1])


    with tf.name_scope("final_filter_conv"):
        x_convt = tf.nn.conv1d(x_convt, tf.expand_dims(hf, axis=1), stride=1, padding="VALID", data_format="NHWC")  
        print(x_convt.get_shape())


        x_convt = tf.reshape(x_convt, shape=[-1, N, 1, output_units, 1])  
        #x_convt = tf.transpose(x_convt, perm=[0, 1, 3, 2, 4])




    with tf.name_scope("fc_layer"):
        fc = tf.compat.v1.reshape(x_convt,[-1,int(L.shape[0]), output_units], name='final')

    return fc, phase



def fc_fn(x, output_units):
    """
    probably wrong description    
    
    Neural network consisting on 1 convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(x)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

    return fc, keep_prob


def fir_tv_fc_fn(x, L, output_units, time_filter_order, vertex_filter_order, num_filters):
    """
    Neural network consisting on 1 FIR-TV layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param time_filter_order: order of time convolution
    :param vertex_filter_order: order of vertex convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """

    #_, _, _, num_channels = x.get_shape()
    num_channels = 1

    print(num_channels)

    with tf.name_scope("FIR-TV"):
        with tf.name_scope("weights"):
            hfir = _weight_variable([vertex_filter_order, time_filter_order, num_channels, num_filters])
            _variable_summaries(hfir)
        with tf.name_scope("biases"):
            bfir = _bias_variable([num_filters])
            _variable_summaries(bfir)

        graph_conv = fir_tv_filtering_einsum(x, L, hfir, bfir, "chebyshev")
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(graph_conv, keep_prob=keep_prob)

    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(dropout)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True,
        )
    return fc, keep_prob


def cheb_fc_fn(x, L, output_units, filter_order, num_filters):
    """
    Neural network consisting on 1 convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """

    num_channels = 1

    print(num_channels)


    with tf.name_scope("chebyshev_conv"):
        with tf.name_scope("weights"):
            Wcheb = _weight_variable([filter_order, num_channels, num_filters])
            variable_summaries(Wcheb)
        with tf.name_scope("biases"):
            bcheb = _bias_variable([num_filters])
            variable_summaries(bcheb)
        graph_conv = chebyshev_convolution(x, L, Wcheb, bcheb)
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(graph_conv, keep_prob=keep_prob)
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(dropout)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

    return fc, keep_prob

def deep_cheb_fc_fn(x, L, output_units, vertex_filter_orders, num_filters, vertex_poolings, shot_noise=1):
    assert len(vertex_filter_orders) == len(num_filters) == len(vertex_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(vertex_filter_orders)
    phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    # Convolutional layers
    vpool = tf.nn.dropout(x, shot_noise)
    # vpool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _cheb_conv_layer(vpool, L, vertex_filter_orders[n], num_filters[n])
            conv = _batch_normalization(conv, is_training=phase, scope="conv%d" % n)
            conv = tf.nn.relu(conv, name="conv%d" % n)

        with tf.name_scope("vertex_pooling%d" % n):
            if vertex_poolings[n] > 1:
                vpool = tf.layers.max_pooling2d(
                    inputs=conv,
                    pool_size=(vertex_poolings[n], 1),
                    padding="same",
                    strides=(vertex_poolings[n], 1)
                )
            else:
                vpool = conv

    # Last fully connected layer
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(vpool)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

        fc = tf.identity(fc, "fc")
    return fc, phase

def deep_sep_fir_fc_fn(x, L, output_units, time_filter_orders, vertex_filter_orders, num_filters, time_poolings,vertex_poolings):
    assert len(time_filter_orders) == len(vertex_filter_orders) == len(num_filters) == len(time_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(time_filter_orders)
    phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    # Convolutional layers
    vpool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _sep_fir_layer(vpool, L[n], time_filter_orders[n], vertex_filter_orders[n], num_filters[n])
            conv = _batch_normalization(conv, is_training=phase, scope="conv%d" % n)
            conv = tf.nn.relu(conv, name="conv%d" % n)

        with tf.name_scope("subsampling%d" % n):
            tpool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=(1, time_poolings[n]),
                padding="same",
                strides=(1, time_poolings[n])
            )
        with tf.name_scope("vertex_pooling%d" % n):
            if vertex_poolings[n] > 1:
                vpool = tf.layers.max_pooling2d(
                    inputs=tpool,
                    pool_size=(vertex_poolings[n], 1),
                    padding="same",
                    strides=(vertex_poolings[n], 1)
                )
            else:
                vpool = tpool

    # Last fully connected layer
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(vpool)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=num_classes,
            activation=None,
            use_bias=True,
        )
        fc = tf.identity(fc, name="fc")
    return fc, phase

def _fir_tv_layer(x, L, time_filter_order, vertex_filter_order, num_filters):
    _, _, _, num_channels = x.get_shape()
    num_channels = int(num_channels)
    with tf.name_scope("fir_tv"):
        with tf.name_scope("weights"):
            hfir = _weight_variable([vertex_filter_order, time_filter_order, num_channels, num_filters])
            _variable_summaries(hfir)
        graph_conv = fir_tv_filtering_conv1d(x, L, hfir, None, "chebyshev")
    return graph_conv

def _fir_tv_layer_time(x, time_filter_order, vertex_filter_order, num_filters):
    _, _, _, num_channels = x.get_shape()
    num_channels = int(num_channels)
    with tf.name_scope("fir_tv"):
        with tf.name_scope("weights"):
            hfir = _weight_variable([vertex_filter_order, time_filter_order, num_channels, num_filters])
            _variable_summaries(hfir)
        graph_conv = fir_tv_filtering_conv1d_time(x, hfir, None)
    return graph_conv



def _sep_fir_layer(x, L, time_filter_order, vertex_filter_order, num_filters):
    _, _, _, num_channels = x.get_shape()
    num_channels = int(num_channels)
    with tf.name_scope("sep_fir"):
        with tf.name_scope("weights"):
            ht = _weight_variable([time_filter_order, num_filters])
            hv = _weight_variable([vertex_filter_order, num_filters])
            hmimo = _weight_variable([num_channels, num_filters])
            _variable_summaries(ht)
            _variable_summaries(hv)
            _variable_summaries(hmimo)
        graph_conv = sep_fir_filtering(x, L, ht, hv, hmimo, None, "chebyshev")
    return graph_conv


def _cheb_conv_layer(x, L, vertex_filter_order, num_filters):
    _, _, _, num_channels = x.get_shape()
    num_channels = int(num_channels)
    with tf.name_scope("fir_tv"):
        with tf.name_scope("weights"):
            hfir = _weight_variable([vertex_filter_order, num_channels, num_filters])
            _variable_summaries(hfir)
        graph_conv = chebyshev_convolution(x, L, hfir, None)
    return graph_conv
