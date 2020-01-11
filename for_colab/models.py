import numpy as np
import tensorflow as tf

from layers import fir_tv_filtering_einsum, chebyshev_convolution, jtv_chebyshev_convolution, \
    fir_tv_filtering_conv1d, sep_fir_filtering


def _weight_variable(shape):
    """_weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=1 / np.prod(shape[:-1]), dtype=tf.float32)
    return tf.Variable(initial)


def _bias_variable(shape):
    """_bias_variable generates a bias variable of a given shape."""
    initial = 0.1 * tf.ones(shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tf.compat.v1.summary.histogram('histogram', var)


def fc_fn(x, output_units):
    """
    Neural network consisting on 1 convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    with tf.name_scope("fc"):
        fc_input = tf.layers.flatten(x)
        fc = tf.layers.dense(
            inputs=fc_input,
            units=output_units,
            activation=None,
            use_bias=True
        )

    return fc


def _batch_normalization(input, is_training=True, scope=None):
    # Note: is_training is tf.compat.v1.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input, is_training=True, decay=0.8,
                                                        center=False, updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input, is_training=False, decay=0.8,
                                                        updates_collections=None, center=False, scope=scope,
                                                        reuse=True))


def deep_fir_tv_fc_fn(x, L, num_classes, time_filter_orders, vertex_filter_orders, num_filters, time_poolings,
                      vertex_poolings, shot_noise=1):
    assert len(time_filter_orders) == len(vertex_filter_orders) == len(num_filters) == len(time_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(time_filter_orders)
    phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    # Convolutional layers
    vpool = tf.nn.dropout(x, shot_noise)
    # vpool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _fir_tv_layer(vpool, L[n], time_filter_orders[n], vertex_filter_orders[n], num_filters[n])
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


def deep_sep_fir_fc_fn(x, L, num_classes, time_filter_orders, vertex_filter_orders, num_filters, time_poolings,
                      vertex_poolings):
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


def deep_cheb_fc_fn(x, L, num_classes, vertex_filter_orders, num_filters, vertex_poolings, shot_noise=1):
    assert len(vertex_filter_orders) == len(num_filters) == len(vertex_poolings), \
        "Filter parameters should all be of the same length"

    n_layers = len(vertex_filter_orders)
    phase = tf.compat.v1.placeholder(tf.bool, name="phase")

    # Convolutional layers
    vpool = tf.nn.dropout(x, shot_noise)
    # vpool = x
    for n in range(n_layers):
        with tf.name_scope("conv%d" % n):
            conv = _cheb_conv_layer(vpool, L[n], vertex_filter_orders[n], num_filters[n])
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
            units=num_classes,
            activation=None,
            use_bias=True
        )

        fc = tf.identity(fc, "fc")
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
    graph_conv = _fir_tv_layer(x, L, time_filter_order, vertex_filter_order, num_filters)
    graph_conv = tf.nn.relu(graph_conv)
    with tf.name_scope("dropout"):
        keep_prob = tf.compat.v1.placeholder(tf.float32)
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
    with tf.name_scope("chebyshev_conv"):
        with tf.name_scope("weights"):
            Wcheb = _weight_variable([filter_order, num_filters])
            _variable_summaries(Wcheb)
        with tf.name_scope("biases"):
            bcheb = _bias_variable([num_filters])
            _variable_summaries(bcheb)
        graph_conv = chebyshev_convolution(x, L, Wcheb, bcheb)
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.compat.v1.placeholder(tf.float32)
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


def jtv_cheb_fc_fn(x, L, output_units, filter_order, num_filters):
    """
    Neural network consisting on 1 joint time-vertex convolutional chebyshev layer and one dense layer
    :param x: input signal
    :param L: graph laplacian
    :param output_units: number of output units
    :param filter_order: order of convolution
    :param num_filters: number of parallel filters
    :return: computational graph
    """
    with tf.name_scope("jtv_chebyshev_conv"):
        with tf.name_scope("weights"):
            Wcheb = _weight_variable([filter_order, num_filters])
            _variable_summaries(Wcheb)
        with tf.name_scope("biases"):
            bcheb = _bias_variable([num_filters])
            _variable_summaries(bcheb)
        graph_conv = jtv_chebyshev_convolution(x, L, Wcheb, bcheb)
        graph_conv = tf.nn.relu(graph_conv)
        with tf.name_scope("dropout"):
            keep_prob = tf.compat.v1.placeholder(tf.float32)
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
