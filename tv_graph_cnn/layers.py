import tensorflow as tf


def sep_fir_filtering(x, S, ht, hv, hmimo, b, kernel="naive"):
    """
    Performs separable FIR-TV (Time-vertex) filtering using conv1d for time FIR
    :param x: Input signal (row dim: time  col dim: vertex)
    :param S: Graph Shift Operator (e.g. Laplacian)
    :param ht: FIR time coefficients (M: Length time filter, F: Number of filters)
    :param hv: FIR vertex coefficients (K: Length vertex filter, F: Number of filters)
    :param hmimo: Channel combination matrix (C: In channels, F: Number of filters)
    :return: Filtered signal
    """

    B, N, T, C = x.get_shape()  # B: number of samples in batch, N: number of nodes, T: temporal length, C: channels
    K, F = hv.get_shape()  # K: Length vertex filter,  F: Number of filters
    M, F = ht.get_shape()  # M: Length time filter, F: Number of filters
    C, F = hmimo.get_shape()  # M: Length time filter, F: Number of filters

    x = tf.transpose(x, perm=[0, 1,  3, 2])  # BxNxCxT
    x = tf.expand_dims(x, axis=4)  # BxNxCxTx1
    x = tf.reshape(x, shape=[-1, T, 1])  # BNCxTx1

    x_convt = tf.nn.conv1d(x, tf.expand_dims(ht, axis=1), stride=1, padding="SAME", data_format="NHWC")  # BNCxTxF
    x_convt = tf.reshape(x_convt, shape=[-1, N, C, T, F])  # BxNxCxTxF
    x_convt = tf.transpose(x_convt, perm=[0, 1, 3, 2, 4])

    with tf.name_scope("kernel_creation"):
        if kernel == "naive":
            SK = _vertex_fir_kernel(S, K)  # KxNxN
        elif kernel == "chebyshev":
            SK = _chebyshev_kernel(S, K)
        else:
            raise ValueError("Specified kernel type {} is not valid." % kernel)

    # KxNxN, BxNxTxCxF -> BxKxNxTxCxF
    # a b c  d c e f g -> d a b e f g
    SKx = tf.einsum("abc,dcefg->dabefg", SK, x_convt)  # BxKxNxTxCxF
    print(SKx.shape)
    # KxF  BxKxNxTxCxF -> BxNxTxCxF
    # a b  c a e f g b -> c e f g b
    Yunmixed = tf.einsum("ab,caefgb->cefgb", hv, SKx)  # BxNxTxCxF
    print(Yunmixed.shape)
    # CxF  BxNxTxCxF -> BxNxTxF
    # a b  c d e a b -> c d e b
    Ymimo = tf.einsum("ab,cdeab->cdeb", hmimo, Yunmixed)
    print(Ymimo.shape)

    if b is not None:
        Ymimo += b
    return Ymimo


def fir_tv_filtering_einsum(x, S, h, b, kernel="naive"):
    """
    Performs FIR-TV (Time-vertex) filtering using left to right computations (using einsum for matrix multiplication)
    K-1
    --
    \     k
    /   S   X  h  = y       t=0,...,T
    --       t  k    t
    k=0
    :param x: Input signal (row dim: time  col dim: vertex)
    :param S: Graph Shift Operator (e.g. Laplacian)
    :param h: FIR-TV coefficients (F: filters of lengths K: Vertex M: Time)
    :return: Filtered signal
    """
    B, N, T, C = x.get_shape()  # B: number of samples in batch, N: number of nodes, T: temporal length, C: channels
    K, M, C, F = h.get_shape()  # K: Length vertex filter, M: Length time filter, C: In channels, F: Number of filters
    M = int(M)
    T = int(T)

    with tf.name_scope("kernel_creation"):
        if kernel == "naive":
            SK = _vertex_fir_kernel(S, K)  # KxNxN
        elif kernel == "chebyshev":
            SK = _chebyshev_kernel(S, K)
        else:
            raise ValueError("Specified kernel type {} is not valid." % kernel)

    with tf.name_scope("tv_conv"):
        for t in range(T):
            if t <= T - M:
                xt = x[:, :, t:t + M, :]  # BxNxMxC
            else:
                # Pad with zeros
                xt = tf.zeros_like(x[:, :, 0:(M - 1) - (T - 1 - t), :])
                xt = tf.concat([xt, x[:, :, t:, :]], axis=2)

            # Use einstein summation for efficiency and compactness
            SKxt = tf.einsum("abc,dcef->dabef", SK, xt)  # BxKxNxMxC
            Yt = tf.einsum("abcde,bdef->abcf", SKxt, h)  # BxKxNxF
            Yt = tf.einsum("abcd->acd", Yt)  # BxNxF
            Yt = tf.reshape(Yt, [-1, N, 1, F])  # BxNx1xF

            if t == 0:
                Y = Yt
            else:
                Y = tf.concat([Y, Yt], axis=2)  # BxNxTxF

    if b is not None:
        Y += b
    return Y


def fir_tv_filtering_conv1d(x, S, h, b, kernel="naive"):
    """
    Performs FIR-TV (Time-vertex) filtering using right to left computations (Time FIR is performed using conv1d)
    K-1
    --
    \     k
    /   S   X  h  = y    t=0,...,T
    --       t  k    t
    k=0
    :param x: Input signal (row dim: time  col dim: vertex)
    :param S: Graph Shift Operator (e.g. Laplacian)
    :param h: FIR-TV coefficients (K: Vertex M: Time F: filters of lengths)
    :return: Filtered signal
    """

    B, N, T, C = x.get_shape()  # B: number of samples in batch, N: number of nodes, T: temporal length, C: channels
    K, M, C, F = h.get_shape()  # K: Length vertex filter, M: Length time filter, C: In channels, F: Number of filters

    x = tf.reshape(x, [-1, T, C])  # BNxTxC
    for k in range(K):
        XHk = tf.nn.conv1d(x, h[k, :, :, :], stride=1, padding="SAME", data_format="NHWC")  # BNxTxF
        XHk = tf.reshape(XHk, [-1, N, T, F])
        XHk = tf.expand_dims(XHk, axis=1)  # BxKxNxTxF
        if k == 0:
            XH = XHk
        else:
            XH = tf.concat([XH, XHk], axis=1)  # BxKxNxTxF

    if kernel == "naive":
        SK = _vertex_fir_kernel(S, K)  # KxNxN
    elif kernel == "chebyshev":
        SK = _chebyshev_kernel(S, K)
    else:
        raise ValueError("Specified kernel type {} is not valid." % kernel)

    # Use einstein summation for efficiency and compactness
    # KxNxN, BxKxNxTxF -> BxKxNxTxF
    # a b c  d a c g h -> d a b g h
    Y = tf.einsum("abc,dacgh->dabgh", SK, XH)  # BxKxNxTxF
    Y = tf.einsum("abcdf->acdf", Y)  # BxNxTxF

    if b is not None:
        Y += b
    return Y

def fir_tv_filtering_conv1d_time(x, h, b):
    #version in which there is no vertex convolutions

    """
    Performs FIR-TV (Time-vertex) filtering using right to left computations (Time FIR is performed using conv1d)
    K-1
    --
    \     k
    /   S   X  h  = y    t=0,...,T
    --       t  k    t
    k=0
    :param x: Input signal (row dim: time  col dim: vertex)
    :param S: Graph Shift Operator (e.g. Laplacian)
    :param h: FIR-TV coefficients (K: Vertex M: Time F: filters of lengths)
    :return: Filtered signal
    """

    B, N, T, C = x.get_shape()  # B: number of samples in batch, N: number of nodes, T: temporal length, C: channels
    K, M, C, F = h.get_shape()  # K: Length vertex filter, M: Length time filter, C: In channels, F: Number of filters

    x = tf.reshape(x, [-1, T, C])  # BNxTxC
    for k in range(K):
        XHk = tf.nn.conv1d(x, h[k, :, :, :], stride=1, padding="SAME", data_format="NHWC")  # BNxTxF
        XHk = tf.reshape(XHk, [-1, N, T, F])
        XHk = tf.expand_dims(XHk, axis=1)  # BxKxNxTxF
        if k == 0:
            XH = XHk
        else:
            XH = tf.concat([XH, XHk], axis=1)  # BxKxNxTxF


    # Use einstein summation for efficiency and compactness
    # KxNxN, BxKxNxTxF -> BxKxNxTxF
    # a b c  d a c g h -> d a b g h
    Y = XH  # BxKxNxTxF
    Y = tf.einsum("abcdf->acdf", Y)  # BxNxTxF

    if b is not None:
        Y += b
    return Y



def chebyshev_convolution(x, L, h, b):
    """
    Graph convolutional layer based on Chebyshev FIR filtering
    :param x: Input signal (NxNvxT)
    :param L: Graph laplacian (NvxNv)
    :param h: Weights of chebyshev filters (KxCxF)
    :param b: biases of chebyshev filters (F)
    :return: Computational graph
    """
    B, N, T, C = x.get_shape()  # B: number of samples in batch, N: number of nodes, T: Time samples, C: input channels
    K, C, F = h.get_shape()  # K: filter order, C: number of input channels, F: number of filters

    # Compute Chebyshev basis
    SK = _chebyshev_kernel(L, K)  # KxNxN

    SKx = tf.einsum("abc,dcef->dabef", SK, x)  # BxKxNxTxC
    Y = tf.einsum("abc,dafgb->dafgc", h, SKx)  # BxKxNxTxF
    Y = tf.einsum("abcde->acde", Y)  # BxNxTxF

    if b is not None:
        Y += b  # BxNxTxF

    return Y


def jtv_chebyshev_convolution(x, L, W, b):
    """
    Joint time-vertex graph convolutional layer based on Chebyshev FIR filtering
    :param x: Input signal (NxNvxT)
    :param L: Graph laplacian (NvxNv)
    :param W: Weights of chebyshev filters (KxF)
    :param b: biases of chebyshev filters (F)
    :return: Computational graph
    """
    # Cast parameters to complex numbers
    xc = tf.complex(x, 0.)
    Wc = tf.complex(W, 0.)
    bc = tf.complex(b, 0.)
    Lc = tf.cast(L, dtype=tf.complex64)

    # Compute FFC
    xf = tf.fft(xc)
    yf = chebyshev_convolution(xf, Lc, Wc, bc)
    yc = tf.ifft(yf)

    # Cast result to real
    yr = tf.real(yc)

    return yr


def _vertex_fir_kernel(S, K):
    """
    Computes polynomial basis for vertex FIR filtering
    :param S: Graph Shift Operator (e.g. Laplacian)
    :param K: Polynomial order
    :return: Tensor with natural powers of S  (KxNxN)
    """
    N, _ = S.get_shape()
    N = int(N)
    Sk = tf.eye(N)
    St = list()
    St.append(tf.reshape(Sk, [1, N, N]))
    for k in range(K - 1):
        Sk = tf.sparse_tensor_dense_matmul(S, Sk)
        St.append(tf.reshape(Sk, [1, N, N]))

    St = tf.concat(St, axis=0)
    return St  # KxNxN


def _chebyshev_kernel(L, K):
    """
    Computes chebyshev kernel to perform convolutions
    :param L: Rescaled [-1, 1] graph laplacian (NxN)
    :return: chebyshev kernel (KxNxN)
    """
    N, _ = L.get_shape()
    N = int(N)
    Tk0 = tf.eye(N)
    Tt = list()
    Tt.append(tf.reshape(Tk0, [1, N, N]))
    if K > 1:
        Tk1 = tf.sparse_tensor_dense_matmul(L, Tk0)
        Tt.append(tf.reshape(Tk1, [1, N, N]))  # KxNxN
    for k in range(2, K):
        Tk2 = 2 * tf.sparse_tensor_dense_matmul(L, Tk1) - Tk0  # NxN
        Tt.append(tf.reshape(Tk2, [1, N, N]))  # KxNxN
        Tk0 = Tk1
        Tk1 = Tk2

    Tt = tf.concat(Tt, axis=0)  # KxNxN
    return Tt


def _chebyshev_basis(x, L, K):
    """
    Computes chebyshev basis to perform convolution
    :param x: Input signal (NxNvxT)
    :param L: Graph laplacian (NvxNv)
    :return: chebyshev basis
    """
    N, Nv, T = x.get_shape()  # N: number of samples, Nv: number of vertices, T: number of features per vertex

    # Transform to Chebyshev basis
    xt0 = tf.transpose(x, perm=[1, 2, 0])  # Nv x T x N
    xt = tf.expand_dims(xt0, 0)  # 1 x Nv x T x N

    # Chebyshev recursion
    def concat(xt, x):
        x = tf.expand_dims(x, 0)  # 1 x Nv x T x N
        return tf.concat([xt, x], axis=0)  # K x Nv x T x N

    if K > 1:
        xt0 = tf.reshape(xt0, [Nv, -1])  # Reshape because matmul only accepts 2D tensors (Nv x TN)
        xt1 = tf.sparse_tensor_dense_matmul(L, xt0)  # Nv x TN
        xt0 = tf.reshape(xt0, [Nv, T, -1])  # Nv x T x N
        xt1 = tf.reshape(xt1, [Nv, T, -1])  # Nv x T x N
        xt = concat(xt, xt1)  # K x Nv x T x N
    for k in range(2, K):
        xt0 = tf.reshape(xt0, [Nv, -1])  # Reshape because matmul only accepts 2D tensors (Nv x TN)
        xt1 = tf.reshape(xt1, [Nv, -1])  # Nv x TN
        xt2 = 2 * tf.sparse_tensor_dense_matmul(L, xt1) - xt0  # Nv x TN
        xt1 = tf.reshape(xt1, [Nv, T, -1])  # Nv x T x N
        xt2 = tf.reshape(xt2, [Nv, T, -1])  # Nv x T x N
        xt = concat(xt, xt2)  # K x Nv x T x N
        xt0, xt1 = xt1, xt2

    xt = tf.transpose(xt, perm=[3, 1, 2, 0])  # N x Nv x T x K
    return xt
