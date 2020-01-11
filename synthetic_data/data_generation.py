import numpy as np

from scipy.sparse.linalg import eigsh
from scipy.signal import square


def delta(Nv, c):
    d = np.zeros([Nv, 1])
    d[c] = 1
    return d


def diffusion_seq(W, x, T):
    N = W.shape[0]
    X = np.zeros((N, T))
    for t in range(T):
        if t == 0:
            X[:, 0] = x.T / np.max(x)
            continue
        X[:, t] = ((W ** t) * X[:, t - 1]).T
        X[:, t] /= np.max(X)
    return X


def generate_diffusion_samples(N, W, T=10, sigma=0.05):
    samples = []
    labels = []
    Nv = W.shape[0]
    for _ in range(N):
        k = np.random.randint(Nv)
        Wk = W ** k
        c = np.random.randint(Nv)
        x = Wk.dot(delta(Nv, c))
        X = diffusion_seq(W, x, T) + sigma * np.random.randn(Nv, T)
        samples.append(X)
        labels.append(c)
    return np.array(samples), np.array(labels)


def wave_seq(Wphi, b, Kstart, T, f, duty_fixed=False, signal="square"):
    Nv = Wphi.shape[0]
    X = np.zeros((Nv, Kstart + T))
    phi = 2 * np.pi * np.random.rand()

    if duty_fixed:
        duty = 0.5
    else:
        duty = np.random.rand()
    if signal == "square":
        X[:, 0] = (b * 0.5 * (1 - square(phi, duty=duty))).T
        for t in range(1, Kstart + T):
            X[:, t] = Wphi.dot(X[:, t - 1]).T + (b * 0.5 * (1 - square(f * t + phi, duty=duty))).T
    elif signal == "cosine":
        X[:, 0] = (b * 0.5 * (1 - np.cos(phi))).T
        for t in range(1, Kstart + T):
            X[:, t] = Wphi.dot(X[:, t - 1]).T + (b * 0.5 * (1 - np.cos(f * t + phi))).T
    else:
        raise ValueError("Signal type is not implemented")

    return X[:, Kstart:Kstart + T]


def generate_wave_samples(N, W, T=10, Kmin=10, Kmax=100, sigma=0.05, signal="square"):
    samples = []
    labels = []
    Nv = W.shape[0]
    _, U = eigsh(W)
    Wphi = U.dot(U.T)
    for _ in range(N):
        kstart = np.random.randint(Kmin, Kmax)
        c = np.random.randint(Nv)
        b = delta(Nv, c)
        f = np.pi * np.random.rand()
        X = wave_seq(Wphi, b, kstart, T, f, signal=signal) + sigma * np.random.randn(Nv, T)
        samples.append(X)
        labels.append(c)
    return np.array(samples), np.array(labels)


def hp_hp_sample(T, G, f_c, lambda_c, sigma=1):
    """
    Generates a random low-pass signal in time and vertex
    :param T: Number of time samples
    :param G: Underlying graph
    :param f_c: Index of cut frequency in time fourier domain
    :param lambda_c: Index of cut frequency in graph fourier domain
    :param sigma: Standard deviation of generator
    :return: Filtered signal
    """
    x = sigma * np.random.randn(G.N, T)
    G.compute_fourier_basis()
    xg = G.gft(x)
    xgf = np.fft.fft(xg)
    xgf = np.fft.fftshift(xgf)
    xgf[:lambda_c, :] = 0
    xgf[:, T // 2 - f_c: T // 2 + f_c + 1] = 0
    xgf = np.fft.ifftshift(xgf)
    xg = np.fft.ifft(xgf)
    x = G.igft(xg)
    return x


def lp_lp_sample(T, G, f_c, lambda_c, sigma=1):
    """
    Generates a random low-pass signal in time and vertex
    :param T: Number of time samples
    :param G: Underlying graph
    :param f_c: Index of cut frequency in time fourier domain
    :param lambda_c: Index of cut frequency in graph fourier domain
    :param sigma: Standard deviation of generator
    :return: Filtered signal
    """
    x = sigma * np.random.randn(G.N, T)
    G.compute_fourier_basis()
    xg = G.gft(x)
    xgf = np.fft.fft(xg)
    xgf = np.fft.fftshift(xgf)
    lp_lp_filter = np.zeros(xgf.shape)
    lp_lp_filter[:lambda_c, T // 2 - f_c: T // 2 + f_c + 1] = 1
    xgf = xgf * lp_lp_filter
    xgf = np.fft.ifftshift(xgf)
    xg = np.fft.ifft(xgf)
    x = G.igft(xg)
    return x


def hp_lp_sample(T, G, f_c, lambda_c, sigma=1):
    """
    Generates a random low-pass signal in time and high-pass vertex
    :param T: Number of time samples
    :param G: Underlying graph
    :param f_c: Index of cut frequency in time fourier domain
    :param lambda_c: Index of cut frequency in graph fourier domain
    :param sigma: Standard deviation of generator
    :return: Filtered signal
    """
    x = sigma * np.random.randn(G.N, T)
    G.compute_fourier_basis()
    xg = G.gft(x)
    xgf = np.fft.fft(xg)
    xgf = np.fft.fftshift(xgf)
    xgf[lambda_c:, :] = 0
    xgf[:, T // 2 - f_c: T // 2 + f_c + 1] = 0
    xgf = np.fft.ifftshift(xgf)
    xg = np.fft.ifft(xgf)
    x = G.igft(xg)
    return x


def lp_hp_sample(T, G, f_c, lambda_c, sigma=1):
    """
    Generates a random high-pass signal in time and low-pass vertex
    :param T: Number of time samples
    :param G: Underlying graph
    :param f_c: Index of cut frequency in time fourier domain
    :param lambda_c: Index of cut frequency in graph fourier domain
    :param sigma: Standard deviation of generator
    :return: Filtered signal
    """
    x = sigma * np.random.randn(G.N, T)
    G.compute_fourier_basis()
    xg = G.gft(x)
    xgf = np.fft.fft(xg)
    xgf = np.fft.fftshift(xgf)
    hp_lp_filter = np.zeros(xgf.shape)
    hp_lp_filter[lambda_c:, T // 2 - f_c: T // 2 + f_c + 1] = 1
    xgf = xgf * hp_lp_filter
    xgf = np.fft.ifftshift(xgf)
    xg = np.fft.ifft(xgf)
    x = G.igft(xg)
    return x


def generate_spectral_samples(N, T, G, f_h, lambda_h, f_l, lambda_l, sigma=10, sigma_n=0):
    """
    Generate dataset composed of quantized filtered hp-hp, lp-lp, hp-lp and lp-hp white noise.
    :param N: Number of samples per type
    :param T: Time length
    :param G: Underlying graph
    :param f_h: Index of hp cut frequency in time fourier domain
    :param lambda_h: Index of hp cut frequency in graph fourier domain
    :param f_l: Index of lp cut frequency in time fourier domain
    :param lambda_l: Index of lp cut frequency in graph fourier domain
    :param sigma: Standard deviation of generator
    :param sigma_n: Standard deviation of noise
    :return: Filtered signal
    """
    dataset = []
    labels = []
    for _ in range(N):
        x = hp_hp_sample(T, G, f_h, lambda_h, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(0)
    for _ in range(N):
        x = lp_lp_sample(T, G, f_l, lambda_l, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(1)
    for _ in range(N):
        x = lp_hp_sample(T, G, f_l, lambda_h, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(2)
    for _ in range(N):
        x = hp_lp_sample(T, G, f_h, lambda_l, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(3)
    dataset = np.array(dataset)
    dataset += sigma_n * np.random.randn(*dataset.shape)
    print(dataset.shape)
    labels = np.array(labels)
    return dataset, labels


def generate_spectral_samples_hard(N, T, G, f_h, lambda_h, f_l, lambda_l, sigma=10, sigma_n=0):
    """
    Generate dataset composed of quantized filtered hp-hp, lp-lp, hp-lp and lp-hp white noise.
    :param N: Number of samples per type
    :param T: Time length
    :param G: Underlying graph
    :param f_h: Index of hp cut frequency in time fourier domain
    :param lambda_h: Index of hp cut frequency in graph fourier domain
    :param f_l: Index of lp cut frequency in time fourier domain
    :param lambda_l: Index of lp cut frequency in graph fourier domain
    :param sigma: Standard deviation of generator
    :param sigma_n: Standard deviation of noise
    :return: Filtered signal
    """
    dataset = []
    labels = []
    for _ in range(N):
        x = hp_hp_sample(T, G, f_h, lambda_h, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(0)
    for _ in range(N):
        x = lp_lp_sample(T, G, f_l, lambda_l, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(1)
    for _ in range(N):
        x = lp_hp_sample(T, G, f_l, lambda_h, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(2)
    for _ in range(N):
        x = hp_lp_sample(T, G, f_h, lambda_l, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(3)

    # Spectral Combinations
    for _ in range(N):
        x = lp_lp_sample(T, G, f_h, lambda_h, sigma) + hp_hp_sample(T, G, f_h, lambda_h, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(4)
    for _ in range(N):
        x = lp_hp_sample(T, G, f_h, lambda_h, sigma) + hp_lp_sample(T, G, f_h, lambda_h, sigma)
        x = np.tanh(x.real)
        x = np.expand_dims(x, axis=3)
        dataset.append(x)
        labels.append(5)

    dataset = np.array(dataset)
    dataset += sigma_n * np.random.randn(*dataset.shape)
    print('\n\n First method')
    print(dataset.shape)
    labels = np.array(labels)
    print(labels.shape)
    return dataset, labels
