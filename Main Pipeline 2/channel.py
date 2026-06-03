import numpy as np
from scipy.signal import resample_poly
from math import gcd

class Channel():
    signal: np.ndarray
    cannel_out: np.ndarray

    def __init__(self, signal):
        self.signal = signal
        self.channel_out = signal

    def isi(self, h:np.ndarray = None):
        self.channel_out = np.convolve(self.channel_out, h)

    def delay(self, n = 48000):
        self.channel_out = np.concat([np.zeros(n), self.channel_out])

    def noise(self, snr=-10):
        A = 10 ** (snr/10) * np.std(self.signal) ** 2
        self.channel_out = self.channel_out + A * np.random.randn(len(self.channel_out))

    def sfo(self, sfo_ppm):
        scale = 1_000_000
        p = round(scale + sfo_ppm)
        q = scale
        g = gcd(p, q)
        p, q = p // g, q // g

        # self.channel_out = resample_poly(self.channel_out, q, p)
        interp = np.interp(np.linspace(0, q,len(self.channel_out) * p //q), np.linspace(0, q, len(self.channel_out)),self.channel_out)
        self.channel_out = interp

    def apply_default_channel(self,  h:np.ndarray = None,n=48000,snr=-10, sfo_ppm=10):
        self.isi(h)
        self.delay(n)
        self.noise(snr)
        self.sfo(sfo_ppm)
        return self.channel_out