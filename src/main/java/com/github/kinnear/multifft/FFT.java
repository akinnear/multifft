package com.github.kinnear.multifft;

public interface FFT {
    double[] fft(double[] input);
    double[] ifft(double[] input);

    class IllegalInput extends RuntimeException {
    }
}
