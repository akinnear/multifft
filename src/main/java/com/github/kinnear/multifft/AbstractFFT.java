package com.github.kinnear.multifft;

public abstract class AbstractFFT implements FFT {
    @Override
    public double[] fft(double[] input) {
        if (input == null) {
            throw new IllegalInput();
        }
        if (input.length == 0) {
            return new double[0];
        }

        return safeFft(input);
    }

    protected abstract double[] safeFft(double[] input);

    @Override
    public double[] ifft(double[] input) {
        if (input == null) {
            throw new IllegalInput();
        }
        if (input.length == 0) {
            return new double[0];
        }

        return safeIfft(input);
    }

    protected abstract double[] safeIfft(double[] input);
}
