package com.github.kinnear.multifft.jtransform;

import com.github.kinnear.multifft.AbstractFFT;
import org.jtransforms.fft.DoubleFFT_1D;

public class JTransformFFT extends AbstractFFT {
    @Override
    protected double[] safeFft(double[] input) {
        DoubleFFT_1D fft = new DoubleFFT_1D(input.length/2);
        fft.complexForward(input);
        return input;
    }

    @Override
    protected double[] safeIfft(double[] input) {
        DoubleFFT_1D fft = new DoubleFFT_1D(input.length/2);
        fft.complexInverse(input, true);
        return input;
    }
}
