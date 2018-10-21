package com.github.kinnear.multifft.commons;

import com.github.kinnear.multifft.AbstractFFT;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class ApacheCommonsFFT extends AbstractFFT {
    @Override
    protected double[] safeFft(double[] input) {
        Complex[] complexInput = doubleArrayToComplex(input);
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] result = fft.transform(complexInput, TransformType.FORWARD);
        return complexToDoubleArray(result);
    }

    @Override
    protected double[] safeIfft(double[] input) {
        Complex[] complexInput = doubleArrayToComplex(input);
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] result = fft.transform(complexInput, TransformType.INVERSE);
        return complexToDoubleArray(result);
    }

    private Complex[] doubleArrayToComplex(double[] values) {
        return IntStream.iterate(0,  i -> i + 2)
                .limit(values.length/2)
                .mapToObj(i -> new Complex(values[i], values[i+1]))
                .toArray(Complex[]::new);
    }

    private double[] complexToDoubleArray(Complex[] values) {
        return Arrays.stream(values)
                .flatMapToDouble(c -> DoubleStream.of(c.getReal(), c.getImaginary()))
                .toArray();
    }
}
