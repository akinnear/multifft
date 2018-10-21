package com.github.kinnear.multifft.jcufft;

import com.github.kinnear.multifft.FFT;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static com.github.kinnear.multifft.FFTResults.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class JcufftFFTTest {
    private static final double TEST_DOUBLE_PRECISION = 1e-10;
    private JcufftFFT jcufftFFT;

    @Before
    public void setUp() {
        jcufftFFT = new JcufftFFT();
    }

    @Test(expected = FFT.IllegalInput.class)
    public void testNullFFT() {
        jcufftFFT.fft(null);
    }

    @Test(expected=FFT.IllegalInput.class)
    public void testNullIFFT() {
        jcufftFFT.ifft(null);
    }

    @Test
    public void testEmpty() {
        double[] input = new double[0];
        double[] result = jcufftFFT.fft(input);
        assertEquals(0, result.length);
        result = jcufftFFT.ifft(input);
        assertEquals(0, result.length);
    }

    @Test
    public void test1Pair() {
        generatedRoundTripTest(FFT_1_TO_2, 1, 2);
    }

    @Test
    public void test2Pairs() {
        generatedRoundTripTest(FFT_1_TO_4, 1, 4);
    }

    @Test
    public void test7Pairs() {
        generatedRoundTripTest(FFT_1_TO_14, 1, 14);
    }

    @Test
    public void test128Pairs() {
        generatedRoundTripTest(FFT_1_TO_256, 1, 256);
    }

    @Test
    public void test2PairsOf128Pairs() {
        generatedRoundTripTest(FFT_1_TO_256, 1, 256);
        generatedRoundTripTest(FFT_257_TO_512, 257, 256);
    }

    @Test
    public void test256Pairs() {
        generatedRoundTripTest(FFT_1_TO_512, 1, 512);
    }

    @Test
    public void test256PairsJustInverse() {
        testIFFT(createInput(1,512), FFT_1_TO_512);
    }

    private void generatedRoundTripTest(double[] expected, int start, int limit) {
        double[] input = createInput(start, limit);
        roundTripTest(expected, input);
    }

    private double[] createInput(int start, int limit) {
        return DoubleStream.iterate(start, i->i+1).limit(limit).toArray();
    }

    private void roundTripTest(double[] expected, double[] input) {
        double[] fftResult = testFFT(expected, input);
        testIFFT(input, fftResult);
    }

    private void testIFFT(double[] expected, double[] input) {
        double[] ifftResult = jcufftFFT.ifft(input);
        double[] scaledIfftResult = handleFftwScaling(ifftResult);
        assertArrayEquals(expected, scaledIfftResult, TEST_DOUBLE_PRECISION);
    }

    private double[] testFFT(double[] expected, double[] input) {
        double[] fftResult = jcufftFFT.fft(input);
        assertArrayEquals(expected, fftResult, TEST_DOUBLE_PRECISION);
        return fftResult;
    }

    private double[] handleFftwScaling(double[] input) {
        int numPairs = input.length / 2;
        return Arrays.stream(input).map(i -> i/numPairs).toArray();
    }

}
