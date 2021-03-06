package com.github.kinnear.multifft.jtransform;

import com.github.kinnear.multifft.FFT;
import org.junit.Before;
import org.junit.Test;

import java.util.stream.DoubleStream;

import static com.github.kinnear.multifft.FFTResults.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class JTransformFFTTest {
    private static final double TEST_DOUBLE_PRECISION = 1e-10;
    private JTransformFFT transformFFT;

    @Before
    public void setUp() {
        transformFFT = new JTransformFFT();
    }

    @Test(expected = FFT.IllegalInput.class)
    public void testNullFFT() {
        transformFFT.fft(null);
    }

    @Test(expected=FFT.IllegalInput.class)
    public void testNullIFFT() {
        transformFFT.ifft(null);
    }

    @Test
    public void testEmpty() {
        double[] input = new double[0];
        double[] result = transformFFT.fft(input);
        assertEquals(0, result.length);
        result = transformFFT.ifft(input);
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
        double[] ifftResult = transformFFT.ifft(input);
        assertArrayEquals(expected, ifftResult, TEST_DOUBLE_PRECISION);
    }

    private double[] testFFT(double[] expected, double[] input) {
        double[] fftResult = transformFFT.fft(input);
        assertArrayEquals(expected, fftResult, TEST_DOUBLE_PRECISION);
        return fftResult;
    }
}
