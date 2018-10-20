package com.github.kinnear.multifft.fftw;

import com.github.kinnear.multifft.FFT;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static com.github.kinnear.multifft.FFTResults.*;
import static org.junit.Assert.*;

public class FftwFFTTest {
    private static final double TEST_DOUBLE_PRECISION = 1e-10;
    private FftwFFT fftwFFT;

    @Before
    public void setUp() {
        fftwFFT = new FftwFFT();
    }

    @After
    public void tearDown() {
        fftwFFT.close();
    }

    @Test(expected = FFT.IllegalInput.class)
    public void testNullFFT() {
        fftwFFT.fft(null);
    }

    @Test
    public void testEmptyFFT() {
        double[] result = fftwFFT.fft(new double[0]);
        assertArrayEquals(new double[0], result, TEST_DOUBLE_PRECISION);
    }

    @Test
    public void testSinglePairFFT() {
        testFFT(2, FFT_1_TO_2);
    }

    @Test
    public void testTwoPairFFT() {
        testFFT(4, FFT_1_TO_4);
    }

    @Test
    public void test128PairFFT() {
        testFFT(256, FFT_1_TO_256);
    }

    @Test
    public void test2PairsOf128PairFFT() {
        testFFT(256, FFT_1_TO_256);
        testFFT(257, 256, FFT_257_TO_512);
    }

    @Test
    public void test256PairFFT() {
        testFFT(512, FFT_1_TO_512);
    }

    private void testFFT(int limit, double[] expected) {
        testFFT(1, limit, expected);
    }

    private void testFFT(int startSeed, int limit, double[] expected) {
        double[] input = DoubleStream.iterate(startSeed, i->i+1).limit(limit).toArray();
        double[] result = fftwFFT.fft(input);
        assertArrayEquals(expected, result, TEST_DOUBLE_PRECISION);
    }

    @Test(expected=FFT.IllegalInput.class)
    public void testNullIFFT() {
        fftwFFT.ifft(null);
    }

    @Test
    public void testEmptyIFFT() {
        double[] result = fftwFFT.ifft(new double[0]);
        assertArrayEquals(new double[0], result, TEST_DOUBLE_PRECISION);
    }

    @Test
    public void testSinglePairIFFT() {
        testIFFT(2, FFT_1_TO_2);
    }

    @Test
    public void testTwoPairIFFT() {
        testIFFT(4, FFT_1_TO_4);
    }

    @Test
    public void test128PairIFFT() {
        testIFFT(256, FFT_1_TO_256);
    }

    @Test
    public void test2PairsOf128PairIFFT() {
        testIFFT(256, FFT_1_TO_256);
        testIFFT(257, 256, FFT_257_TO_512);
    }

    @Test
    public void test256PairIFFT() {
        testIFFT(512, FFT_1_TO_512);
    }

    private void testIFFT(int limit, double[] input) {
        testIFFT(1, limit, input);
    }

    private void testIFFT(int seed, int limit, double[] input) {
        // scaling is required for IFFT to match
        double[] expected = DoubleStream.iterate(seed, i->i+1).limit(limit).map(i->i*input.length/2).toArray();
        double[] result = fftwFFT.ifft(Arrays.stream(input).toArray());
        assertArrayEquals(expected, result, TEST_DOUBLE_PRECISION);
    }

}
