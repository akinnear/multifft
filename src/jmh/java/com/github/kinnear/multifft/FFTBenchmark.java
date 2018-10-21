package com.github.kinnear.multifft;

import com.github.kinnear.multifft.commons.ApacheCommonsFFT;
import com.github.kinnear.multifft.fftw.FftwFFT;
import com.github.kinnear.multifft.jcufft.JcufftFFT;
import com.github.kinnear.multifft.jtransform.JTransformFFT;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Random;

import static java.util.concurrent.TimeUnit.SECONDS;

@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@Measurement(iterations = 3, time = 10, timeUnit = SECONDS)
@Warmup(iterations = 2, time = 10, timeUnit = SECONDS)
@Fork(value=1)
public class FFTBenchmark {
    @Param({
            "1024",
            "4096"})
    public int size=0;

    @Param({"COMMONS", "FFTW", "JCUFT", "JTRANSFORM"})
    public String implementation="";

    private FFT fft;
    private double[] input;
    private double[] result;

    @Setup
    public void setup(){
        FFTType fftType = FFTType.valueOf(implementation);
        fft = createFFT(fftType);
        input = new Random().doubles(size, -Math.PI, Math.PI).toArray();
        result = new double[size];
    }

    @TearDown
    public void tearDown() {
        if (fft instanceof FftwFFT) {
            ((FftwFFT) fft).close();
        }
    }

    private FFT createFFT(FFTType fftType) {
        switch (fftType) {
            case COMMONS:
                return new ApacheCommonsFFT();
            case FFTW:
                return new FftwFFT();
            case JCUFT:
                return new JcufftFFT();
            case JTRANSFORM:
                return new JTransformFFT();
        }
        return null;
    }

    @Benchmark
    public double[] forwardFft() {
        result = fft.fft(input);
        return result;
    }

    @Benchmark
    public double[] inverseFft() {
        result = fft.ifft(input);
        return result;
    }
}