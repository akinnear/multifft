package com.github.kinnear.multifft.fftw;

import com.github.kinnear.multifft.FFT;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.presets.fftw3;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

import static org.bytedeco.javacpp.fftw3.*;

public class FftwFFT implements FFT, AutoCloseable {
    static {
        Loader.load(fftw3.class);
    }

    private Map<SizeDirection, fftw_plan> planMap = new ConcurrentHashMap<>();

    @Override
    public double[] fft(double[] input) {
        if (input == null) {
            throw new IllegalInput();
        }
        if (input.length == 0) {
            return new double[0];
        }

        return fftWithSign(input, FFTW_FORWARD);
    }

    @Override
    public double[] ifft(double[] input) {
        if (input == null) {
            throw new IllegalInput();
        }
        if (input.length == 0) {
            return new double[0];
        }

        return fftWithSign(input, FFTW_BACKWARD);
    }

    private double[] fftWithSign(double[] input, int sign) {
        int inputSize = input.length;
        int numPairs = inputSize / 2;
        DoublePointer signal = new DoublePointer(input);
        DoublePointer result = new DoublePointer(inputSize);
        fftw_plan plan = planMap.computeIfAbsent(
                new SizeDirection(numPairs, sign),
                i -> fftw_plan_dft_1d(i.size, signal, result, sign, (int) FFTW_ESTIMATE));
        fftw_execute_dft(plan, signal, result);
        double[] output = new double[inputSize];
        result.get(output);
        return output;
    }

    @Override
    public void close() {
        try {
            planMap.values().forEach(this::closePlan);
            planMap.clear();
        } finally {
            fftw_cleanup();
        }
    }

    private void closePlan(fftw_plan plan) {
        if (plan != null) {
            try {
                fftw_destroy_plan(plan);
            } finally {
                plan.close();
            }
        }
    }

    private static class SizeDirection {
        private int size;
        private int direction;

        public SizeDirection(int size, int direction) {
            this.size = size;
            this.direction = direction;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            SizeDirection that = (SizeDirection) o;
            return size == that.size &&
                    direction == that.direction;
        }

        @Override
        public int hashCode() {

            return Objects.hash(size, direction);
        }
    }
}
