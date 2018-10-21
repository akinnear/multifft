package com.github.kinnear.multifft.fftw;

import com.github.kinnear.multifft.AbstractFFT;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.presets.fftw3;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

import static org.bytedeco.javacpp.fftw3.FFTW_BACKWARD;
import static org.bytedeco.javacpp.fftw3.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.fftw3.FFTW_FORWARD;
import static org.bytedeco.javacpp.fftw3.fftw_cleanup;
import static org.bytedeco.javacpp.fftw3.fftw_destroy_plan;
import static org.bytedeco.javacpp.fftw3.fftw_execute_dft;
import static org.bytedeco.javacpp.fftw3.fftw_plan;
import static org.bytedeco.javacpp.fftw3.fftw_plan_dft_1d;

public class FftwFFT extends AbstractFFT implements AutoCloseable {
    static {
        Loader.load(fftw3.class);
    }

    private Map<SizeDirection, fftw_plan> planMap = new ConcurrentHashMap<>();

    @Override
    protected double[] safeFft(double[] input) {
        return fftWithSign(input, FFTW_FORWARD);
    }

    @Override
    protected double[] safeIfft(double[] input) {
        return fftWithSign(input, FFTW_BACKWARD);
    }

    private double[] fftWithSign(double[] input, int sign) {
        int inputSize = input.length;
        int numPairs = inputSize / 2;
        DoublePointer signal = new DoublePointer(input);
        DoublePointer result = new DoublePointer(inputSize);
        fftw_plan plan = planMap.computeIfAbsent(
                new SizeDirection(numPairs, sign),
                i -> fftw_plan_dft_1d(i.size, signal, result, i.direction, (int) FFTW_ESTIMATE));
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

        SizeDirection(int size, int direction) {
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
