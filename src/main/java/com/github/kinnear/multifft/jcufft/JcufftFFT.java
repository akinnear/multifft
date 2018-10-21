package com.github.kinnear.multifft.jcufft;

import com.github.kinnear.multifft.AbstractFFT;
import jcuda.jcufft.cufftHandle;

import static jcuda.jcufft.JCufft.CUFFT_FORWARD;
import static jcuda.jcufft.JCufft.CUFFT_INVERSE;
import static jcuda.jcufft.JCufft.cufftDestroy;
import static jcuda.jcufft.JCufft.cufftExecZ2Z;
import static jcuda.jcufft.JCufft.cufftPlan1d;
import static jcuda.jcufft.cufftType.CUFFT_Z2Z;

public class JcufftFFT extends AbstractFFT {
    @Override
    protected double[] safeFft(double[] input) {
        cufftHandle plan = new cufftHandle();
        cufftPlan1d(plan, input.length/2, CUFFT_Z2Z, 1);
        double[] result = new double[input.length];
        cufftExecZ2Z(plan, input, result, CUFFT_FORWARD);
        cufftDestroy(plan);
        return result;
    }

    @Override
    protected double[] safeIfft(double[] input) {
        cufftHandle plan = new cufftHandle();
        cufftPlan1d(plan, input.length/2, CUFFT_Z2Z,1);
        double[] result = new double[input.length];
        cufftExecZ2Z(plan, input, result, CUFFT_INVERSE);
        cufftDestroy(plan);
        return result;
    }
}
