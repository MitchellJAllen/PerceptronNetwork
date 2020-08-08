package com.perceptronnetwork.perceptron;

public class HyperbolicTangentPerceptron extends Perceptron {
	public HyperbolicTangentPerceptron(int inputCount) {
		super(inputCount);
	}

	public float calculateOutput(float weightedSum) {
		float squaredExponential = (float)Math.exp(2 * weightedSum);

		return (squaredExponential - 1) / (squaredExponential + 1);
	}

	public float calculateDerivative(float output) {
		return 1 - output * output;
	}
}
