package com.perceptronnetwork.perceptron;

public class LogisticPerceptron extends Perceptron {
	public LogisticPerceptron(int inputCount) {
		super(inputCount);
	}

	public float calculateOutput(float weightedSum) {
		return (float)(1 / (1 + Math.exp(-weightedSum)));
	}

	public float calculateDerivative(float output) {
		return output * (1 - output);
	}
}
