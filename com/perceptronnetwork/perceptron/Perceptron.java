package com.perceptronnetwork.perceptron;

import java.util.Random;

public abstract class Perceptron implements ReadOnlyPerceptron {
	private float[] inputs;
	private float[] weights;
	private float bias;

	private float output;
	private float derivative;
	private float error;

	private boolean outputCached;
	private boolean derivativeCached;

	public abstract float calculateOutput(float weightedSum);
	public abstract float calculateDerivative(float output);

	public void reset() {
		Random random = new Random();

		for (int inputIndex = 0; inputIndex < this.weights.length; inputIndex++) {
			this.weights[inputIndex] = (float)(random.nextGaussian());
		}

		this.bias = 0;
		this.error = 0;

		this.outputCached = false;
		this.derivativeCached = false;
	}

	public Perceptron(int inputCount) { // TODO: validate inputCout >= 0
		this.inputs = new float[inputCount];
		this.weights = new float[inputCount];

		this.reset();
	}

	public int getInputCount() {
		return this.inputs.length;
	}

	public float getInputValue(int inputIndex) { // TODO: validate inputIndex
		return this.inputs[inputIndex];
	}

	public void setInputValue(int inputIndex, float inputValue) { // TODO: validate inputIndex
		this.inputs[inputIndex] = inputValue;

		this.outputCached = false;
		this.derivativeCached = false;
	}

	public void setInputValues(float[] inputValues) { // TODO: validate inputs.length
		System.arraycopy(inputValues, 0, this.inputs, 0, this.inputs.length);

		this.outputCached = false;
		this.derivativeCached = false;
	}

	public float getWeight(int inputIndex) { // TODO: validate inputIndex
		return this.weights[inputIndex];
	}

	public float getBias() {
		return this.bias;
	}

	public float getOutput() {
		if (!this.outputCached) {
			float weightedSum = this.bias;

			for (int inputIndex = 0; inputIndex < this.inputs.length; inputIndex++) {
				weightedSum += this.inputs[inputIndex] * this.weights[inputIndex];
			}

			this.output = this.calculateOutput(weightedSum);
			this.outputCached = true;
		}

		return this.output;
	}

	public float getDerivative() {
		if (!this.derivativeCached) {
			this.derivative = this.calculateDerivative(this.getOutput());
			this.derivativeCached = true;
		}

		return this.derivative;
	}

	public float getError() {
		return this.error;
	}

	public void resetError() {
		this.error = 0;
	}

	public void addError(float error) {
		this.error += error;
	}

	public float getErrorContribution(int inputIndex) { // TODO: validate inputIndex
		return this.error * this.getDerivative() * this.weights[inputIndex];
	}

	public void trainFromError(float learningRate) { // TODO: add learningRate member
		float correctionValue = this.getDerivative() * this.error;

		for (int inputIndex = 0; inputIndex < this.inputs.length; inputIndex++) {
			this.weights[inputIndex] += this.inputs[inputIndex] * correctionValue * learningRate;
		}

		this.bias += correctionValue * learningRate;

		this.outputCached = false;
		this.derivativeCached = false;
	}
}
