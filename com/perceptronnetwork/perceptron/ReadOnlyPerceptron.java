package com.perceptronnetwork.perceptron;

public interface ReadOnlyPerceptron {
	int getInputCount();
	float getInputValue(int inputIndex);

	float getWeight(int inputIndex);
	float getBias();

	float getOutput();
	float getDerivative();

	float getError();
	float getErrorContribution(int inputIndex);
}
