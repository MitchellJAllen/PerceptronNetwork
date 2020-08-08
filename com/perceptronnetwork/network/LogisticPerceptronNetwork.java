package com.perceptronnetwork.network;

import com.perceptronnetwork.perceptron.LogisticPerceptron;
import com.perceptronnetwork.perceptron.Perceptron;

public class LogisticPerceptronNetwork extends PerceptronNetwork {
	public LogisticPerceptronNetwork(int inputDimensions, int[] hiddenDimensions, int outputDimensions) {
		super(inputDimensions, hiddenDimensions, outputDimensions);
	}

	protected Perceptron constructPerceptron(
		int inputCount,
		int layerSize,
		int layerIndex,
		int elementIndex
	) {
		return new LogisticPerceptron(inputCount); // all Perceptrons use Logistic function
	}
}
