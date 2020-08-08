package com.perceptronnetwork.network;

import com.perceptronnetwork.perceptron.Perceptron;
import com.perceptronnetwork.perceptron.ReadOnlyPerceptron;

public abstract class PerceptronNetwork {
	private Perceptron[][] perceptrons;

	protected abstract Perceptron constructPerceptron(
		int inputCount,
		int layerSize,
		int layerIndex,
		int elementIndex
	);

	public PerceptronNetwork(int inputDimensions, int[] hiddenDimensions, int outputDimensions) { // TODO: validate dimensions > 0
		if (hiddenDimensions == null) {
			hiddenDimensions = new int[0];
		}

		// join all dimensions into single array
		int[] dimensions = new int[hiddenDimensions.length + 2];
		this.perceptrons = new Perceptron[dimensions.length - 1][];

		dimensions[0] = inputDimensions;
		dimensions[dimensions.length - 1] = outputDimensions;

		System.arraycopy(hiddenDimensions, 0, dimensions, 1, hiddenDimensions.length);

		// construct perceptron layers based on dimensions array
		for (int layerIndex = 0; layerIndex < dimensions.length - 1; layerIndex++) {
			int inputCount = dimensions[layerIndex];
			int layerSize = dimensions[layerIndex + 1];

			this.perceptrons[layerIndex] = new Perceptron[layerSize];

			for (int elementIndex = 0; elementIndex < layerSize; elementIndex++) {
				this.perceptrons[layerIndex][elementIndex] = this.constructPerceptron(
					inputCount,
					layerSize,
					layerIndex,
					elementIndex
				);
			}
		}
	}

	public ReadOnlyPerceptron getPerceptron(int layerIndex, int elementIndex) { // TODO: validate indices
		return this.perceptrons[layerIndex][elementIndex];
	}

	public float[] evaluate(float[] inputValues) {
		// set network input
		for (Perceptron perceptron : this.perceptrons[0]) {
			perceptron.setInputValues(inputValues);
		}

		// propogate values through layers
		for (int layerIndex = 0; layerIndex < this.perceptrons.length - 1; layerIndex++) {
			Perceptron[] currentLayer = this.perceptrons[layerIndex];
			Perceptron[] nextLayer = this.perceptrons[layerIndex + 1];

			float[] currentLayerOutputs = new float[currentLayer.length];

			for (int currentLayerIndex = 0; currentLayerIndex < currentLayer.length; currentLayerIndex++) {
				currentLayerOutputs[currentLayerIndex] = currentLayer[currentLayerIndex].getOutput();
			}

			for (Perceptron nextLayerPerceptron : nextLayer) {
				nextLayerPerceptron.setInputValues(currentLayerOutputs);
			}
		}

		// get network output
		Perceptron[] lastLayer = this.perceptrons[this.perceptrons.length - 1];
		float[] output = new float[lastLayer.length];

		for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
			output[outputIndex] = lastLayer[outputIndex].getOutput();
		}

		return output;
	}

	public void train(float[] inputValues, float[] expectedOutput) {
		// calculate output error
		float[] actualOutput = this.evaluate(inputValues);
		Perceptron[] lastLayer = this.perceptrons[this.perceptrons.length - 1];

		for (int outputIndex = 0; outputIndex < lastLayer.length; outputIndex++) {
			float error = expectedOutput[outputIndex] - actualOutput[outputIndex];

			lastLayer[outputIndex].resetError();
			lastLayer[outputIndex].addError(error);
		}

		// propagate error backwards through layers
		for (int layerIndex = this.perceptrons.length - 1; layerIndex > 0; layerIndex--) {
			Perceptron[] currentLayer = this.perceptrons[layerIndex];
			Perceptron[] previousLayer = this.perceptrons[layerIndex - 1];

			for (int previousLayerIndex = 0; previousLayerIndex < previousLayer.length; previousLayerIndex++) {
				previousLayer[previousLayerIndex].resetError();

				for (Perceptron currentLayerPerceptron : currentLayer) {
					float errorContribution = currentLayerPerceptron.getErrorContribution(previousLayerIndex);

					previousLayer[previousLayerIndex].addError(errorContribution);
				}
			}
		}

		// train all perceptrons in network
		for (Perceptron[] layer : this.perceptrons) {
			for (Perceptron perceptron : layer) {
				perceptron.trainFromError(0.03f);
			}
		}
	}
}
