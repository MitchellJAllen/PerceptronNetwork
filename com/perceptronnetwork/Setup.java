package com.perceptronnetwork;

import com.perceptronnetwork.network.LogisticPerceptronNetwork;
import com.perceptronnetwork.perceptron.LogisticPerceptron;

import java.util.Random;

public class Setup {
	private static float calculateLogistic(float weightedSum) {
		return (float)(1 / (1 + Math.exp(-weightedSum)));
	}

	public static void testLogisticPerceptron() {
		LogisticPerceptron test = new LogisticPerceptron(2);
		Random random = new Random();

		float[] desiredWeights = {0.1f, 0.8f};

		for (int sampleIndex = 0; sampleIndex < 10000; sampleIndex++) {
			float x = (float)(2 * random.nextDouble() - 1);
			float y = (float)(2 * random.nextDouble() - 1);

			float expected = calculateLogistic(desiredWeights[0] * x + desiredWeights[1] * y);

			test.setInputValue(0, x);
			test.setInputValue(1, y);

			test.resetError();
			test.addError(expected - test.getOutput());

			if (sampleIndex == 9999) {
				System.out.println("expected: " + expected);
				System.out.println("actual: " + test.getOutput());

				System.out.println("weight 0: " + test.getWeight(0));
				System.out.println("weight 1: " + test.getWeight(1));
			}

			test.trainFromError(0.03f);
		}
	}

	public static void testLogisticPerceptronNetwork() {
		int[] hiddenDimensions = {80};
		LogisticPerceptronNetwork test = new LogisticPerceptronNetwork(2, hiddenDimensions, 2);
		Random random = new Random();

		float[] inputValues = {0, 0};
		float[] expectedOutput = {0, 0};

		for (int sampleIndex = 0; sampleIndex < 20000; sampleIndex++) {
			inputValues[0] = (float)random.nextDouble();
			inputValues[1] = (float)random.nextDouble();

			expectedOutput[0] = (inputValues[0] + inputValues[1]) / 2; // average
			expectedOutput[1] = inputValues[0] * inputValues[1]; // product

			test.train(inputValues, expectedOutput);
		}

		inputValues[0] = 0.5f;
		inputValues[1] = 0.7f;

		float[] output = test.evaluate(inputValues);

		System.out.println("expected result: [0.5, 0.7] becomes [0.6, 0.35]");
		System.out.println("actual result: [" + output[0] + ", " + output[1] + "]");
	}

	public static void main(String[] args) {
		testLogisticPerceptron();
		testLogisticPerceptronNetwork();
	}
}
