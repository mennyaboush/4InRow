
import java.text.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

import org.omg.Messaging.SyncScopeHelper;

public class NeuralNetwork {
	static {
		Locale.setDefault(Locale.ENGLISH);
	}

	final static int MAX_RUNS = 50000;
	final static double MIN_ERROR_CONDITION = 4;
	final boolean isTrained = false;
	final DecimalFormat df;
	final Random rand = new Random();

	final ArrayList<ArrayList<Neuron>> hiddenLayer = new ArrayList<ArrayList<Neuron>>();

	final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();

	final Neuron bias = new Neuron();
	final int randomWeightMultiplier = 1;

	final double epsilon = 0.00001;

	final double learningRate = 0.9f;
	final double momentum = 0.5f;

	int maxDepth = 4;
	// Returns the next step that the machine needs to do according to the
	// minmax algorithm
	MinimaxAI OPlayer = new MinimaxAI(maxDepth, Board.O);

	// Inputs for 4 in a row problem
	private int inputs[][] = new int[Board.ROWS][Board.COLUMNS];

	// Corresponding outputs, xor training data
	final double expectedOutputs[] = new double[Board.COLUMNS];
	// double resultOutputs [] = new double[Field.COLUMNS]; // dummy init
	double output[];

	// for weight update all
	final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();

	public void InitHidden(int numOfHidden) {
		for (int i = 0; i < numOfHidden; i++) {
			hiddenLayer.add(new ArrayList<Neuron>());
		}

	}

	/**
	 * Create all neurons and connections Connections are created in the neuron
	 * class
	 */
	public void InitConnction(int layerSize, int outputSize,int numOfHidden) {
		for (int j = 0; j < layerSize; j++) {
			Neuron neuron = new Neuron();
			inputLayer.add(neuron);
		}
		for (int j = 0; j < numOfHidden; j++) {
			for (int i = 0; i < layerSize; i++) {
				Neuron neuron = new Neuron();
				if (j == 0)
					neuron.addInConnectionsS(inputLayer);
				else
					neuron.addInConnectionsS(hiddenLayer.get(j - 1));
				neuron.addBiasConnection(bias);
				hiddenLayer.get(j).add(neuron);
			}
		}
		for (int j = 0; j < outputSize; j++) {
			Neuron neuron = new Neuron();
			neuron.addInConnectionsS(hiddenLayer.get(hiddenLayer.size() - 1));
			neuron.addBiasConnection(bias);
			outputLayer.add(neuron);
		}
	}

	public void Initweight(int layerSize, int outputSize, int numOfHidden) {
		for (int i = 0; i < numOfHidden; i++) {
			for (Neuron neuron : hiddenLayer.get(i)) {
				ArrayList<Connection> connections = neuron.getAllInConnections();
				for (Connection conn : connections) {
					double newWeight = getRandom();
					conn.setWeight(newWeight);
				}
			}
		}
		for (Neuron neuron : outputLayer) {
			ArrayList<Connection> connections = neuron.getAllInConnections();
			for (Connection conn : connections) {
				double newWeight = getRandom();
				conn.setWeight(newWeight);
			}
		}
		// reset id counters
		Neuron.counter = 0;
		Connection.counter = 0;

		if (isTrained) {
			trainedWeights();
			updateAllWeights();
		}
	}

	double getRandom() {
		Double x=randomWeightMultiplier * (rand.nextDouble() * 2 - 1);
		return x; // [-1;1[
	}

	String weightKey(int neuronId, int conId) {
		return "N" + neuronId + "_C" + conId;
	}

	public NeuralNetwork(int layerSize, int outputSize, int numOfHidden) {
		initField();
		InitHidden(numOfHidden);
		InitConnction(layerSize, outputSize, numOfHidden);
		Initweight(layerSize, outputSize, numOfHidden);

		df = new DecimalFormat("#.0#");
	}

	// random

	private void initField() {

		for (int i = 0; i < Board.ROWS; i++) {
			for (int j = 0; j < Board.COLUMNS; j++) {
				inputs[i][j] = Board.EMPTY;
			}
		}
	}

	public void newField(Board connect4) {
		inputs = connect4.getGameBoard();
	}

	/**
	 * 
	 * @param inputs
	 *            There is equally many neurons in the input layer as there are
	 *            in input variables
	 */
	public void setInput(int inputs[][]) {
		for (int i = 0; i < Board.ROWS; i++) {
			for (int j = 0; j < Board.COLUMNS; j++)
				inputLayer.get(i).setOutput(inputs[i][j]);
		}
	}

	public double[] getOutput() {
		double[] outputs = new double[outputLayer.size()];
		for (int i = 0; i < outputLayer.size(); i++)
			outputs[i] = outputLayer.get(i).getOutput();
		return outputs;
	}

	/**
	 * Calculate the output of the neural network based on the input The forward
	 * operation
	 */
	public void activate() {
		for (int i = 0; i < hiddenLayer.size(); i++)
			for (Neuron n : hiddenLayer.get(i))
				n.calculateOutput();

		for (Neuron n : outputLayer)
			n.calculateOutput();
	}

	/**
	 * all output propagate back
	 * 
	 * @param expectedOutput
	 *            first calculate the partial derivative of the error with
	 *            respect to each of the weight leading into the output neurons
	 *            bias is also updated here
	 */
	public void applyBackpropagation(double expectedOutput[]) {

		// error check, normalize value ]0;1[
		for (int i = 0; i < expectedOutput.length; i++) {
			double d = expectedOutput[i];
			if (d < 0 || d > 1) {
				if (d < 0)
					expectedOutput[i] = 0 + epsilon;
				else
					expectedOutput[i] = 1 - epsilon;
			}
		}

		int i = 0;
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double ak = n.getOutput();
				double ai = con.leftNeuron.getOutput();
				double desiredOutput = expectedOutput[i];
				double partialDerivative = -ak * (1 - ak) * ai * (desiredOutput - ak);
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
			i++;
		}
		for (Neuron n : hiddenLayer.get(hiddenLayer.size()-1)){
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double aj = n.getOutput();
				double ai = con.leftNeuron.getOutput();
				double sumKoutputs = 0;
				int j = 0;
				for (Neuron out_neu : outputLayer) {
					double wjk = out_neu.getConnection(n.id).getWeight();
					double desiredOutput = outputLayer.get(j).getOutput();
					double ak = out_neu.getOutput();
					j++;
					sumKoutputs = sumKoutputs + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
				}

				double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
		}
		for (i = hiddenLayer.size()-2; i > 0; i--){
			for (Neuron n :hiddenLayer.get(i)) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					double aj = n.getOutput();
					double ai = con.leftNeuron.getOutput();
					double sumKoutputs = 0;
					int j = 0;
					for (Neuron out_neu : hiddenLayer.get(i+1)) {
						double wjk = out_neu.getConnection(n.id).getWeight();
						double desiredOutput = hiddenLayer.get(i).get(j).getOutput();
						double ak = out_neu.getOutput();
						j++;
						sumKoutputs = sumKoutputs + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
					}

					double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
					double deltaWeight = -learningRate * partialDerivative;
					double newWeight = con.getWeight() + deltaWeight;
					con.setDeltaWeight(deltaWeight);
					con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
				}
			}
		}
		for (Neuron n : inputLayer){
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double aj = n.getOutput();
				double ai = con.leftNeuron.getOutput();
				double sumKoutputs = 0;
				int j = 0;
				for (Neuron out_neu : hiddenLayer.get(1)) {
					double wjk = out_neu.getConnection(n.id).getWeight();
					double desiredOutput = hiddenLayer.get(0).get(j).getOutput();
					double ak = out_neu.getOutput();
					j++;
					sumKoutputs = sumKoutputs + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
				}

				double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
		}
	}

	public int run(Board connect4) {
		int i;
		// Train neural network until minError reached or maxSteps exceeded
		double error = 5;
		newField(connect4);
		setInput(inputs);
		getexpectedOutputs(expectedOutputs, connect4);
		for (i = 0; i < MAX_RUNS && error > MIN_ERROR_CONDITION; i++) {
			error = 0;
			activate();
			for (int p = 0; p < inputs.length; p++) {
				output = getOutput();
				for (int j = 0; j < expectedOutputs.length; j++) {
					double err = Math.pow(output[j] - expectedOutputs[p], 2);
					error += err;
				}
				applyBackpropagation(expectedOutputs);
			}
			error = error / 2;
			System.out.println(error + "\n");
		}

		// printResult();
		System.out.println("Sum of squared errors = " + error);
		System.out.println("##### EPOCH " + i + "\n");
		if (i == MAX_RUNS) {
			System.out.println("!Error training try again");
		}
		return getResult();
	}

	private int getResult() {
		double max = output[0];
		int index = 0;
		for (int i = 1; i < Board.COLUMNS; i++) {
			if (max < output[i]) {
				max = output[i];
				index = i;
			}
		}
		return index;
	}

	private void getexpectedOutputs(double[] arr, Board connect4) {
		int temp = OPlayer.MiniMax(connect4).getCol();
		for (int i = 0; i < Board.COLUMNS; i++)
			if (i == temp)
				arr[i] = 1;
			else
				arr[i] = 0;
	}

	//	void printResult() {
	//		System.out.println("NN training");
	//		for (int p = 0; p < inputs.length; p++) {
	//			System.out.print("INPUTS: ");
	//			for (int x = 0; x < layers[0]; x++) {
	//				System.out.print(inputs[p][x] + " ");
	//			}
	//
	//			System.out.print("EXPECTED: ");
	//			for (int x = 0; x < layers[2]; x++) {
	//				// System.out.print(expectedOutputs[p][x] + " ");
	//			}
	//
	//			System.out.print("ACTUAL: ");
	//			for (int x = 0; x < layers[2]; x++) {
	//				System.out.print(output[x] + " ");
	//			}
	//			System.out.println();
	//		}
	//		System.out.println();
	//	}
	//
	//	String weightKey(int neuronId, int conId) {
	//		return "N" + neuronId + "_C" + conId;
	//	}
	//
	//	/**
	//	 * Take from hash table and put into all weights
	//	 */
	public void updateAllWeights() {
		// update weights for the output layer
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				String key = weightKey(n.id, con.id);
				double newWeight = weightUpdate.get(key);
				con.setWeight(newWeight);
			}
		}
		for (int i = hiddenLayer.size()-1 ; i >= 0; i++){
			for (Neuron n : hiddenLayer.get(i)) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					String key = weightKey(n.id, con.id);
					double newWeight = weightUpdate.get(key);
					con.setWeight(newWeight);
				}
			}
		}
	}


	// trained data
	void trainedWeights() {
		weightUpdate.clear();
		weightUpdate.put(weightKey(3, 0), 1.03);
		weightUpdate.put(weightKey(3, 1), 1.13);
		weightUpdate.put(weightKey(3, 2), -.97);
		weightUpdate.put(weightKey(4, 3), 7.24);
		weightUpdate.put(weightKey(4, 4), -3.71);
		weightUpdate.put(weightKey(4, 5), -.51);
		weightUpdate.put(weightKey(5, 6), -3.28);
		weightUpdate.put(weightKey(5, 7), 7.29);
		weightUpdate.put(weightKey(5, 8), -.05);
		weightUpdate.put(weightKey(6, 9), 5.86);
		weightUpdate.put(weightKey(6, 10), 6.03);
		weightUpdate.put(weightKey(6, 11), .71);
		weightUpdate.put(weightKey(7, 12), 2.19);
		weightUpdate.put(weightKey(7, 13), -8.82);
		weightUpdate.put(weightKey(7, 14), -8.84);
		weightUpdate.put(weightKey(7, 15), 11.81);
		weightUpdate.put(weightKey(7, 16), .44);
	}

	public void printWeightUpdate() {
		System.out.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
		// weights for the hidden layer
		for (int i = 0 ; i >= hiddenLayer.size(); i++){
			for (Neuron n : hiddenLayer.get(i)) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					String w = df.format(con.getWeight());
					System.out.println("weightUpdate.put(weightKey(" + n.id + ", " + con.id + "), " + w + ");");
				}
			}
		}
		// weights for the output layer
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				String w = df.format(con.getWeight());
				System.out.println("weightUpdate.put(weightKey(" + n.id + ", " + con.id + "), " + w + ");");
			}
		}
		System.out.println();
	}

	public void printAllWeights() {
		System.out.println("printAllWeights");
		// weights for the hidden layer
		for (int i = 0 ; i >= hiddenLayer.size(); i++){
			for (Neuron n : hiddenLayer.get(i)) {
				ArrayList<Connection> connections = n.getAllInConnections();
				for (Connection con : connections) {
					double w = con.getWeight();
					System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
				}
			}
		}
		// weights for the output layer
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double w = con.getWeight();
				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
			}
		}
		System.out.println();
	}

	public Move NeuralNetworkMove(Board connect4) {
		int col = run(connect4);
		int row = connect4.getRowPosition(col);
		return new Move(row, col);
	}
}
