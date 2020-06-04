/*_______________________________________________________________________________
# CE889 Neural Networks and Deep Learning | Ogulcan Ozer. | 11 December 2018
UNFINISHED. See lines 192-202-414.
_______________________________________________________________________________*/
/*-------------------------------------------------------------------------------
#	CE889 Left edge following multilayer perceptron using gradient descent,
with momentum.

Usage: Program has 2 modes.
1: Feed-forward only -> Looks for 'yWeights.txt' and 'hWeights.txt' in the
running directory. If they exist, assumes proper weights are provided and
starts running on feed-forward mode using the sonar reading as inputs and
provided files as weights of the network.

2: Training -> If 'yWeights.txt' and 'hWeights.txt' do not exist in the
running directory, looks for 'inputClean.txt' and 'targetClean.txt' If
they exsist, checks the dimensions of the data. Each newline in the
input file are assumed to be features, and each newline in the output file
are assumed to be predictions. According to the data dimensions, input and
output nodes are created. starts the training of the network.

Provide the files according to the desired output and run the executable.

-------------------------------------------------------------------------------*/
#include <limits>
#include <cstddef>
#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include <string>
#include<cmath>
#include <algorithm>
#include <random>
#include "Aria.h"

using namespace std;

/*-------------------------------------------------------------------------------
#	Variables and structs
-------------------------------------------------------------------------------*/
int HIDDEN = 5;//Number of hidden nodes
double LEARN = 0.0;//Learning rate of the NN
double ALPHA = 0.0;//Alpha value used in momentum
double LAMBDA = 0.5;//Lambda value used in logistic function.

vector<int> indexValues;//Index record keeping for data shuffle.
vector<double> shuffleTemp;//Temporary vector for creating new training data from shuffled indexes.
vector<vector<double>> dataInput;//Training data
vector<vector<double>> dataTarget;//Labels or targets for training data
vector<vector<double>> validateInput;//Validation data
vector<vector<double>> validateTarget;//Labels or targets for validation data
vector<vector<double>> testInput;//Test data
vector<vector<double>> testTarget;//Labels or targets for test data

//Minimum and maximum values for each input and target, to be used in normalization and un-normalization:
vector<double> inputMin;
vector<double> inputMax;
vector<double> targetMin;
vector<double> targetMax;

//Records of output errors and validation errors.
vector<vector<double>> eValidate;

//For recording minimum RMSE
double minRMSE = INT_MAX;//

//Data struct for a single input object:
typedef struct Data {

	vector<double> x; //Input values
	vector<double> y; //Output values
	vector<double> h; //Hidden values
	vector<double> d; //Target values/labels
	vector<double> e; //Errors
	vector<vector<double> > w; //Hidden-Output weights
	vector<vector<double> > wH; //Input-Hidden weights
	vector<vector<double> > wMoment; //Past delta weights
	vector<vector<double> > wHmoment;; //Past hidden delta weights

} Data;
/////////////////////////////////////////////////////////////////////////////////

/*-------------------------------------------------------------------------------
#	Function definitions
-------------------------------------------------------------------------------*/

//Read the sample data from a .txt file.
int read_data(string input, string target);

//Normalize the data for each input and target.
void normalizeData(void);

//UN-Normalize the data for each input and target.
double unNormalizeData(double d, int node);

//Returns the avarage rmse of all the outputs.
double get_rmse(vector<vector<double>> errorset);

//Seperate data as 70/15/15
void seperate_data(void);

//Initialize single Input with associated weights.
Data init_data(int inputNodes, int hiddenNodes, int outputNodes);

//Set the error values of a data object
void set_errors(Data *data);

//Run validation data on given weights, nested inside a data object.
double validate(Data *data);

//Run test data on given weights, nested inside a data object.
double test(Data *data);

//Feedforward operation.
void feed_forward(Data *data);

//Backpropogation operation.
void backpropogate(Data *data);

//Shuffle the training data
void shuffleData(void);




/*-------------------------------------------------------------------------------
#	Main Program
-------------------------------------------------------------------------------*/
int main(int argc, char **argv)
{
	ifstream File;
	Data minData;
	File.open("yWeights.txt");//Check if there are any weights provided.
	if (!File.is_open())///////
	{
		cout << "Weights could not be found. Training..." << endl;

		if (read_data("inputClean", "targetClean") == 1)//If there are no weights, start the training.
		{
			shuffleTemp.resize(dataInput[0].size());
			indexValues.resize(dataInput[0].size());
			for (int i = 0; i < dataInput[0].size(); i++) {
				indexValues[i] = i;
			}
			shuffleData();//Shuffle
			normalizeData();//Normalize
			seperate_data();//Seperate the data 70/15/15

							//Get the sizes of input and output nodes from the read data.
			int inValue = dataInput.size() + 1;
			int outValue = dataTarget.size();

			//Resize the shuffle and index arrays to seperated data.
			shuffleTemp.resize(dataInput[0].size());
			indexValues.resize(dataInput[0].size());
			for (int i = 0; i < dataInput[0].size(); i++) {
				indexValues[i] = i;
			}


			//Do- for different hidden nodes, alpha values and learning rates.
			for (HIDDEN = 3; HIDDEN < 9; HIDDEN++) {
				ALPHA = 0.1;
				for (int d = 0; d < 6; d++) {
					ALPHA = ALPHA + 0.01;
					LEARN = 0.4;
					for (int o = 0; o < 10; o++) {

						LEARN = LEARN + 0.05;


						//Initialize the weights of the data structs:
						Data test1 = init_data(inValue, HIDDEN, outValue);//Struct for feedforward input.
						minData = init_data(inValue, HIDDEN, outValue);//Struct for record keeping - minimum weights

						//Initialize the error history vector.
						eValidate.resize(2);
						eValidate[0].resize(0);//for
						eValidate[1].resize(0);
						minRMSE = INT_MAX;

						int tmpLEARN = LEARN;//Save original LEARN rate.

						//Do hundred epochs * 1 / (LEARN + 0.4) :
						for (int epoch = 0; epoch < 100 * (1 / (LEARN + 0.4)); epoch++)
						{

							
							double passErr = 0;
							for (int i = 0; i < static_cast<int>(dataInput[0].size()); i++)
							{
								//Initialize features and target values and biases.
								test1.x[0] = 1;
								test1.h[0] = 1;
								test1.x[1] = dataInput[0][i];/*Needs to be corrected for initializing the values one by one. Should be doing it dynamically*/
								test1.x[2] = dataInput[1][i];
								test1.d[0] = dataTarget[0][i];
								test1.d[1] = dataTarget[1][i];

								//Do one pass on training data:
								feed_forward(&test1);
								set_errors(&test1);
								backpropogate(&test1);

								passErr = passErr + ((test1.e[0] + test1.e[1]) / 2);//sum the errors

							}
							/**************END OF ONE PASS**************/


							//Get validation error for the last pass, and add it to the error records with the average output errors.
							passErr = passErr / dataInput[0].size();
							double val = validate(&test1);
							cout << "Validation error:" << val << endl;
							eValidate[0].push_back(val);
							eValidate[1].push_back(passErr);
							//////////////////////////////////////////

							if(minRMSE < 0.1)//Might be in the minimum zone, Lower LEARN rate.
							{
								LEARN = 0.05;
							}

							//Cutoff point
							if (((val - minRMSE > 0.18) && minRMSE < 0.1) || (val > 0.33)) {
								break;
							}
							//Save the weights with minimum rmse value in this pass
							if (minRMSE > val)
							{
								minRMSE = val;

								for (int y = 0; y < outValue; ++y)
								{
									for (int h = 0; h < HIDDEN; ++h)
									{
										minData.w[y][h] = test1.w[y][h];
									}
								}
								for (int h = 0; h < HIDDEN - 1; ++h)
								{
									for (int x = 0; x < inValue; ++x)
									{
										minData.wH[h][x] = test1.wH[h][x];
									}
								}
							}


							shuffleData();//Shuffle



						}
						/**************END OF HUNDRED EPOCHS**************/

						LEARN = tmpLEARN;//Set LEARN back to its original value.

						//Get Expected Error
						double testError = test(&minData);

						/*-------------------------------------------------------------------------------
						#	Output data to a .txt file for later evaluation
						-------------------------------------------------------------------------------*/
						std::ofstream ofs;
						ofs.open("test.txt", std::ofstream::out | std::ofstream::app);
						ofs << "Number of hidden nodes: " << HIDDEN - 1 << endl;
						ofs << "Lambda: " << LAMBDA << endl;
						ofs << "Alpha: " << ALPHA << endl;
						ofs << "Learning Rate: " << LEARN << endl;
						ofs << "Minimum RMSE for this run: " << minRMSE << endl;
						ofs << "Expected Error for this run: " << testError << endl;
						ofs << "ValErrors: ";
						for (int i = 0; i < eValidate[0].size(); i++)
						{
							ofs << eValidate[0][i] << ",";
						}
						ofs << endl;
						ofs << "Errors: ";
						for (int i = 0; i < eValidate[1].size(); i++)
						{
							ofs << eValidate[1][i] << ",";
						}
						ofs << endl;
						ofs.close();
						cout << "Validation min RMSE reached: " << minRMSE << endl;
						ofs.open("yWeights.txt", std::ofstream::out | std::ofstream::app);

						for (int y = 0; y < minData.w.size(); ++y)
						{
							for (int h = 0; h < minData.w[0].size(); ++h)
							{
								ofs << test1.w[y][h] << ",";
							}
							ofs << endl;
						}
						ofs.close();
						ofs.open("hWeights.txt", std::ofstream::out | std::ofstream::app);
						for (int h = 0; h < minData.wH.size(); ++h)
						{
							for (int x = 0; x < minData.wH[0].size(); ++x)
							{
								ofs << minData.wH[h][x] << ",";
							}
							ofs << endl;
						}
						ofs.close();

					}
				}
			}
		}
		else
		{
			cout << "Could not read the file/s.";
		}
	}
	else//Run the robot if there are weights provided.
	{

		vector<vector<double> > w; //Hidden-Output weights
		vector<vector<double> > wH; //Input-Hidden weights
		double temp = 0;
		vector<double> values;

		// Read the weights for output.
		while (File >> temp)
		{
			values.push_back(temp);
			if (File.peek() == ',')
				File.ignore();
			if ((File.peek() == '\n') || (File.peek() == '\r'))
			{
				w.push_back(values);
				values.clear();
			}

		}

		File.close();

		//Read the weights for hidden layer.
		File.open("hWeights.txt");
		values.clear();
		temp = 0;
		while (File >> temp)
		{
			values.push_back(temp);
			if (File.peek() == ',')
				File.ignore();
			if ((File.peek() == '\n') || (File.peek() == '\r'))
			{
				wH.push_back(values);
				values.clear();
			}

		}
		minData = init_data(wH[0].size() + 1, wH.size() + 1, w.size());
		for (int y = 0; y < w.size(); ++y)
		{
			for (int h = 0; h < w[0].size(); ++h)
			{
				minData.w[y][h] = w[y][h];
			}
		}
		for (int h = 0; h < wH.size(); ++h)
		{
			for (int x = 0; x < wH[0].size(); ++x)
			{
				minData.wH[h][x] = wH[h][x];
			}
		}
		File.close();
	}
	/*-------------------------------------------------------------------------------
	#	Main loop for robot feedforward
	-------------------------------------------------------------------------------*/

	//Initialize robot.
	Aria::init();
	ArRobot robot;
	ArPose pose;
	ArSensorReading *sonarSensor[8];

	//Read command line args.
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();
	argParser.addDefaultArgument("-connectLaser");
	ArRobotConnector robotConnector(&argParser, &robot);

	//Set normalization values
	inputMin.push_back(230.63);
	inputMax.push_back(1000);
	inputMin.push_back(673.37);
	inputMax.push_back(4835.3);
	targetMin.push_back(90);
	targetMax.push_back(300);
	targetMin.push_back(90);
	targetMax.push_back(300);
	//

	if (robotConnector.connectRobot()) {
		std::cout << "Robot Connected !" << std::endl;
	}

	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();

	//Feed-forward only loop.
	while (true)
	{
		//Read sonar values.
		double sonarRange[8];
		for (int i = 0; i < 8; i++)
		{
			sonarSensor[i] = robot.getSonarReading(i);
			sonarRange[i] = sonarSensor[i]->getRange();

		}
		//Set inputs of the struct from sonar readings and set biases.
		minData.x[0] = 1;
		minData.h[0] = 1;/*Needs to be corrected for initializing the values one by one. Should be doing it dynamically*/
		minData.x[1] = (sonarRange[0] - inputMin[0]) / (inputMax[0] - inputMin[0]);
		minData.x[2] = (sonarRange[1] - inputMin[1]) / (inputMax[1] - inputMin[1]);

		//Feed-forward
		feed_forward(&minData);

		//Set the speed of the robot.
		robot.setVel2(unNormalizeData(minData.y[0], 0), unNormalizeData(minData.y[1], 1));

		//Command line output for important information.
		cout << "SONAR 0 = " << sonarRange[0] << endl;
		cout << "SONAR 1 = " << sonarRange[1] << endl;
		cout << "OUTPUT 0 = " << minData.y[0] << endl;
		cout << "OUTPUT 1 = " << minData.y[1] << endl;
		cout << "UNORM 0 = " << unNormalizeData(minData.y[0], 0) << endl;
		cout << "UNORM 1 = " << unNormalizeData(minData.y[1], 1) << endl;

		//Sleep/Delay for 100 msec.
		ArUtil::sleep(100);

	}
	return 0;

}//End of robot feed-forward loop.


 /*-------------------------------------------------------------------------------
 #	Functions
 -------------------------------------------------------------------------------*/

 //Initialize single Input with associated weights.
Data init_data(int inputNodes, int hiddenNodes, int outputNodes)
{
	Data data;

	//Initialize according to the dimensions of the data.
	data.x.resize(inputNodes);
	data.y.resize(outputNodes);
	data.h.resize(hiddenNodes);
	data.d.resize(outputNodes);
	data.e.resize(outputNodes);

	srand(time(NULL));

	data.wHmoment.resize(hiddenNodes - 1);
	data.wH.resize(hiddenNodes - 1);
	for (int h = 0; h < hiddenNodes - 1; ++h) {
		data.wH[h].resize(inputNodes);
		data.wHmoment[h].resize(inputNodes);
	}
	for (int h = 0; h < hiddenNodes - 1; ++h) {
		for (int x = 0; x < inputNodes; ++x) {
			data.wH[h][x] = (((double)rand() / (RAND_MAX) * 2) - 1);

		}
	}
	data.wMoment.resize(outputNodes);
	data.w.resize(outputNodes);
	for (int y = 0; y < outputNodes; ++y) {
		data.w[y].resize(hiddenNodes);
		data.wMoment[y].resize(hiddenNodes);
	}
	for (int y = 0; y < outputNodes; ++y) {
		for (int h = 0; h < hiddenNodes; ++h) {
			data.w[y][h] = (((double)rand() / (RAND_MAX) * 2) - 1);

		}
	}
	return data;
}

//Feedforward operation.
void feed_forward(Data *data) {

	vector<double> vH;
	vH.resize(static_cast<int>(data->h.size()) - 1);//Hidden node sum.
	vector<double> vK;
	vK.resize(static_cast<int>(data->y.size()));//Output sum.



//Feed forward - 1 . (Finding each vH value, vH < hidden node size - 1, since h0 = bias)
	for (int h = 0; h < static_cast<int>(data->h.size() - 1); h++) {
		double tmp = 0;
		for (int x = 0; x < static_cast<int>(data->x.size()); x++) {

			tmp = tmp + data->x[x] * data->wH[h][x];

		}

		vH[h] = tmp;
	}

	//Apply logistic function to hidden layer sums.
	for (int h = 1; h < static_cast<int>(data->h.size()); h++) {

		data->h[h] = 1 / (1 + (exp(-(LAMBDA*(vH[h - 1])))));


	}
	//Feed forward - 2 . (Finding each vK value)
	for (int y = 0; y < static_cast<int>(data->y.size()); y++) {
		double tmp = 0;
		for (int h = 0; h < static_cast<int>(data->h.size()); h++) {

			tmp = tmp + data->h[h] * data->w[y][h];

		}
		vK[y] = tmp;

	}
	//Apply logistic function to output sums.
	for (int y = 0; y < static_cast<int>(data->y.size()); y++) {

		data->y[y] = 1 / (1 + (exp(-(LAMBDA*vK[y]))));;


	}

	//
	//End of Feedforward.
	return;
}

//Backpropogation
void backpropogate(Data *data) {

	//Initialize local gradients.
	vector<double> minH;
	minH.resize(static_cast<int>(data->h.size()));
	vector<double> minKsum;
	minKsum.resize(static_cast<int>(data->h.size()));
	vector<double> minK;
	minK.resize(static_cast<int>(data->y.size()));

	//Initialize delta weights.
	vector<vector<double>> deltaW;
	deltaW.resize(static_cast<int>(data->y.size()));
	for (int y = 0; y < static_cast<int>(data->y.size()); ++y) {
		deltaW[y].resize(data->h.size());
	}
	vector<vector<double>> deltaWh;
	deltaWh.resize(static_cast<int>(data->h.size() - 1));
	for (int h = 0; h < static_cast<int>(data->h.size() - 1); ++h) {
		deltaWh[h].resize(data->x.size());
	}

	//Calculate local minK gradient. (output gradient)
	for (int y = 0; y < static_cast<int>(data->y.size()); y++) {

		minK[y] = LAMBDA * data->y[y] * (1 - data->y[y]) * data->e[y];

	}
	//Sum of minK to be used in minH gradient.(hidden node gradient)
	for (int h = 0; h < static_cast<int>(data->h.size()); h++) {
		for (int y = 0; y < static_cast<int>(data->y.size()); y++) {

			minKsum[h] = minKsum[h] + (minK[y] * data->w[y][h]);

		}
	}
	//Calculate local minH gradient.
	for (int h = 1; h < static_cast<int>(data->h.size()); h++) {

		minH[h - 1] = LAMBDA * data->h[h] * (1 - data->h[h])*(minKsum[h - 1]);

	}
	//Calculate delta Wk for each weight, with moment.
	for (int y = 0; y < static_cast<int>(data->y.size()); y++) {
		for (int h = 0; h < static_cast<int>(data->h.size()); h++) {

			deltaW[y][h] = (LEARN * minK[y] * data->h[h]) + (ALPHA * data->wMoment[y][h]);
			data->wMoment[y][h] = deltaW[y][h];

		}
	}
	//Calculate delta Wh for each weight, with moment.
	for (int h = 0; h < static_cast<int>(data->h.size() - 1); h++) {
		for (int x = 0; x < static_cast<int>(data->x.size()); x++) {

			deltaWh[h][x] = (LEARN * minH[h] * data->x[x]) + (ALPHA * data->wHmoment[h][x]);
			data->wHmoment[h][x] = deltaWh[h][x];
		}
	}
	//Calculate new Wh weights.
	for (int h = 0; h < static_cast<int>(data->h.size() - 1); h++) {
		for (int x = 0; x < static_cast<int>(data->x.size()); x++)
		{
			data->wH[h][x] = data->wH[h][x] + deltaWh[h][x];

		}
	}
	//Calculate new Wk weights.
	for (int y = 0; y < static_cast<int>(data->y.size()); y++) {
		for (int h = 0; h < static_cast<int>(data->h.size()); h++) {

			data->w[y][h] = data->w[y][h] + deltaW[y][h];

		}
	}


}

//Shuffle the data 
void shuffleData(void)
{
	//Get a random number generator and shuffle the index numbers.
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(indexValues), std::end(indexValues), rng);


	//Shuffle both training and target values according to index.

	for (int i = 0; i < dataInput.size(); i++) {//Do for training.
		for (int j = 0; j < dataInput[0].size(); j++) {

			shuffleTemp[j] = dataInput[i][indexValues[j]];

		}
		dataInput[i] = shuffleTemp;



		shuffleTemp.clear();
		shuffleTemp.resize(dataInput[i].size());
	}
	for (int i = 0; i < dataTarget.size(); i++) {//Do for target.
		for (int j = 0; j < dataTarget[0].size(); j++) {

			shuffleTemp[j] = dataTarget[i][indexValues[j]];

		}
		dataTarget[i] = shuffleTemp;
		shuffleTemp.clear();
		shuffleTemp.resize(dataInput[i].size());
	}
}

//Seperate function for setting errors, since feedforward will also be used in real life, without target values to evaluate an error.
void set_errors(Data *data)
{

	for (int y = 0; y < static_cast<int>(data->y.size()); y++) {

		data->e[y] = data->d[y] - data->y[y];

	}

}

//Run validation on current weights.
double validate(Data  *data)
{
	vector<vector<double>> errors;
	errors.resize(data->y.size() * 2);
	for (int y = 0; y < data->y.size() * 2; y++) {
		errors[y].resize(validateTarget[0].size());
	}
	//Validate for each data in validation set.
	for (int v = 0; v < validateInput[0].size(); v++) {
		for (int i = 0; i < validateInput.size(); i++) {
			data->x[i] = validateInput[i][v];
		}
		for (int o = 0; o < validateTarget.size(); o++) {
			data->d[o] = validateInput[o][v];
		}

		feed_forward(data);
		set_errors(data);

		for (int o = 0; o < data->y.size() * 2; o = o + 2) {
			errors[o][v] = data->y[o / 2];
			errors[o + 1][v] = data->d[o / 2];
		}
	}
	return get_rmse(errors);
}

//Run test on current weights.
double test(Data *data)
{
	vector<vector<double>> errors;
	errors.resize(data->y.size() * 2);
	for (int y = 0; y < data->y.size() * 2; y++) {
		errors[y].resize(testTarget[0].size());
	}
	//Test for each data in test.
	for (int v = 0; v < testInput[0].size(); v++) {
		for (int i = 0; i < testInput.size(); i++) {
			data->x[i] = testInput[i][v];
		}
		for (int o = 0; o < testTarget.size(); o++) {
			data->d[o] = testInput[o][v];
		}

		feed_forward(data);
		set_errors(data);

		for (int o = 0; o < data->y.size() * 2; o = o + 2) {
			errors[o][v] = data->y[o / 2];
			errors[o + 1][v] = data->d[o / 2];
		}
	}
	return get_rmse(errors);

}

//Read the data from .txt files and get the min and max values at the same time, for normalization.
int read_data(string input, string target) {

	double temp = 0;
	int c = 0;
	vector<double> values;


	ifstream File;
	File.open(input + ".txt");
	if (!File.is_open())
	{
		cout << "It failed" << endl;
		return 0;
	}
	double min = 9999999.0;
	double max = numeric_limits<double>::lowest();
	while (File >> temp)
	{
		if (temp <= min)
			min = temp;
		if (temp >= max)
			max = temp;
		values.push_back(temp);
		if (File.peek() == ',')
			File.ignore();
		if ((File.peek() == '\n') || (File.peek() == '\r')) {

			dataInput.push_back(values);
			inputMin.push_back(min);//Keep record of min and max values while reading
			inputMax.push_back(max);//
			min = 9999999.0;
			max = numeric_limits<double>::lowest();
			values.clear();
		}


	}
	File.close();// End of reading input file.
	temp = 0;
	c = 0;
	values.clear();
	File.open(target + ".txt");
	if (!File.is_open())
	{
		cout << "It failed" << endl;
		return 0;
	}

	min = 9999999.0;
	max = numeric_limits<double>::lowest();
	while (File >> temp)
	{
		if (temp <= min)
			min = temp;
		if (temp >= max)
			max = temp;
		values.push_back(temp);
		if (File.peek() == ',')
			File.ignore();
		if ((File.peek() == '\n') || (File.peek() == '\r')) {

			dataTarget.push_back(values);
			targetMin.push_back(min);//Keep record of min and max values while reading.
			targetMax.push_back(max);//
			min = 9999999.9;
			max = numeric_limits<double>::lowest();
			values.clear();
		}


	}
	File.close();//End of reading output file.

	return 1;
	//
	// End of reading training-target data from the file.

}

//Normalize the data for each input and target.
void normalizeData(void) {

	for (int i = 0; i < static_cast<int>(dataInput.size()); i++) {
		for (int j = 0; j < static_cast<int>(dataInput[i].size()); j++) {

			dataInput[i][j] = (dataInput[i][j] - inputMin[i]) / (inputMax[i] - inputMin[i]);

		}
	}
	for (int i = 0; i < static_cast<int>(dataTarget.size()); i++) {
		for (int j = 0; j < static_cast<int>(dataTarget[i].size()); j++) {

			dataTarget[i][j] = (dataTarget[i][j] - targetMin[i]) / (targetMax[i] - targetMin[i]);

		}
	}


}

//UnNormalize the given value according to its output.
double unNormalizeData(double d, int node)
{
	return (targetMin[node] + (d * (targetMax[node] - targetMin[node])));
}

//Returns the avarage rmse of all the outputs.
double get_rmse(vector<vector<double>> errorset)
{
	int nsize = static_cast<int>(errorset.size()) / 2;
	vector<double> sum;
	sum.resize(nsize);
	//Rmse for each output.
	for (int y = 0; y < static_cast<int>(errorset.size()); y = y + 2) {
		for (int e = 0; e < static_cast<int>(errorset[y].size()); e++) {

			sum[y / 2] = sum[y / 2] + pow((errorset[y][e] - errorset[y + 1][e]), 2);

		}
		sum[y / 2] = sqrt(sum[y / 2] / errorset[y].size());
	}
	double avg = 0;
	//Average rmse for all outputs.
	for (int s = 0; s < nsize; s++) {
		avg = avg + sum[s];
	}
	avg = avg / nsize;

	return avg;
}

//Seperate data as 70/15/15
void seperate_data(void)
{

	validateInput.resize(dataInput.size());
	testInput.resize(dataInput.size());
	validateTarget.resize(dataTarget.size());
	testTarget.resize(dataTarget.size());

	int tr = dataInput[0].size() / 100 * 70;//Divide 70
	int vl = dataInput[0].size() / 100 * 15;//		 15	
	int ts = dataInput[0].size() - (tr + vl);//      and rest

	for (int i = 0; i < dataInput.size(); i++) {
		validateInput[i].resize(vl);
		testInput[i].resize(ts);
	}
	for (int t = 0; t < dataTarget.size(); t++) {
		validateTarget[t].resize(vl);
		testTarget[t].resize(ts);
	}

	for (int i = 0; i < dataInput.size(); i++) {
		for (int j = tr; j < dataInput[i].size(); j++) {

			if (j < (tr + vl))
				validateInput[i][j - tr] = dataInput[i][j];
			if ((j >= (tr + vl)) && (j < (tr + vl + ts + 1)))
				testInput[i][j - (tr + vl)] = dataInput[i][j];
		}
		dataInput[i].resize(tr);
	}

	for (int i = 0; i < dataTarget.size(); i++) {
		for (int j = tr; j < dataTarget[i].size(); j++) {

			if (j < (tr + vl))
				validateTarget[i][j - tr] = dataTarget[i][j];
			if ((j >= (tr + vl)) && (j < (tr + vl + ts + 1)))
				testTarget[i][j - (tr + vl)] = dataTarget[i][j];
		}
		dataTarget[i].resize(tr);
	}


	return;
}
/*-------------------------------------------------------------------------------
#	End of program
-------------------------------------------------------------------------------*/
