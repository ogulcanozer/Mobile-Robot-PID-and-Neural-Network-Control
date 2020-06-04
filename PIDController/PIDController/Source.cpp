/*_______________________________________________________________________________
#	CE801 Intelligent Systems and Robotics | Ogulcan Ozer. | December 2018
_______________________________________________________________________________*/
/*-------------------------------------------------------------------------------
# CE801 Assignment, Right edge following PID controller for ARIA ROBOT

Usage: Upload the executable to the robot and run.
-------------------------------------------------------------------------------*/
#include <iostream>
#include "Aria.h"

/*-------------------------------------------------------------------------------
#	Constants and variables used in PID calculation
-------------------------------------------------------------------------------*/

double const baseVal = 100;//Base speed of the robot.
double const dWheel = 260;//Distance between two wheels .
double const d_distance7 = 475;//Minimum desired distance for sensor 7.
double const d_distance6 = 650;//Minimum desired distance for sensor 6.
double const d_distance5 = 1080;//Minimum desired distance for sensor 5.

//PID variables
double pid = 0;//PID OUT
double p = 0.13, in = 0.00000006, d = 0.000000000000000000001;//coefficients for the proportional, integral, and derivative terms.
double cErr = 0, iErr = 0, pErr = 0, dErr = 0.0;//current, integral, and derivative errors.

/*-------------------------------------------------------------------------------
#	Function definitions
-------------------------------------------------------------------------------*/
//Function for PID error calculation .
double pidError(double current);

//Function for finding the minimum in a given array with its size.
int getMinIndex(double array[], int size);

/*-------------------------------------------------------------------------------
#	 Main Program
-------------------------------------------------------------------------------*/
int main(int argc, char **argv)
{
	//ARIA Robot initialization.
	Aria::init();
	ArRobot robot;
	ArPose pose;
	ArSensorReading *sonarSensor[8];
	//Parse command line args.
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();
	ArRobotConnector robotConnector(&argParser, &robot);

	if (robotConnector.connectRobot()) {
		std::cout << "Robot Connected !" << std::endl;
	}
	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();


	double sonArr[]{ 0.0, 0.0, 0.0 };// Array for holding 7th, 6th and 5th sonar readings .
	int arrMin = 0;//Index of the minimum of sonars 7,6,5 .

	while (true)
	{
		//Get sonar readings.
		double sonarRange[8];
		for (int i = 0; i < 8; i++)
		{
			sonarSensor[i] = robot.getSonarReading(i);
			sonarRange[i] = sonarSensor[i]->getRange();

		}
		sonArr[0] = sonarRange[5];
		sonArr[1] = sonarRange[6];
		sonArr[2] = sonarRange[7];
		arrMin = getMinIndex(sonArr, 3);//Get the minimum sonar.

		//Command line output for important values.
		std::cout << "SONAR 7 :" << sonarRange[7] << std::endl;
		std::cout << "-----------------" << std::endl;
		std::cout << "SONAR 6 :" << sonarRange[6] << std::endl;
		std::cout << "-----------------" << std::endl;
		std::cout << "SONAR 5 :" << sonarRange[5] << std::endl;
		std::cout << "-----------------" << std::endl;
		std::cout << "SONAR MIN :" << sonArr[arrMin] << std::endl;
		std::cout << "-----------------" << std::endl;
		std::cout << "CURRENT ERROR :" << cErr << std::endl;
		std::cout << "-----------------" << std::endl;
		std::cout << "INTEGRAL ERROR :" << iErr << std::endl;
		std::cout << "-----------------" << std::endl;
		std::cout << "DERIVATIVE ERROR :" << dErr << std::endl;
		std::cout << "-----------------" << std::endl;

		//Calculate PID error according to the minimum reading.
		if (arrMin == 2) {
			pid = pidError(d_distance7 - sonArr[arrMin]);
		}
		else if (arrMin == 1) {
			pid = pidError(d_distance6 - sonArr[arrMin]);
		}
		else {
			pid = pidError(d_distance5 - sonArr[arrMin]);
		}


		//Slow down the robot if all the sensors are reading over 5000 - to prevent sudden movements.
		if (sonArr[arrMin] >= 5000)
		{
			pid = 2;

		}

		//Calculate PID output.
		pid = -2 * pid / 260;

		//Calculate speed for each wheel from PID value .
		int wLeft = baseVal + pid * dWheel / 2;
		int wRight = baseVal - pid * dWheel / 2;

		//Set the speed of the robot.
		robot.setVel2(wLeft, wRight);

		//Sleep/Delay 100 msec.
		ArUtil::sleep(100);


	}

	robot.lock();
	robot.stop();
	robot.unlock();

	//Terminate.
	Aria::exit();

	return 0;
}


/*-------------------------------------------------------------------------------
#	Functions
-------------------------------------------------------------------------------*/

//Function for finding the index of the minimum in a given array with its size.
int getMinIndex(double array[], int size)
{
	int index = 0;

	for (int i = 1; i < size; i++)
	{
		if (array[i] < array[index])
			index = i;
	}

	return index;
}

//Function for PID error calculation.
double pidError(double current) {

	cErr = current;

	//Prevent the integral error from exploding.
	if (iErr <= 2000 && iErr >= -2000)
		iErr = iErr + cErr;//Calculate.
	else if (iErr < -2000)
		iErr = -2000;
	else
		iErr = 2000;

	//Calculate derivative error.
	dErr = cErr - pErr;

	//Calculate proportional error.
	pErr = cErr;

	return  p * cErr + in * iErr + d * dErr;
}

