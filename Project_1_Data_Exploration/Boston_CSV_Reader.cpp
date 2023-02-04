#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <math.h>

using namespace std;



double findSum(vector<double> value) {
	double sum = 0;
	for(int i = 0; i < value.size(); i++){
		sum += value[i];
	}
	return sum;
}

double findMean(vector<double> value){
	return findSum(value) / value.size();
}

double findVariance(vector<double> value){
	double variance;
	double sum = 0;
	double mean = findMean(value);
	for(int i = 0; i < value.size(); i++){
		sum += pow((value[i] - mean),2);
	}
	variance = sum / (value.size()-1);
	return variance;
}

double findStandardDeviation(vector<double> value){
	double variance = findVariance(value);
	double stdev = sqrt(variance);
	return stdev;
}

//If the size is even, then take the average of 2 values in the middle (Size /2  + size/2 + 1) / 2
//Otherwise, find ceiling of size/2
double findMedian(vector<double> value){
	double returnValue;
	sort(value.begin(), value.end() + 1);
	if(value.size() % 2){
		int middle = ceil(value.size() / 2);
		returnValue = value[middle];
	} else {
		int lowerMiddle = value.size() / 2;
		returnValue = (value[lowerMiddle] + value[lowerMiddle + 1]) / 2;
	}
	return returnValue;
}

vector<double> findRange(vector<double> value){
	double maxValue = value[0];
	double minValue = value[0];
	for(int i = 0; i < value.size(); i++){
		if(value[i] > maxValue) maxValue = value[i];
		if(value[i] < minValue) minValue = value[i];
	}
	vector<double> range;
	range.insert(range.end(), minValue);
	range.insert(range.end(), maxValue);
	return range;
}

double covar(vector<double> rm, vector<double> medv){
	//Sum of (x[i] - xavg)(y[i] - yavg) / n - 1
	double sum = 0;
	double rmMean = findMean(rm);
	double medvMean = findMean(medv);
	double covariance;
	for(int i = 0; i < rm.size(); i++){
		sum += (rm[i] - rmMean)*(medv[i] - medvMean);
	}
	covariance = sum / (rm.size() - 1);
	return covariance;
}

double cor(vector<double> rm, vector<double> medv, double covariance){
	double rmStDev = findStandardDeviation(rm);
	double medvStDev = findStandardDeviation(medv);
	double correlation = covariance / (rmStDev * medvStDev);
	return correlation;
}

void print_stats(vector<double> value){
	//Takes the vector value provided from the csv
	//Finds the Sum, Mean, Median, Range of vector values
	//Prints found values
	double sum, mean, median;
	vector<double> valueRange;
	sum = findSum(value);
	mean = findMean(value);
	median = findMedian(value);
	valueRange = findRange(value);

	cout << "Printing Stats: " << endl;
	cout << "Sum: " << sum << endl;
	cout << "Mean: " << mean << endl;
	cout << "Median: " << median << endl;
	cout << "Range: " << valueRange.front() << ", " << valueRange.back() << endl;
}

int main(int argc, char** argv) {
	ifstream inFS; // Input File Stream
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	double covariance;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);
	
	// Try to open file
	cout << "Opening file Boston.csv" << endl;
	
	inFS.open("Boston.csv");
	if(!inFS.is_open()) {
		cout << "Could not open file Boston.csv." << endl;
		return 1; // 1 indicates error
	}
	
	// Can now use inFS stream like cin stream
	// Boston.csv should contain 2 doubles
	
	cout << "Reading line 1" << endl;
	getline(inFS, line);
	
	// echo heading
	cout << "Heading: " << line << endl;
	
	int numObservations = 0;
	while(inFS.good()){
		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');
		
		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);
		
		numObservations++;
	}
	
	rm.resize(numObservations);
	medv.resize(numObservations);
	
	cout << "New length: " << rm.size() << endl;
	
	cout << "Closing file Boston.csv" << endl;
	inFS.close(); // DOne with file so close iter_swap
	
	cout << "Num of records: " << numObservations << endl;
	
	cout << "\nStats for rm" << endl;
	print_stats(rm);
	
	cout << "\nStats for medv" << endl;
	print_stats(medv);
	
	covariance = covar(rm,medv);
	cout << "\nCovariance = " << covariance << endl;
	
	cout << "\nCorrelation = " << cor(rm, medv, covariance) << endl;
	
	cout << "\nProgram Terminated";
	
	return 0;
}

