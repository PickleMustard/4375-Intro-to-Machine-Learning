#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <math.h>

using namespace std;

const int MAX_LEN = 2000;
const int TRAINING_SIZE = 800;

struct TitanicDataframe{
    vector<int> pclass;
    vector<bool> survived;
    vector<bool> sex;
    vector<int> age;
};

void printMatrix(vector<double> m, int rows, int col, bool cf){
    if(!cf){
        for(int i = 0; i <= col; i++){
            cout << "[," << ((i==0) ? "" : to_string(i)) << "]" << "\t";
        }
        cout << endl;
        for(int i = 0; i < rows; i++){
            cout << "[" << to_string(i+1) << ",]" << "\t";
            for(int j = 0; j < col; j++){
                cout << m.at(i*col +j) << "\t";
            }
            cout << endl;
        }
    } else {
        cout << "[,]\t[1]\t[0]" << endl;
        cout << "[1]\t"<<m.at(0)<<"\t"<<m.at(1)<<endl;
        cout << "[0]\t"<<m.at(2)<<"\t"<<m.at(3)<<endl;
    }
}

vector<double> matrixMultiplication(vector<double> v1, vector<double>v2){
    vector<double> new_matrix;
    for(int i = 0; i < v1.size(); i+=2){
        new_matrix.push_back(v1.at(i) * v2.at(0) + v1.at(i+1) * v2.at(1));
    }
    return new_matrix;
}

vector<double> matrixSubtraction(vector<bool> v1, vector<double> v2){
    vector<double> rtr;
    for(int i = 0; i < v1.size(); i++){
        rtr.push_back(v1.at(i) - v2.at(i));
    }
    return rtr;
}

vector<double> matrixTranspose(vector<double> v1, vector<double> v2){
    double w1 = 0, w2 = 0;
    for(int i = 0; i < v2.size(); i++){
        w1 += v1.at(i*2) * v2.at(i);
        w2 += v1.at(i*2+1) * v2.at(i);
    }
    vector<double> rtr = {w1, w2};
    return rtr;
}

vector<double> sigmoidFunction(vector<double> z){
    vector<double> rtr(z.size());
    for(int i = 0; i < z.size(); i++){
        rtr.at(i) = 1.0 / (1 + exp(-z.at(i)));
    }
    return rtr;
}

vector<double> predictionFunction(vector<double> z){
    vector<double> rtr(z.size());
    for(int i = 0; i < z.size(); i++){
        rtr.at(i) = exp(z.at(i)) / (1 + exp(z.at(i)));
    }
    return rtr;
}

void separateTestTrain(vector<int> pclass, vector<bool> survived, vector<bool> sex, vector<int> age, TitanicDataframe* train, TitanicDataframe* test, int size){
    //Separate 80% of values into train
    int randomIndex;
    vector<int> usedIndices(TRAINING_SIZE);
    for(int i = 0; i < TRAINING_SIZE; i++){
        do{
            randomIndex = rand() % size;
        }while(find(begin(usedIndices), end(usedIndices), randomIndex) != end(usedIndices));
        train->pclass.push_back(pclass.at(randomIndex));
        train->sex.push_back(sex.at(randomIndex));
        train->survived.push_back(survived.at(randomIndex));
        train->age.push_back(age.at(randomIndex));
        usedIndices.at(i) = randomIndex;
    }

    //If the i value doesn't exist in the usedIndices, add it to test
    for(int i = 0; i < size; i++){
        if(find(begin(usedIndices), end(usedIndices), i) == end(usedIndices)){
            test->pclass.push_back(pclass.at(i));
            test->sex.push_back(sex.at(i));
            test->survived.push_back(survived.at(i));
            test->age.push_back(age.at(i));
        }
    }
}

double findAccuracy(vector<double> cf){
    return (cf.at(0) + cf.at(3))/(cf.at(0)+cf.at(1)+cf.at(2)+cf.at(3));
}

double findSensitivity(vector<double> cf){
    return cf.at(0) / (cf.at(0) + cf.at(1));
}

double findSpecificity(vector<double> cf){
    return cf.at(3) / (cf.at(3) + cf.at(2));
}

void doLogisticRegression(TitanicDataframe* train, TitanicDataframe* test){
    double learning_rate = .001;
    vector<double> weights = {1,1};
    vector<double> data_matrix(train->sex.size() * 2);
    vector<double> test_matrix(test->sex.size() * 2);

    vector<double> prob_vector;
    vector<double> error;
    vector<double> transpose;

    vector<double> predicted;
    vector<double> probabilities;
    vector<bool> predictions;

    //Generate the data matrix, with 1's in the 1st column and the values of sex in the second column
    //Even indicies are 1s and odd are sex values
    for(int i = 0; i < data_matrix.size(); i++){
        if(!(i % 2)){
             data_matrix.at(i) = 1;
        }
        else {
             data_matrix.at(i) = train->sex.at(floor(i/2));
        }
    }
    //
    for(int i = 0; i < 500000; i++){
        prob_vector = sigmoidFunction(matrixMultiplication(data_matrix, weights));
        error = matrixSubtraction(train->survived, prob_vector);
        transpose = matrixTranspose(data_matrix, error);
        for(int j = 0; j < weights.size(); j++){
            weights.at(j) = weights.at(j) + learning_rate * transpose.at(j);
        }
    }

    cout << "Weights: " << weights.at(0) << " | " << weights.at(1) << endl;

    for(int i = 0; i < test_matrix.size(); i++){
        if(!(i % 2)){
             test_matrix.at(i) = 1;
        }
        else {
             test_matrix.at(i) = test->sex.at(floor(i/2));
        }
    }
    predicted = matrixMultiplication(test_matrix, weights);
    probabilities = predictionFunction(predicted);

    for(int i = 0; i < probabilities.size(); i++){
        if(probabilities.at(i) > .5) predictions.push_back(1);
        else predictions.push_back(0);
    }
    vector<double> confusion_matrix={0,0,0,0}; //TP, FN, FP, TN
    for(int i = 0; i < predictions.size(); i++){
        if(predictions.at(i) == test->survived.at(i)) {
            if(predictions.at(i) == 0) confusion_matrix.at(3)++;
            else confusion_matrix.at(0)++;
        } else{
            if(predictions.at(i) == 0) confusion_matrix.at(1)++;
            else confusion_matrix.at(2)++;
        }
    }
    cout << "---Confusion Matrix---" << endl;
    printMatrix(confusion_matrix, 2, 2, 1);
    cout << "---Accuracy---" << endl << findAccuracy(confusion_matrix) << endl;
    cout << "---Sensitivity---" << endl << findSensitivity(confusion_matrix) << endl;
    cout << "---Specificity---" << endl << findSpecificity(confusion_matrix) << endl;


}

int main(int argc, char** argv) {
    srand(1234);
    ifstream inFS; // Input File Stream
	string line;
	string pclass_in, survived_in, sex_in, age_in, unclassified_data_in;
    TitanicDataframe *train = new TitanicDataframe();
    TitanicDataframe *test = new TitanicDataframe();
    
	double covariance;
	vector<int> pclass(MAX_LEN);
	vector<bool> survived(MAX_LEN);
    vector<bool> sex(MAX_LEN);
    vector<int> age(MAX_LEN);
    vector<int> unclassified_data(MAX_LEN);
	
	// Try to open file
	cout << "Opening file titanic_project.csv" << endl;
	
	inFS.open("titanic_project.csv");
	if(!inFS.is_open()) {
		cout << "Could not open file titanic_project.csv." << endl;
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
        getline(inFS,unclassified_data_in, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');
		
        //unclassified_data.at(numObservations) = stof(unclassified_data_in);
		pclass.at(numObservations) = stof(pclass_in);
		survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

		
		numObservations++;
	}
	
	pclass.resize(numObservations);
	survived.resize(numObservations);
	sex.resize(numObservations);
    age.resize(numObservations);

    separateTestTrain(pclass, survived, sex, age, train, test, pclass.size());
    
    cout << "Size of training set: " << train->pclass.size() << endl;
    cout << "Size of test set: " << test->pclass.size() << endl;
	cout << "New length: " << pclass.size() << endl;

    doLogisticRegression(train, test);
	
	cout << "Closing file Titanic.csv" << endl;
	inFS.close(); // DOne with file so close iter_swap
}