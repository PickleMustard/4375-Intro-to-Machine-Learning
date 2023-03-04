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

int findNumberSurvived(vector<bool> survived){
    int numSurvived = 0;
    for(int i = 0; i < survived.size(); i++){
        if(survived.at(i) == 1) numSurvived++;
    }
    return numSurvived;
}

//Calculate Aprioris for Survived, 
vector<double> calculateApriori(TitanicDataframe* d) {
    vector<double> apriori;
    double survivedCount = 0;
    double size = d->survived.size();
    for(int i = 0; i < d->survived.size(); i++){
        if(d->survived.at(i) == 1) {
            survivedCount++;
        }
    }
    //Didnt survive then survived
    apriori.push_back((d->survived.size() - survivedCount) / d->survived.size());
    apriori.push_back(survivedCount / d->survived.size());
    printMatrix(apriori, 1,2,0);
    return apriori;
}


vector<double> calculatePClassLikelihood(vector<int> factor, vector<bool> sur, int factors, int numSurv){
    vector<double> likelihoodSums{0,0,0,0,0,0}; // (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
    vector<double> likelihoods;
    for(int i = 0; i < factor.size(); i++){
        if(factor.at(i) == 0 && sur.at(i) == 0) likelihoodSums.at(0)++;
        else if(factor.at(i) == 0 && sur.at(i) == 1) likelihoodSums.at(1)++;
        else if(factor.at(i) == 1 && sur.at(i) == 0) likelihoodSums.at(2)++;
        else if(factor.at(i) == 1 && sur.at(i) == 1) likelihoodSums.at(3)++;
        else if(factor.at(i) == 2 && sur.at(i) == 0) likelihoodSums.at(4)++;
        else if(factor.at(i) == 2 && sur.at(i) == 1) likelihoodSums.at(5)++;
    }
    for(int i = 0; i < 6; i++){
        likelihoods.push_back(likelihoodSums.at(i) / numSurv);
    }
    printMatrix(likelihoods, 2, 3, 0);
    return likelihoods;
}

vector<double> calculateSexLikelihood(vector<bool> factor, vector<bool> sur, int factors, int numSurv){
    vector<double> likelihoodSums{0,0,0,0};
    vector<double> likelihoods;
    for(int i = 0; i < factor.size(); i++){
        if(factor.at(i) == 0 && sur.at(i) == 0) likelihoodSums.at(0)++;
        else if(factor.at(i) == 0 && sur.at(i) == 1) likelihoodSums.at(1)++;
        else if(factor.at(i) == 1 && sur.at(i) == 0) likelihoodSums.at(2)++;
        else if(factor.at(i) == 1 && sur.at(i) == 1) likelihoodSums.at(3)++;
    }
    for(int i = 0; i < 4; i++){
        likelihoods.push_back(likelihoodSums.at(i) / numSurv);
    }
    printMatrix(likelihoods, 2,2, 0);
    return likelihoods;
}

//Calculate mean of ages where passengers survived and mean of ages where passengers didn't survive
vector<double> calculateMean(vector<int> age, vector<bool> survived, int countSurvived){
    vector<double> means;
    int survivedSum = 0, dnSurviveSum = 0;
    for(int i = 0; i < survived.size(); i++){
        if(survived.at(i) == 0) dnSurviveSum += age.at(i);
        else survivedSum += age.at(i);
    }
    means.push_back(dnSurviveSum / (survived.size() - countSurvived));
    means.push_back(survivedSum / countSurvived);

    return means;
}

vector<double> calculateVar(vector<double> means, vector<bool> survived, vector<int> age, int countSurvived){
    vector<double> var;
    int sum1 = 0, sum2 = 0;
    for(int i = 0; i < survived.size(); i++){
        if(survived.at(i) ==0) sum1 += pow((age.at(i) - means.at(0)),2);
        else sum2 += pow((age.at(i) - means.at(1)),2);
    }
    var.push_back(sum1 / (survived.size() - countSurvived));
    var.push_back(sum2 / (countSurvived));

    return var;
}

double calculateAgeProb(double v, double meanV, double varV){
    return (1 / sqrt(2 * 3.14 * varV) * exp(-pow((v-meanV), 2) / (2 * varV)));
}

//Num S is Likelihood survived for pclass, likelihood survived for sex, apriori for survived, likelihood for survived for age.
//Num P is same but for didn't survive
vector<double> calculateRawProbabilities(vector<double> sexLikelihood, vector<double> pclassLiklihood, vector<double> apriori, vector<int> age, vector<bool> sex, vector<int> pclass, vector<double> mean, vector<double> var, vector<bool> survived){
    vector<double> probabilities(2*survived.size());
    
    for(int i = 0; i < survived.size(); i++){
        double num_s = apriori.at(1) * pclassLiklihood.at(2*(pclass.at(i)-1)) * sexLikelihood.at(2*sex.at(i)) * calculateAgeProb(age.at(i), mean.at(1), mean.at(1));
        double num_p = apriori.at(0) * pclassLiklihood.at(pclass.at(i)) * sexLikelihood.at(sex.at(i)) * calculateAgeProb(age.at(i), mean.at(0), mean.at(0));
        double denom = pclassLiklihood.at(2*(pclass.at(i)-1)) * sexLikelihood.at(2*sex.at(i)) * calculateAgeProb(age.at(i), mean.at(1), mean.at(1)) * apriori.at(1) + apriori.at(0) * pclassLiklihood.at(pclass.at(i)) * sexLikelihood.at(sex.at(i)) * calculateAgeProb(age.at(i), mean.at(0), mean.at(0));
        probabilities.push_back(num_s / denom);
        probabilities.push_back(num_p / denom);
    }

    return probabilities;
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


int main(int argc, char** argv) {
    srand(1234);
    ifstream inFS; // Input File Stream
	string line;
	string pclass_in, survived_in, sex_in, age_in, unclassified_data_in;
    TitanicDataframe *train = new TitanicDataframe();
    TitanicDataframe *test = new TitanicDataframe();
    
	double covariance;
    int survivedCount;
    vector<double> mean;
    vector<double> var;
    vector<double> apriori;
    vector<double> sex_likelihood, pclass_likelihood;
    vector<double> probs;
	vector<int> pclass(MAX_LEN);
	vector<bool> survived(MAX_LEN);
    vector<bool> sex(MAX_LEN);
    vector<int> age(MAX_LEN);
    vector<int> unclassified_data(MAX_LEN);
    vector<bool> predictions;
	
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

    survivedCount = findNumberSurvived(train->survived);
    cout << "NUm Sur: " << survivedCount << endl;
    apriori = calculateApriori(train);
    pclass_likelihood = calculatePClassLikelihood(train->pclass, train->survived, 3, survivedCount);
    sex_likelihood = calculateSexLikelihood(train->sex, train->survived, 2, survivedCount);
    cout << "Means" << endl;
    mean = calculateMean(train->age, train->survived, survivedCount);
    cout << "Means" << endl;
    var = calculateVar(mean, train->survived, train->age, survivedCount);
    cout << "Means" << endl;
    probs = calculateRawProbabilities(sex_likelihood, pclass_likelihood, apriori, test->age, test->sex, test->pclass, mean, var, test->survived);
    cout << "Means" << endl;

    for(int i = 0; i < probs.size() / 2; i+=2){
        if(probs.at(i) > probs.at(i+1)) predictions.push_back(1);
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

	cout << "Closing file Titanic.csv" << endl;
	inFS.close(); // DOne with file so close iter_swap
}