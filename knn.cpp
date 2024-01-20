#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

struct Person {
    string gender;
    int age;
    int salary;
    int purchaseIphone;
};

class KNNClassifier {
    private:
        vector<Person> trainingData;
        int k;
        double euclideanDistance(const Person& p1, const Person& p2);
    public:
        KNNClassifier(int k): k(k) {};

        void trainTestSplit(const vector<Person>& X, const vector<int>& y,
                            double test_size, 
                            vector<Person>& X_train, vector<int>& y_train, 
                            vector<Person>& X_test, vector<int>& y_test);
        int predict(const vector<Person>& dataset, const Person& newPerson);
        vector<Person> readCSV(const string& filename);
        int weightedKNN(const vector<Person>& dataset, const Person& newPerson);
        vector<vector<int>> confusionMatrix(vector<int>& y_test, vector<int>& y_pred);
        double accuracyScore(vector<int>& y_test, vector<int>& y_pred);
        double precisionScore(int truePositive, int falsePositive);
        double recallScore(int truePositive, int falseNegative);
        double F1Score(double precision, double recall);
        void setK(int newK);

};

double KNNClassifier::euclideanDistance(const Person& p1, const Person& p2) {
    // Euclidean distance calculation with gender included
    int genderFactor = (p1.gender == p2.gender) ? 0 : 1;
    return sqrt(pow(p1.age - p2.age, 2) + pow(p1.salary - p2.salary, 2) + pow(genderFactor, 2));
}

void KNNClassifier::trainTestSplit(const vector<Person>& X, const vector<int>& y,
                                   double test_size, 
                                   vector<Person>& X_train, vector<int>& y_train, 
                                   vector<Person>& X_test, vector<int>& y_test)
{
    // Combine X and y into a single dataset
    vector<pair<Person, int>> dataset;
    for (size_t i = 0; i < X.size(); i++) {
        dataset.push_back({X[i], y[i]});
    }

    // Shuffle the dataset randomly
    random_device rd;
    default_random_engine rng(rd());
    shuffle(dataset.begin(), dataset.end(), rng);

    // Calculate the split index based on the test_size ratio
    size_t split_index = static_cast<size_t>(test_size * dataset.size());

    // Clear the training and testing set variables
    X_train.clear();
    y_train.clear();
    X_test.clear();
    y_test.clear();

    // Split the dataset into training and testing sets
    for (size_t i = 0; i < dataset.size(); i++) {
        if (i < split_index) {
            X_train.push_back(dataset[i].first);
            y_train.push_back(dataset[i].second);
        } else {
            X_test.push_back(dataset[i].first);
            y_test.push_back(dataset[i].second);
        }
    }
}

int KNNClassifier::predict(const vector<Person>& dataset, const Person& newPerson) {
    // Calculate distances between the new person and all existing persons in the dataset
    vector<pair<double, int>> distances;
    for (size_t i = 0; i < dataset.size(); ++i) {
        double distance = euclideanDistance(newPerson, dataset[i]);
        distances.emplace_back(distance, i);
    }

    // Sort distances in ascending order
    sort(distances.begin(), distances.end());

    // Count the number of purchases among the k nearest neighbors
    int purchaseCount = 0;
    for (int i = 0; i < k; ++i) {
        int neighborIndex = distances[i].second;
        purchaseCount += dataset[neighborIndex].purchaseIphone;
    }

    // Predict purchase based on majority vote
    return (purchaseCount > k / 2) ? 1 : 0;
}

vector<Person> KNNClassifier::readCSV(const string& filename) {
    // vector<Person> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return trainingData;  // Return an empty vector in case of an error
    }

    // Read the header line to skip it
    string line;
    getline(file, line);

    // Read the data
    while (getline(file, line)) {
        istringstream iss(line);
        string gender;
        string ageStr, salaryStr, purchaseStr;

        // Assuming the CSV structure is "Gender,Age,Salary,Purchase Iphone"
        getline(iss, gender, ',');
        getline(iss, ageStr, ',');
        getline(iss, salaryStr, ',');
        getline(iss, purchaseStr, ',');

        // Convert strings to the appropriate types
        Person person;
        person.gender = gender;
        person.age = stoi(ageStr);
        person.salary = stoi(salaryStr);
        person.purchaseIphone = stoi(purchaseStr);

        // Add the person to the vector
        trainingData.push_back(person);
    }

    file.close();
    return trainingData;
}

int KNNClassifier::weightedKNN(const vector<Person>& dataset, const Person& newPerson) {
    // Calculate distances between the new person and all existing persons in the dataset
    vector<pair<double, int>> distances;
    for (size_t i = 0; i < dataset.size(); ++i) {
        double distance = euclideanDistance(newPerson, dataset[i]);
        distances.emplace_back(distance, i);
    }

    // Sort distances in ascending order
    sort(distances.begin(), distances.end());

    // Consider the first k elements and two groups
    double freq1 = 0;
    double freq2 = 0;
    // Compute weighing function and increment for each frequency
    for (int i = 0; i < k; i++) {
        int neighborIndex = distances[i].second;
        if (dataset[neighborIndex].purchaseIphone == 0) {
            freq1 += double(1 / distances[i].first);
        }
        else if (dataset[neighborIndex].purchaseIphone == 1) {
            freq2 += double(1 / distances[i].first);
        }
    }
    return (freq1 > freq2 ? 0 : 1);
}

double KNNClassifier::accuracyScore(vector<int>& y_test, vector<int>& y_pred) {
    int correctPredictions = 0;
    size_t numPredictions = y_pred.size();
    for (int i = 0; i < y_pred.size(); i++) {
        if (y_pred[i] == y_test[i]) {
            correctPredictions++;
        }
    }
    double accuracy = double(correctPredictions) / double(numPredictions);
    return accuracy;
}

vector<vector<int>> KNNClassifier::confusionMatrix(vector<int>& y_test, vector<int>& y_pred) {
    int tp, tn, fp, fn = 0;

    for (int i = 0; i < y_pred.size(); i++) {
        // True Positive (TP): when both the actual and predicted values are 1.
        if (y_test[i] == 1 && y_pred[i] == 1) {
            tp++;
        }
        // True Negative (TN): when both the actual and predicted values are 0.
        else if (y_test[i] == 0 && y_pred[i] == 0) {
            tn++;
        }
        // False Positive (FP): when the actual value is 0 but the predicted value is 1.
        else if (y_test[i] == 0 && y_pred[i] == 1) {
            fp++;
        }    
        // False Negative (FN): when the actual value is 1 but the predicted value is 0.
        else if (y_test[i] == 1 && y_pred[i] == 0) {
            fn++;
        }
    }

    // Populate the confusion matrix
    vector<vector<int>> confusionMatrix = {
        {tn, fp},
        {fn, tp}
    };

    return confusionMatrix;
}

double KNNClassifier::precisionScore(int truePositive, int falsePositive) {
    double precision = double(truePositive) / double(truePositive + falsePositive);
    return precision;
}

double KNNClassifier::recallScore(int truePositive, int falseNegative) {
    double recall =  double(truePositive) / double(truePositive + falseNegative);
    return recall;
}

double KNNClassifier::F1Score(double precision, double recall) {
    double F1 = double(2 * precision * recall) / double(precision + recall);
    return F1;
}

void KNNClassifier::setK(int newK) {
    if (newK > 0) {
        k = newK;
    }
    else {
        cout << "Error: k must be a positive integer." << endl;
    }
}

int main() {
    // Instantiate KNN model with k = 5
    KNNClassifier knn(5);

    // Define the dataset
    vector<Person> X = knn.readCSV("iphone_purchase_records.csv");
    vector<int> y;
    for (int i = 0; i < X.size(); i++) {
        y.push_back(X[i].purchaseIphone);
    }

    double test_size = 0.8; // 80% training, 20% testing
    
    vector<Person> X_train;
    vector<int> y_train;
    vector<Person> X_test;
    vector<int> y_test;

    knn.trainTestSplit(X, y, test_size, X_train, y_train, X_test, y_test);

    // Make predictions on the test dataset
    vector<int> y_pred;
    for (int i = 0; i < X_test.size(); i++) {
        int prediction = knn.predict(X_train, X_test[i]);
        y_pred.push_back(prediction);
    }

    // Evaluate Prediction Accuracy of Standard KNN
    double accuracy = knn.accuracyScore(y_test, y_pred);
    vector<vector<int>> confusionMatrix = knn.confusionMatrix(y_test, y_pred);
    double precision = knn.precisionScore(confusionMatrix[1][1], confusionMatrix[0][1]);
    double recall = knn.recallScore(confusionMatrix[1][1], confusionMatrix[1][0]);
    double F1 = knn.F1Score(precision, recall);

    cout << "Regular KNN Results: " << endl << endl;
    cout << "Confusion Matrix: " << endl;
    cout << confusionMatrix[0][0] << " " << confusionMatrix[0][1] << endl;
    cout << confusionMatrix[1][0] << "  " << confusionMatrix[1][1] << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1 Score: " << F1 << endl;
    cout << endl;

    // Use the Training and Testing sets to perform Weighted KNN prediction
    vector<int> y_weighted_pred;
    for (int i = 0; i < X_test.size(); i++) {
        int weightedPredicition = knn.weightedKNN(X_train, X_test[i]);
        y_weighted_pred.push_back(weightedPredicition);
    }

    // Evaluate Prediction Accuracy of Weighted KNN
    double w_accuracy = knn.accuracyScore(y_test, y_weighted_pred);
    vector<vector<int>> w_confusionMatrix = knn.confusionMatrix(y_test, y_weighted_pred);
    double w_precision = knn.precisionScore(w_confusionMatrix[1][1], w_confusionMatrix[0][1]);
    double w_recall = knn.recallScore(w_confusionMatrix[1][1], w_confusionMatrix[1][0]);
    double w_F1 = knn.F1Score(w_precision, w_recall);


    cout << "Weighted KNN Results: " << endl << endl;
    cout << "Confusion Matrix: " << endl;
    cout << w_confusionMatrix[0][0] << " " << w_confusionMatrix[0][1] << endl;
    cout << w_confusionMatrix[1][0] << "  " << w_confusionMatrix[1][1] << endl;
    cout << "Accuracy: " << w_accuracy << endl;
    cout << "Precision: " << w_precision << endl;
    cout << "Recall: " << w_recall << endl;
    cout << "F1 Score: " << w_F1 << endl;

    return 0;
}