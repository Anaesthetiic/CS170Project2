import random
import pandas as pd                 # pip install pandas
from typing import List
import time
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# EVAL FUNCTION STUB, RETURNS RANDOM %VAL
def evalFunc(upperbound=100):
    # return random.randint(0, upperbound)
    return random.uniform(0, upperbound)
# Not sure if we'll need this
class Node:
    def __init__(self):
        self.data = []
        self.parent = None
        self.children = []
        self.indexBestFeature = -1
        self.highestAccuracy = -1  # float [0..100] representing percentage

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
        
    def add_data(self, evalFunc):
        self.data.append(evalFunc)
        
    def set_index_best_feature(self, index):
        self.indexBestFeature = index

    def get_index_best_feature(self):
        return self.indexBestFeature
    
    def set_highest_accuracy(self, accuracy):
        self.highestAccuracy = accuracy
        
    def get_highest_accuracy(self):
        return self.highestAccuracy


class Classifier:
    def __init__(self):
        self.data = []
        self.data = []

    def train(self, training_instances):
        # print("\nhello world")
        # print(training_instances)
        self.data.append(training_instances)
        # for i in range(training_instances.shape[0]):
        #     self.labels.append(float(training_instances.iloc[i,0]))
        #     self.features.append(training_instances.iloc[i, 1:].values.tolist())
        
    def test(self, test_instance):
        distances = []
        test_features = test_instance[1:].values
        # print(test_features)
        # print(self.data[0].shape[0])
        for train_instance in range(self.data[0].shape[0]):
            train_features = self.data[0].iloc[train_instance][1:].values
            # print(train_features)
            distance = euclidean_distance(test_features, train_features)
            distances.append(distance)
            # print(f"Distance between (i = {train_instance}) {test_features} and {train_features} is {distance}")
        # print(distances)
        nearest = distances.index(min(distances))
        # print(f"The nearest point was at {nearest}, with distance of {min(distances)}")
        # print(self.data[0].iloc[nearest,0])
        return self.data[0].iloc[nearest,0]
    

class Validator:
    
    # Implemented using leave-one-out validation method
    # Input: feature is list of strings corresponding to name of column for feature subset , classifier object, dataset (DataFrame or df) of only feature subset
    # Output: Float representing accuracy [0..1]
    # ex: NN(["Feature 1", "Feature 2", "Feature 5"], classifier, df)
    @staticmethod
    def NN(feature: List[str], classifier, dataset: pd.DataFrame):
        num_instances = dataset.shape[0]    # num instances
        correct_count = 0                   # tracks accuracy
        # print(target_feature)
        # repeat reserving single instance for all instances 
        for testInstance in tqdm(range(num_instances), desc="Processing instances for feature \"{}\"".format(feature), unit="instance"): # https://www.geeksforgeeks.org/progress-bars-in-python/
            # reserve testInstance as test data, use other instances as training data
            # print(f"Reserving instance {testInstance} as test data. Using other instances as training data.")
            classifier.data = []
            training_data = dataset.drop(testInstance)
            classifier.train(training_data)
            test_row = dataset.iloc[testInstance]
            prediction = classifier.test(test_row)
            output_bool = (prediction == dataset.iloc[testInstance][feature])
            if(output_bool): # NN output is correct
                correct_count += 1
            else: # NN output is incorrect
                pass
            # print(f"\tCheck if Classifier Test outputs correct classifier. {prediction} == {dataset.iloc[testInstance][feature]} is {output_bool}")

        accuracy = correct_count / num_instances
        return accuracy



def forward_selection(n, feature_names, classifier, dataset):
    # selected_features = []
    selected_node = Node()
    target_feature = feature_names[0]
    print("Using no features and “random” evaluation, I get an accuracy of {:.1f}%\n".format(evalFunc()))
    print("Beginning search.\n")

    feature_subset_cache = {}
    
    while (len(selected_node.data) < int(n)):
        highestAccuracyPtr = Node()
        for i in range(1, int(n)+1):
            if i not in selected_node.data:
                currNode = Node()
                # print(feature_names[i])
                selected_features = [target_feature] + [feature_names[j] for j in selected_node.data] + [feature_names[i]]
                feature_subset_key = tuple(sorted(selected_features))
                if feature_subset_key not in feature_subset_cache:
                    datasubset = dataset[selected_features].copy()
                    feature_subset_cache[feature_subset_key] = datasubset
                else:
                    datasubset = feature_subset_cache[feature_subset_key]
                    
                # print(datasubset)
                currNode.set_highest_accuracy(Validator.NN(target_feature, classifier, datasubset))
                currNode.data = selected_node.data + [i]
                print("\nUsing feature(s) {} accuracy is {:.2f}\n".format([feature_names[j] for j in currNode.data], currNode.get_highest_accuracy()))
                if (currNode.get_highest_accuracy() > highestAccuracyPtr.get_highest_accuracy()):
                        highestAccuracyPtr = currNode
        print("\nFeature set {} was best, accuracy is {:.2f}\n".format([feature_names[j] for j in highestAccuracyPtr.data], highestAccuracyPtr.get_highest_accuracy()))
        if (highestAccuracyPtr.get_highest_accuracy() <= selected_node.get_highest_accuracy()):
            print("(Warning, Accuracy has decreased!)")
            print("Best feature subset is {}, which has an accuracy of {:.2f}".format([feature_names[j] for j in selected_node.data], selected_node.get_highest_accuracy()))
            break
        else:
            selected_node = highestAccuracyPtr

def euclidean_distance(p1, p2):

    return sum((a-b) ** 2 for a,b in zip(p1,p2)) ** 0.5


def backward_elimination(n):
    selected_node = Node()
    print("Using no features and “random” evaluation, I get an accuracy of {:.1f}%\n".format(evalFunc()))
    for i in range(1, int(n)+1):
        selected_node.add_data(i)
    print(selected_node.data)

    while (len(selected_node.data) > 1):
        highestAccuracyPtr = Node()
        for i in reversed(range(1, int(n)+1)):
            if i in selected_node.data:
                currNode = Node()
                currNode.set_highest_accuracy(evalFunc())
                currNode.data = selected_node.data.copy()
                currNode.data.remove(i)
                print("Using feature(s) {} accuracy is {:.1f}".format(currNode.data, currNode.get_highest_accuracy()))
                if (currNode.get_highest_accuracy() >= highestAccuracyPtr.get_highest_accuracy()):
                        highestAccuracyPtr = currNode
        print("\nFeature set {} was best, accuracy is {:.1f}\n".format(highestAccuracyPtr.data, highestAccuracyPtr.get_highest_accuracy()))
        if (highestAccuracyPtr.get_highest_accuracy() <= selected_node.get_highest_accuracy()):
            print("(Warning, Accuracy has decreased!)")
            print("Best feature subset is {}, which has an accuracy of {:.2f}".format(selected_node.data, selected_node.get_highest_accuracy()))
            break
        else:
            selected_node = highestAccuracyPtr




    
def main():
    print("1. Part 1: Forward Selection/Backward Elimination")
    print("2. Part 2: Nearest Neighbor")
    print("3. Part 3: Titatnic Dataset")
    choice = input("Option: ")
    if choice == '1':
        print("Welcome to afranco/tcast054's Feature Selection Algorithm.")
        total_features = input("Please enter total number of features: ")
        print("Type the number of the algorithm you want to run.")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        choice = input("Option: ")
        if choice == '1':
            forward_selection(total_features)
        elif choice == '2':
            backward_elimination(total_features)    
        else:
            print("Invalid option. Please select 1, 2, or 3.")

    elif choice == '2':
        print("1. Small Test Dataset")
        print("2. Large Test Dataset")
        print("Select a Dataset: ")
        choice = input("Option: ")

        df = pd.DataFrame

        if choice == '1':
            filename = "small-test-dataset.txt"
            df = pd.read_csv(filename, sep='\s+', engine="python", names=["Classifier", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7", "Feature 8", "Feature 9", "Feature 10"])    
            # separated by whitespace, expect 10 features
        elif choice == '2':
            filename = "large-test-dataset.txt"
            df = pd.read_csv(filename, sep='\s+', engine="python", names=["Classifier", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7", "Feature 8", "Feature 9", "Feature 10", "Feature 11", "Feature 12", "Feature 13", "Feature 14", "Feature 15", "Feature 16", "Feature 17", "Feature 18", "Feature 19", "Feature 20", "Feature 21", "Feature 22", "Feature 23", "Feature 24", "Feature 25", "Feature 26", "Feature 27", "Feature 28", "Feature 29", "Feature 30", "Feature 31", "Feature 32", "Feature 33", "Feature 34", "Feature 35", "Feature 36", "Feature 37", "Feature 38", "Feature 39", "Feature 40"])  
            # separated by whitespace, expect 40 features
        else:
            print("Invalid option. Please select 1 or 2")

        # print(df)
        scaler = MinMaxScaler()
        cols_to_normalize = df.columns[1:]
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        normalized_df = df
        print(normalized_df)
        # normalized_df=(df-df.min())/(df.max()-df.min())     # Min-max Normalization https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
        # print(normalized_df)
        
        classifier = Classifier()
        # for i in range(len(normalized_df)):
        #     classifier.train(normalized_df.iloc[i])
        # prediction = classifier.test(normalized_df.iloc[0])
        # print(prediction)
        # print(normalized_df.iloc[0])
    
        #print(normalized_df.iloc[1]["Feature 2"]) Example of how to reference index and feature
    
        if choice == '1':
            start_time = time.time()
            test_df = normalized_df.loc[:, ["Classifier", "Feature 3", "Feature 5", "Feature 7"]]
            # print(test_df)
    
            accuracy = Validator.NN(["Feature 3", "Feature 5", "Feature 7"], classifier, test_df)
            end_time = time.time()

            print(f"Accuracy = {accuracy}")
            print(f"Time taken = {end_time - start_time} seconds")
            print("Feature {3, 5, 7}, accuracy should be about 0.89")
        elif choice == '2':
            start_time = time.time()
            test_df = normalized_df.loc[:, ["Classifier", "Feature 1", "Feature 15", "Feature 27"]]
            accuracy = Validator.NN(["Classifier", "Feature 1", "Feature 15", "Feature 27"], classifier, test_df)
            end_time = time.time()

            print(f"Accuracy = {accuracy}")
            print(f"Time taken = {end_time - start_time} seconds")
            print("Feature {1, 15, 27} accuracy should be about 0.949")
        
    elif choice == '3':
        print("Welcome to afranco/tcast054's Feature Selection Algorithm for Titanic dataset.")
        total_features = input("Please enter total number of features: ")
        print("Type the number of the algorithm you want to run.")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        choice = input("Option: ")

        classifier = Classifier()
        df = pd.DataFrame
        df = pd.read_csv('titanic-clean.txt', sep='\s+', engine="python", names=["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
        # print(df)

        feature_names = list(df.columns)
        # print(feature_names)

        if choice == '1':
            forward_selection(total_features, feature_names, classifier, df)
        elif choice == '2':
            pass   
        else:
            print("Invalid option. Please select 1, 2, or 3.")
        pass
    else:
        print("Invalid option. Please select 1, 2, or 3.")
    

if __name__ == "__main__":
    main()

