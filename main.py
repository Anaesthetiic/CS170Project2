import random
import pandas as pd                 # pip install pandas
from typing import List

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

def forward_selection(n):
    # selected_features = []
    selected_node = Node()
    print("Using no features and “random” evaluation, I get an accuracy of {:.1f}%\n".format(evalFunc()))
    print("Beginning search.\n")
    while (len(selected_node.data) < int(n)):
        highestAccuracyPtr = Node()
        for i in range(1, int(n)+1):
            if i not in selected_node.data:
                currNode = Node()
                currNode.set_highest_accuracy(evalFunc())
                currNode.data = selected_node.data + [i]
                print("Using feature(s) {} accuracy is {:.1f}".format(currNode.data, currNode.get_highest_accuracy()))
                if (currNode.get_highest_accuracy() > highestAccuracyPtr.get_highest_accuracy()):
                        highestAccuracyPtr = currNode
        print("\nFeature set {} was best, accuracy is {:.1f}\n".format(highestAccuracyPtr.data, highestAccuracyPtr.get_highest_accuracy()))
        if (highestAccuracyPtr.get_highest_accuracy() <= selected_node.get_highest_accuracy()):
            print("(Warning, Accuracy has decreased!)")
            print("Best feature subset is {}, which has an accuracy of {:.2f}".format(selected_node.data, selected_node.get_highest_accuracy()))
            break
        else:
            selected_node = highestAccuracyPtr




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




class Validator:
    
    # Implemented using leave-one-out validation method
    # Input: feature is list of strings corresponding to name of column for feature subset , classifier object, dataset (DataFrame or df)
    # Output: Float representing accuracy [0..1]
    # ex: NN(["Feature 1", "Feature 2", "Feature 5"], classifier, df)
    @staticmethod
    def NN(feature: List[str], classifier, dataset: pd.DataFrame):
        num_instances = dataset.size()      # num instances
        correct_count = 0                   # tracks accuracy
        # repeat reserving single instance for all instances 
        for testInstance in range(num_instances):
            # reserve testInstance as test data, use other instances as training data
            for instance in range(num_instances):
                if(instance == testInstance): pass    # pass if instance we want to reserve
                else:
                    # classifier.train() - train NN 
                    pass
            # training complete at this point
            
            # test NN output, compare to known answer at dataset.iloc[testInstance]["Classifier"]
            # classifier.test()
            output = 0  # 1 or 2
            if(output == dataset.iloc[testInstance]["Classifier"]): # NN output is correct
                correct_count += 1
            else: # NN output is incorrect
                pass
            
            accuracy = correct_count / num_instances
            return accuracy
    


def main():
    print("1. Part 1: Forward Selection/Backward Elimination")
    print("2. Part 2: Nearest Neighbor")
    print("3. Part 3")
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
            df = pd.read_csv(filename, sep='\s+', engine="python", names=["Classifier", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7", "Feature 8", "Feature 9", "Feature 10"])    # separated by whitespace, expect 10 features
        elif choice == '2':
            filename = "large-test-dataset.txt"
            df = pd.read_csv(filename, sep='\s+', engine="python", names=["Classifier", "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7", "Feature 8", "Feature 9", "Feature 10", "Feature 11", "Feature 12", "Feature 13", "Feature 14", "Feature 15", "Feature 16", "Feature 17", "Feature 18", "Feature 19", "Feature 20", "Feature 21", "Feature 22", "Feature 23", "Feature 24", "Feature 25", "Feature 26", "Feature 27", "Feature 28", "Feature 29", "Feature 30", "Feature 31", "Feature 32", "Feature 33", "Feature 34", "Feature 35", "Feature 36", "Feature 37", "Feature 38", "Feature 39", "Feature 40"])  
            # separated by whitespace, expect 40 features
        else:
            print("Invalid option. Please select 1 or 2")
        
        # print(df)
        normalized_df=(df-df.min())/(df.max()-df.min())     # Min-max Normalization https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
        print(normalized_df)
    
        #print(normalized_df.iloc[1]["Feature 2"]) Example of how to reference index and feature
        # print(normalized_df.iloc["Feature 1"])
        
        if choice == '1':
            print("Feature {3, 5, 7}, accuracy should be about 0.89")
            # print("Accuracy = " + Validator.NN(["Feature 3", "Feature 5", "Feature 7"], classifier, df))
        elif choice == '2':
            print("Feature {1, 15, 27} accuracy should be about 0.949")
            # print("Accuracy = " + Validator.NN(["Feature 1", "Feature 15", "Feature 27"], classifier, df))
        
    elif choice == '3':
        pass
    else:
        print("Invalid option. Please select 1, 2, or 3.")
    

if __name__ == "__main__":
    main()

