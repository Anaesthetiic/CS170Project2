import random

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
    highestAccuracyPtr = None
    currNode = None
    firstIt = True
    while firstIt or (currNode.get_highest_accuracy() > highestAccuracyPtr.get_highest_accuracy()):
        firstIt = False
        node = Node()
        for i in range(n):      # for n features
            node.add_data(evalFunc())   # feature i (indexed) has a eval of ___


def backward_elimination(n):
    pass

def main():
    print("Welcome to afranco/tcast054's Feature Selection Algorithm.")
    total_features = input("Please enter total number of features: ")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    # print("3. Bertie’s Special Algorithm.")
    choice = input("Option: ")
    if choice == '1':
        print("Beginning search.\n")
        forward_selection(total_features)
    elif choice == '2':
        backward_elimination(total_features)
    # elif choice == '3':
    #     berties_special_algorithm(total_features)
    else:
        print("Invalid option. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
    
# EVAL FUNCTION STUB, RETURNS RANDOM %VAL
def evalFunc(upperbound=100):
    # return random.randint(0, upperbound)
    return random.uniform(0, upperbound)