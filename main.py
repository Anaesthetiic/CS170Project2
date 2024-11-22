import random
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
            print("You're screwed, the latest highest accuracy was {:.2f}".format(selected_node.get_highest_accuracy()))
            print("Best feature subset is {}, which has an accuracy of {:.2f}".format(selected_node.data, selected_node.get_highest_accuracy()))
            break
        else:
            selected_node = highestAccuracyPtr




def backward_elimination(n):
    selected_node = Node()
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
            print("You're screwed, the latest highest accuracy was {:.2f}".format(selected_node.get_highest_accuracy()))
            print("Best feature subset is {}, which has an accuracy of {:.2f}".format(selected_node.data, selected_node.get_highest_accuracy()))
            break
        else:
            selected_node = highestAccuracyPtr

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
    
