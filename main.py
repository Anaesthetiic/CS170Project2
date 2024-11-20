import random

# Not sure if we'll need this
class Node:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.children = []

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

def forward_selection(n):
    pass


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
        forward_selection(total_features)
    elif choice == '2':
        backward_elimination(total_features)
    # elif choice == '3':
    #     berties_special_algorithm(total_features)
    else:
        print("Invalid option. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
    
# EVAL FUNCTION STUB, RETURNS RANDOM VAL
def evalFunc(upperbound=255):
    return random.randint(0, upperbound)