# Use regular expressions routines
import re
import random
import numpy as np
import math

from sklearn.model_selection import train_test_split

HIGH_RANGE = 1
LOW_RANGE = -1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

##################################################
# Training and Testing Sets
##################################################
def read_file( file_name ):
    sets = []
    with open( file_name ) as file:
        line = file.readline()
        while line:
            split_line = [val.strip(' ') for val in re.split(r'[()]', line.strip('(' + ')' + '\n'))]

            feature_vector = list( map( int, split_line[1].split(' ') ) )     # Convert string values to integer values
            label_val = int( split_line[2] )                                  # Save Class Value

            # Append new Node object w/ label value, and the feature vector values
            sets.append( (label_val, feature_vector) )

            line = file.readline()
    
    return sets

##################################################
# NeuralNetwork
##################################################
class NeuralNetwork:
    def __init__( self, input_nodes, output_nodes, hidden_nodes ):
        self.input_nodes = input_nodes       # 10 Input Nodes [0, . . ., 96]
        self.hidden_nodes = hidden_nodes     # 12 Hidden Nodes
        self.output_nodes = output_nodes     # 8 Output Node [0, . . ., 7]

        # Weights for [ INPUT -> HIDDEN ]
        self.hidden_weights = np.array( np.random.uniform( LOW_RANGE, HIGH_RANGE, size = (self.hidden_nodes, self.input_nodes) ) )
        # Weights for [ HIDDEN -> OUTPUT ]
        self.output_weights = np.array( np.random.uniform( LOW_RANGE, HIGH_RANGE, size = (self.output_nodes, self.hidden_nodes) ) )

        # Bias for [ INPUT -> HIDDEN ]
        self.hidden_bias = np.array( np.random.uniform( LOW_RANGE, HIGH_RANGE, size = (self.hidden_nodes, 1) ) )
        # Bias for [ HIDDEN -> OUTPUT ]
        self.output_bias = np.array( np.random.uniform( LOW_RANGE, HIGH_RANGE, size = (self.output_nodes, 1) ) )

    def feed_forward( self, x ):
        # Vectorized sigmoid function to take numpy arrays as inputs and returns a numpy array 
        sigmoid_function = np.vectorize( sigmoid )

        # Generate Hidden
        # -----------------------
        # weight (dot) inputs = hidden
        inputs = np.array( x )
        inputs = inputs.reshape( len( x ), 1 )

        hidden = self.hidden_weights.dot( inputs )
        # add bias  
        hidden = hidden + self.hidden_bias
        # activation function
        hidden = sigmoid_function( hidden )

        # Generate Outputs
        # -----------------------
        # weight (dot) hidden = output
        output = self.output_weights.dot( hidden )
        # add bias
        output = output + self.output_bias
        # activation function
        output = sigmoid_function( output )

        # Return List
        return output.ravel().tolist()

    def train( self, x, y ):
        expected_output = [0] * self.output_nodes
        expected_output.insert( y, 1 )

        guessed_output = self.feed_forward( x )


        output_error = []
        for i, j in zip( expected_output, guessed_output ):
            output_error.append( i - j )
        
        # hidden_error = []
        
        return output_error

    def test( self, x ):
        pass

##################################################
# Main
##################################################
def main():

    input_sets = read_file( 'training_vectors.txt' )

    training_set, testing_set = train_test_split( input_sets, test_size = 0.20, shuffle = True )
    
    n = NeuralNetwork( 10, 8, 12 )

    print( n.train( [17, 13, 74, 63, 78, 12, 22, 82, 55, 15], 5 ) )

if __name__ == "__main__":
    main()