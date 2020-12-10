# Use regular expressions routines
import re
import random
import numpy as np
import math

from sklearn.model_selection import train_test_split

UPPER = 1
LOWER = -1

def sigmoid( x ):
    return 1 / ( 1 + math.exp( -x ) )

def sigmoid_derivative( x ):
    return x * ( 1 - x )

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

        self.learning_rate = 0.1

        # Weights for [ INPUT -> HIDDEN ]
        self.input_weights = np.array( np.random.uniform( LOWER, UPPER, size = (self.hidden_nodes, self.input_nodes) ) )
        # Weights for [ HIDDEN -> OUTPUT ]
        self.output_weights = np.array( np.random.uniform( LOWER, UPPER, size = (self.output_nodes, self.hidden_nodes) ) )
        print( self.output_weights )

        # Bias for [ INPUT -> HIDDEN ]
        self.hidden_bias = np.array( np.random.uniform( LOWER, UPPER, size = (self.hidden_nodes, 1) ) )
        # Bias for [ HIDDEN -> OUTPUT ]
        self.output_bias = np.array( np.random.uniform( LOWER, UPPER, size = (self.output_nodes, 1) ) )

    def feed_forward( self, x ):
        # Vectorized sigmoid function to take a numpy array as input and returns a numpy array as output
        sigmoid_function = np.vectorize( sigmoid )

        # Generate Hidden
        # -----------------------
        # weight (dot) inputs = hidden
        inputs = np.array( x )
        inputs = inputs.reshape( len( x ), 1 )

        hidden = self.input_weights.dot( inputs ) + 1
        # add bias  
        # hidden = hidden + 1
        # activation function
        hidden = sigmoid_function( hidden )

        # Generate Outputs
        # -----------------------
        # weight (dot) hidden = output
        output = self.output_weights.dot( hidden ) + 1
        # add bias
        # output = output 
        # activation function
        output = sigmoid_function( output )

        # Return List
        return ( output.ravel().tolist(), hidden.ravel().tolist() )

    def train( self, x, y ):
        inputs = np.array( x )
        inputs = inputs.reshape( len( x ), 1 )

        expected_outputs = [0] * ( self.output_nodes - 1 )
        expected_outputs.insert( y, 1 )

        # Convert lists to matrix
        expected_outputs = np.array( expected_outputs ).reshape( self.output_nodes, 1)

        guessed_output, hidden_outputs = self.feed_forward( x )

        guessed_output = np.array( guessed_output ).reshape( self.output_nodes, 1)      # sigmoid already applied
        hidden_outputs = np.array( hidden_outputs ).reshape( self.hidden_nodes, 1)      # sigmoid already applied

        # Calculate errors
        output_errors = expected_outputs - guessed_output
        hidden_errors = np.transpose( self.output_weights ).dot( output_errors )

        # # Calculate hidden->output deltas
        output_deltas = self.calc_deltas( hidden_outputs, guessed_output, output_errors )

        # # Maybe ???
        # np.add( self.output_weights, output_deltas )

        # # Calculate input->hidden deltas
        hidden_deltas = self.calc_deltas( inputs, hidden_outputs, hidden_errors )

        # Return List
        return ( output_deltas, hidden_deltas )

    def test( self, x ):
        pass

    def calc_deltas( self, x, y, z ):
        sigmoid_derivative_function = np.vectorize( sigmoid_derivative )
        # Calculate gradiant
        # gradiant = np.gradient( y )                   # Derivative function method
        gradiant = sigmoid_derivative_function( y )     # Derivative function method
        
        # # Hadamard Product = [ a * b ] = np.multiply(a, b) 
        gradiant = np.multiply( z, gradiant )                       # gradiant = [ output_error * gradiant ]
        gradiant = np.multiply( self.learning_rate, gradiant )      # gradiant = [ learning_rate * gradiant ]

        # Calculate hidden->output weight deltas
        # deltas = [ learning_rate * error * gradiant ] x transpose( input )
        return gradiant.dot( np.transpose( x ) )

    

##################################################
# Main
##################################################
def main():

    input_sets = read_file( 'training_vectors.txt' )

    training_set, testing_set = train_test_split( input_sets, test_size = 0.20, shuffle = True )
    
    n = NeuralNetwork( 10, 8, 12 )

    # n.train( [17, 13, 74, 63, 78, 12, 22, 82, 55, 15], 5 )

    for x in n.train( [17, 13, 74, 63, 78, 12, 22, 82, 55, 15], 5 ):
        print( x )

if __name__ == "__main__":
    main()