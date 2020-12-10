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
    def __init__( self, input_nodes, hidden_nodes, output_nodes ):
        self.input_nodes = input_nodes     # 10 Input Nodes [0, . . ., 96] + 1 Bias Input
        self.hidden_nodes = hidden_nodes   # 11 Hidden Nodes + 1 Bias Node
        self.output_nodes = output_nodes        # 8 Output Node [0, . . ., 7]

        self.learning_rate = 0.1

        # Weights for [ INPUT -> HIDDEN ]
        self.input_weights = np.array( np.random.uniform( LOWER, UPPER, size = (self.hidden_nodes, self.input_nodes + 1) ) )
        # Weights for [ HIDDEN -> OUTPUT ]
        self.output_weights = np.array( np.random.uniform( LOWER, UPPER, size = (self.output_nodes, self.hidden_nodes + 1) ) )

        # Bias for [ INPUT -> HIDDEN ]
        self.hidden_bias = 1
        # Bias for [ HIDDEN -> OUTPUT ]
        self.output_bias = 1

    def feed_forward( self, x ):
        # Vectorized sigmoid function to take a numpy array as input and returns a numpy array as output
        sigmoid_function = np.vectorize( sigmoid )

        inputs = np.array( x )
        inputs = inputs.reshape( len( x ), 1 )
        inputs = np.vstack([inputs, [[self.hidden_bias]]] )       # Add bias value
       
        # hidden = weight x inputs
        hidden = self.input_weights.dot( inputs )
    
        # activation function
        hidden = sigmoid_function( hidden )
        hidden = np.vstack([hidden,  [[self.output_bias]]] )       # Add bias value


        # # output = weight x hidden 
        outputs = self.output_weights.dot( hidden )
        # activation function
        outputs = sigmoid_function( outputs )

        # Return List
        return ( inputs, hidden, outputs )

    def add_bias( self, x ):
        inputs = x
        inputs.append( 1 )

        return np.array( x )

    def train( self, x, y ):

        sigmoid_derivative_function = np.vectorize( sigmoid_derivative )

        expected_outputs = [0] * ( self.output_nodes - 1 )
        expected_outputs.insert( y, 1 )

        # Convert lists to matrix
        expected_outputs = np.array( expected_outputs ).reshape( self.output_nodes, 1)

        inputs, hidden_outputs, guess_outputs = self.feed_forward( x )

        # Calculate errors
        output_errors = expected_outputs - guess_outputs
        hidden_errors = np.transpose( self.output_weights ).dot( output_errors )

        # Calculate output gradiant
        output_gradient = sigmoid_derivative_function( guess_outputs )
        output_gradient = np.multiply( output_errors, output_gradient )
        output_gradient = np.multiply( self.learning_rate, output_gradient )

        # Calculate hidden->output deltas
        output_deltas = output_gradient.dot( np.transpose( hidden_outputs ) )

        # TODO: Adjust hidden->output weights by deltas
        self.output_weights = np.add( self.output_weights, output_deltas )

        # TODO: Adjust bias by deltas( which is just the gradients )
        pass

        # Calculate hidden gradiant
        hidden_gradient = sigmoid_derivative_function( hidden_outputs )
        hidden_gradient = np.multiply( hidden_errors, hidden_gradient )
        hidden_gradient = np.multiply( self.learning_rate, hidden_gradient )

        # Calculate hidden->output deltas
        hidden_deltas = hidden_gradient.dot( np.transpose( inputs ) )

        # Adjust hidden->output weights by deltas
        self.input_weights = np.add( self.input_weights, hidden_deltas )

        # TODO: Adjust bias by deltas( which is just the gradients )
        pass

        # Return List
        return ( output_deltas, hidden_deltas )

    def test( self, x ):
        pass


##################################################
# Main
##################################################
def main():
    input_sets = read_file( 'training_vectors.txt' )

    # training_set, testing_set = train_test_split( input_sets, test_size = 0.20, shuffle = True )
    
    n = NeuralNetwork( 10, 11, 8 )
    
    for x in n.train( [17, 13, 74, 63, 78, 12, 22, 82, 55, 15], 5):
        print( x )

if __name__ == "__main__":
    main()