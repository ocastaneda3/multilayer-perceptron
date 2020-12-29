import re
import random
import numpy as np
import math

from sklearn.model_selection import train_test_split

# Limits for randomized weight generated
UPPER = 1
LOWER = -1

# Quick row/col identifiers for easier reading of code
ROWS = 0
COLS = 1

# Identifier for prining feed-forward output variable specifically
OUTPUT = 2

##################################################
# sigmoid( x )
##################################################
def sigmoid( x ):
    return 1.0 / ( 1.0 + np.exp( -x ) )

##################################################
# sigmoid'( x )
##################################################
def sigmoid_derivative( x ):
    return x * ( 1 - x )

##################################################
# Read training/testing sets
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
# Convert numberic value into coresponding 
# vector to be using in neural network
##################################################
def convert_output( old_data ):
    upper_lim = max( old_data )[0]

    new_data = []
    for x in old_data:
        new_label = [0] * ( upper_lim + 1 )
        new_label[x[0]] = 1
        new_vector = x[1]

        new_data.append( (new_label, new_vector) )

    return new_data

##################################################
# NeuralNetwork
##################################################
class NeuralNetwork:
    def __init__( self, input_nodes, hidden_nodes, output_nodes, learning_rate ):
        self.input_nodes = input_nodes          # 10 Input Nodes [0, . . ., 96] + 1 Bias Input
        self.hidden_nodes = hidden_nodes        # 11 Hidden Nodes + 1 Bias Node
        self.output_nodes = output_nodes        # 8 Output Node [0, . . ., 7]

        # Weights for [ INPUT -> HIDDEN ]
        self.input_weights = np.random.uniform( LOWER, UPPER, size = (self.hidden_nodes, self.input_nodes) )
        # print(self.input_weights, '\n')

        # Weights for [ HIDDEN -> OUTPUT ]
        self.output_weights = np.random.uniform( LOWER, UPPER, size = (self.output_nodes, self.hidden_nodes) )
        # print(self.output_weights, '\n')

        # Bias for [ INPUT -> HIDDEN ]
        self.hidden_bias = np.random.uniform( LOWER, UPPER, size = (self.hidden_nodes, 1) )

        # Bias for [ HIDDEN -> OUTPUT ]
        self.output_bias = np.random.uniform( LOWER, UPPER, size = (self.output_nodes, 1) )

        self.learning_rate = learning_rate

        self.epoch = 0

    def trained_output( self, x ):
        return list( self.feed_forward( x )[OUTPUT].flatten() )

    def feed_forward( self, x ):
        inputs = np.array( x ).reshape( len( x ), 1 )

        hidden = np.dot( self.input_weights, inputs )
        hidden = np.add( hidden, self.hidden_bias )

        # Activation Function
        hidden = sigmoid( hidden )

        outputs = np.dot( self.output_weights, hidden )
        outputs = np.add( outputs, self.output_bias )

        # Activation Function
        outputs = sigmoid( outputs )

        return ( inputs, hidden, outputs )

    def train( self, x, y ):
        inputs, hidden, outputs = self.feed_forward( x )

        target = np.array( y ).reshape( len( y ), 1 )

        # Calculate output error
        output_error = np.subtract( target, outputs )

        # Calculate hidden error
        hidden_error = np.dot( self.output_weights.T, output_error )

        # Calculate output gradient descent
        output_gradient = self.learning_rate * output_error * sigmoid_derivative( outputs )

        # Calculate output delta values
        output_deltas = np.dot( output_gradient, hidden.T )

        # Adjust output weight by calculated output deltas
        self.output_weights = np.add( self.output_weights, output_deltas )

        # Adjust output bias by calculated output gradient calculated
        self.output_bias = np.add( self.output_bias, output_gradient )
        
        # Calculate hidden gradient descent
        hidden_gradient = self.learning_rate * hidden_error * sigmoid_derivative( hidden )

        # Calculate hidden delta values
        hidden_deltas = np.dot( hidden_gradient, inputs.T )

        # Adjust input weights by the calculated hidden deltas
        self.input_weights = np.add( self.input_weights, hidden_deltas )

        # Adjust hidden bias by calculated hidden gradient calculated
        self.hidden_bias = np.add( self.hidden_bias, hidden_gradient )

        # Track number of epochs during training phase
        self.epoch += 1

##################################################
# Main
##################################################
def main():
    n = NeuralNetwork( 2, 4, 1, 0.4 )

    data = [( [0, 0], [0] ),
            ( [0, 1], [1] ),
            ( [1, 0], [1] ),
            ( [1, 1], [0] )]


    while any( output < 0.95 for output in n.trained_output( [0, 1] ) ):
        training_data = random.choice( data )
        n.train( training_data[0], training_data[1] )

    print( n.trained_output( [0, 0] ) )
    print( n.trained_output( [0, 1] ) )
    print( n.trained_output( [1, 0] ) )
    print( n.trained_output( [1, 1] ) )

    print( "Epoch: ", n.epoch )

if __name__ == "__main__":
    main()