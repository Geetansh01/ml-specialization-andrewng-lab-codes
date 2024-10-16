import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, sigmoid, relu

def test_my_softmax(target):
    z = np.array([1., 2., 3., 4.])
    a = target(z)
    atf = tf.nn.softmax(z)
    
    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"
    
    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    atf = tf.nn.softmax(z)
    
    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"
    
    print("\033[92m All tests passed.")

# This is the original test_model, it was not working!  
# def test_model(target, classes, input_size):
#     target.build(input_shape=(None,input_size))
    
#     assert len(target.layers) == 3, \
#         f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
#     assert target.input.shape.as_list() == [None, input_size], \
#         f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
#     i = 0
#     expected = [[Dense, [None, 25], relu],
#                 [Dense, [None, 15], relu],
#                 [Dense, [None, classes], linear]]

#     for layer in target.layers:
#         assert type(layer) == expected[i][0], \
#             f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
#         assert layer.output.shape.as_list() == expected[i][1], \
#             f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
#         assert layer.activation == expected[i][2], \
#             f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
#         i = i + 1

#     print("\033[92mAll tests passed!")


# Modified test_model
def test_model(target, classes, input_size):
    # Build the model to ensure input shape is properly initialized
    target.build(input_shape=(None, input_size))

    # Check the number of layers
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    
    # Check the input shape
    assert target.input_shape == (None, input_size), \
        f"Wrong input shape. Expected (None, {input_size}) but got {target.input_shape}"
    
    # Define the expected structure for comparison
    expected = [
        [Dense, (None, 25), relu],
        [Dense, (None, 15), relu],
        [Dense, (None, classes), linear]
    ]

    # Verify each layer
    for i, layer in enumerate(target.layers):
        # Check the type of the layer
        assert isinstance(layer, expected[i][0]), \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        
        # Check the output shape
        assert layer.output.shape == expected[i][1], \
            f"Wrong output shape in layer {i}. Expected {expected[i][1]} but got {layer.output.shape}"
        
        # Check the activation function
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"

    print("\033[92mAll tests passed!")



    