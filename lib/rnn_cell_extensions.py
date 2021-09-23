import tensorflow as tf
from lib.config import *

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

from pkg_resources import parse_version as pv
if pv(tf.__version__) >= pv('1.2.0'):
    from tensorflow.contrib.rnn import LSTMStateTuple
else:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv


class ResidualWrapper(RNNCell):
    """Operator adding residual connections to a given cell."""

    def __init__(self, cell):
        """Create a cell with added residual connection.
        Args:
            cell: an RNNCell. The input is added to the output.
        Raises:
            TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        output, new_state = self._cell(inputs, state, scope)
        output = tf.add(output, inputs[:, :-context_len]) 
        output = tf.concat([output, inputs[:, -context_len:]], 1)  

        return output, new_state


class LinearSpaceDecoderWrapper(RNNCell):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size, w_name, b_name, scope=None):
        """Create a cell with with a linear encoder in space.
        Args:
          cell: an RNNCell. The input is passed through a linear layer.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell
        self._scope = scope
        print( 'output_size = {0}'.format(output_size) )
        print( 'state_size = {0}'.format(self._cell.state_size) )

        if isinstance(self._cell.state_size,tuple):

            insize = self._cell.state_size[-1]

            if isinstance( insize, LSTMStateTuple ):
                insize = insize.h

        else:
            insize = self._cell.state_size

        with tf.variable_scope(scope):
            self.w_out = tf.get_variable(w_name,
                [insize, output_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
            self.b_out = tf.get_variable(b_name, [output_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""
        with tf.variable_scope(self._scope):
            output, new_state = self._cell(inputs, state, scope)
            output = tf.matmul(output, self.w_out) + self.b_out

            return output, new_state
