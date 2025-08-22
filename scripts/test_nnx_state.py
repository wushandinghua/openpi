import tensorflow as tf
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import numpy as np

state = nnx.State({
    'layer1': {
        'weights': nnx.variablelib.VariableState( type=nnx.Param, value=jnp.array([[1.0, 2.0], [3.0, 4.0]],  dtype=jnp.float32)),
        'bias': nnx.variablelib.VariableState( type=nnx.Param, value=jnp.array([0.1, 0.2], dtype=jnp.float32))
    },
    'layer2': {
        'weights': nnx.variablelib.VariableState( type=nnx.Param, value=jnp.array([[5.0, 6.0], [7.0, 8.0]],  dtype=jnp.float32)),
        'bias': nnx.variablelib.VariableState( type=nnx.Param, value=jnp.array([0.3, 0.4],  dtype=jnp.float32))
    }
})

#state_vars = tf.nest.map_structure(tf.Variable, state)
def extract_value(p):
    if isinstance(p, (dict, nnx.State)):
        return {k: extract_value(v) for k, v in p.items()}
    elif isinstance(p, nnx.variablelib.VariableState):
        return p.value
    return p
params_plain = extract_value(state)
print(params_plain)

def to_tf_variable(x):
    if isinstance(x, (float, int, bool, list, tuple)):
        return tf.Variable(x)
    elif isinstance(x, dict):
        return {k: to_tf_variable(v) for k, v in x.items()}
    elif isinstance(x, (jax.Array)):
        return tf.Variable(tf.convert_to_tensor(np.asarray(x, copy=False)))
    return x
params_vars = to_tf_variable(params_plain) 
print("params_vars:", params_vars)
