import jax
import argparse
import jax.numpy as jnp
import numpy as np
import onnx
from jax.experimental import jax2tf
import tensorflow as tf
from flax import traverse_util
import flax.nnx as nnx
from openpi.models.pi0 import Pi0
from openpi.models import model as _model
from openpi.training import config as _config

def create_inference_fn(model: Pi0):
    """Creates a pure inference function for ONNX export."""
    
    # 提取模型的图结构和状态
    graphdef, frozen_state = nnx.split(model)
    print("create inference fn1 ")
    
    def inference_fn(
        params, 
        rng: jnp.ndarray,  # shape: (2,)
        images_base: jnp.ndarray,  # shape: (B, H, W, 3)
        images_left: jnp.ndarray, 
        images_right: jnp.ndarray,
        state: jnp.ndarray,        # shape: (B, action_dim)
        tokens: jnp.ndarray,       # shape: (B, max_token_len)
        token_mask: jnp.ndarray,   # shape: (B, max_token_len)
    ):
        # 重构 Observation
        obs = _model.Observation(
            images={
                "base_0_rgb": images_base,
                "left_wrist_0_rgb": images_left,
                "right_wrist_0_rgb": images_right,
            },
            image_masks={
                "base_0_rgb": jnp.ones((images_base.shape[0],), dtype=jnp.bool),
                "left_wrist_0_rgb": jnp.ones((images_left.shape[0],), dtype=jnp.bool),
                "right_wrist_0_rgb": jnp.ones((images_right.shape[0],), dtype=jnp.bool),
            },
            state=state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_mask
        )
        
        # 重建模型并调用推断
        module = nnx.merge(graphdef, params)
        return module.sample_actions(rng, obs)
    print("create inference fn2 ")
    return inference_fn, frozen_state

def jax2onnx(inference_fn, save_path, batch_size):
    input_specs = [(2, ), (batch_size, 480, 640, 3), (batch_size, 480, 640, 3), (batch_size, 480, 640, 3), (batch_size, model.action_dim), (batch_size, model.max_token_len), (batch_size, model.max_token_len)] 
    from jax2onnx import to_onnx
    onnx_model = to_onnx(inference_fn, input_specs)
    print("jax2onnx finished")

    # Save the model
    onnx.save_model(onnx_model, f"{save_path}/my_callable.onnx")
    print("save onnx finished")

def jax2tf_saved_model(inference_fn, params, save_path, batch_size, action_dim, max_token_len):
    """Convert JAX function to TensorFlow and then to ONNX."""
    # This function is not used in the final export, but can be useful for debugging.
    def extract_value(p):
        if isinstance(p, (dict, nnx.State)):
            return {k: extract_value(v) for k, v in p.items()}
        elif isinstance(p, nnx.variablelib.VariableState):
            return p.value
        return p   

    params_plain = extract_value(params)
    # print("params_plain:", params_plain)
    print("get value finished")

    params_vars = tf.nest.map_structure(tf.Variable, params_plain)
    del params_plain
    # print("params_vars:", params_vars)
    print("to tf variable finished")
    
    input_specs = [
        tf.TensorSpec([2], tf.uint32),  # rng
        tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),  # base image
        tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),  # left image
        tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),  # right image
        tf.TensorSpec([batch_size, action_dim], tf.float32),  # state
        tf.TensorSpec([batch_size, max_token_len], tf.int32),  # tokens
        tf.TensorSpec([batch_size, max_token_len], tf.bool),  # token mask
    ]
    my_model = tf.Module()
    my_model._variables = tf.nest.flatten(params_vars)
    """
    predict_fn = jax2tf.convert(inference_fn, native_serialization=False, with_gradient=False)
    @tf.function(autograph=False, jit_compile=True, input_signature=input_specs)
    def predict_tf(*args):
        return predict_fn(params_vars, *args)
    my_model.f = predict_tf
    """
    prediction_tf = lambda *inputs: jax2tf.convert(inference_fn, native_serialization=False, with_gradient=False)(params_vars, *inputs)
    my_model.f = tf.function(prediction_tf, jit_compile=True, autograph=False, input_signature=input_specs)
    tf.saved_model.save(my_model, f'{save_path}/tf_model', options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

    # class SavedModel(tf.Module):
    #     def __init__(self, params_vars):
    #         super().__init__()
    #         # 将参数作为类属性来确保追踪
    #         # to_tf_virable = lambda x: tf.Variable(x, trainable=False, dtype=tf.as_type(x)) if not isinstance(x, tf.Variable) else x
    #         self.params_vars = tf.nest.map_structure(tf.Variable, params_vars)
        
    #     @tf.function(autograph=False, jit_compile=True)
    #     def __call__(self, rng, images_base, images_left, images_right, state, tokens, token_mask):
    #         predict_fn = jax2tf.convert(inference_fn, enable_xla=True, with_gradient=False)
    #         return predict_fn(self.params_vars, rng, images_base, images_left, images_right, 
    #                         state, tokens, token_mask)

    # savemodel = SavedModel(params_vars)
    # input_specs = {
    #     'rng': tf.TensorSpec([2], tf.uint32),
    #     'images_base': tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),
    #     'images_left': tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),
    #     'images_right': tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),
    #     'state': tf.TensorSpec([batch_size, action_dim], tf.float32),
    #     'tokens': tf.TensorSpec([batch_size, max_token_len], tf.int32),
    #     'token_mask': tf.TensorSpec([batch_size, max_token_len], tf.bool),
    # }
    
    # # 创建具体的签名
    # signatures = {
    #     'serving_default': savemodel.__call__.get_concrete_function(**input_specs)
    # }
    # tf.saved_model.save(
    #     savemodel, 
    #     f'{save_path}/tf_model',
    #     signatures=signatures,
    #     options=tf.saved_model.SaveOptions(experimental_custom_gradients=True)
    # )
    print("jax2tf finished")

def export_to_onnx(model: Pi0, save_path: str, batch_size: int = 1):
    """Export Pi0 model to ONNX format, preserving module_jit behavior."""
    
    inference_fn, params = create_inference_fn(model)
    print("create inference fn finished")
    
    # method1: Convert to ONNX directly
    # jax2onnx(inference_fn, save_path, batch_size)
    
    # method2: Convert to tensorflow and then convert to onnx
    jax2tf_saved_model(inference_fn, params, save_path, batch_size, model.action_dim, model.max_token_len)
    
    # python -m tf2onnx.convert --saved-model /dev/shm/tmp/tf_model --output /dev/shm/pi0_galaxea_lora.onnx --opset 21 --large_model --verbose

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge datasets from multiple sources.")
    parser.add_argument("--checkpoint_config_name", type=str, default="pi0_galaxea_lora", help="checkpoint_dir_config_name")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="checkpoint_dir")
    parser.add_argument("--output_dir", type=str, default="", help="output_dir")

    args = parser.parse_args()
    import pathlib
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_config = args.checkpoint_config_name 
    output_path = args.output_dir
    train_config = _config.get_config(checkpoint_config)
    print("args.checkpoint_dir:", args.checkpoint_dir)
    print("args.checkpoint_config_name:", args.checkpoint_config_name)
    print("args.output_dir:", args.output_dir)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float16))
    print("model:", model.__ne__)
    
    # 导出为 ONNX
    export_to_onnx(model, output_path, batch_size=1)
