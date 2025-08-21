import jax
import jax.numpy as jnp
import onnx
from jax.experimental import jax2tf
import tensorflow as tf
import flax.nnx as nnx
from openpi.models.pi0 import Pi0
from openpi.models import model as _model
from openpi.training import config as _config

def create_inference_fn(model: Pi0):
    """Creates a pure inference function for ONNX export."""
    
    # 提取模型的图结构和状态
    graphdef, frozen_state = nnx.split(model)
    
    @jax.jit  # 保持与原始代码相同的 jit 行为
    def inference_fn(
        rng: jnp.ndarray,  # shape: (2,)
        images_base: jnp.ndarray,  # shape: (B, H, W, 3)
        images_left: jnp.ndarray, 
        images_right: jnp.ndarray,
        state: jnp.ndarray,        # shape: (B, action_dim)
        tokens: jnp.ndarray,       # shape: (B, max_token_len)
        token_mask: jnp.ndarray,   # shape: (B, max_token_len)
    ):
        # 重构 Observation
        obs = Observation(
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
        module = nnx.merge(graphdef, frozen_state)
        return module.sample_actions(rng, obs)
    
    return inference_fn

def export_to_onnx(model: Pi0, save_path: str, batch_size: int = 1):
    """Export Pi0 model to ONNX format, preserving module_jit behavior."""
    
    inference_fn = create_inference_fn(model)
    
    # 创建输入规范，注意数据类型要与实际使用匹配
    input_specs = [
        tf.TensorSpec([2], tf.uint32),  # rng
        tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),  # base image
        tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),  # left image
        tf.TensorSpec([batch_size, 480, 640, 3], tf.float32),  # right image
        tf.TensorSpec([batch_size, model.action_dim], tf.float32),  # state
        tf.TensorSpec([batch_size, model.max_token_len], tf.int32),  # tokens
        tf.TensorSpec([batch_size, model.max_token_len], tf.bool),  # token mask
    ]

    #method1: Convert to ONNX directly
    input_specs = [(2, ), (batch_size, 480, 640, 3), (batch_size, 480, 640, 3), (batch_size, 480, 640, 3), (batch_size, model.action_dim), (batch_size, model.max_token_len), (batch_size, model.max_token_len)] 
    from jax2onnx import to_onnx
    onnx_model = to_onnx(inference_fn, input_specs)
    print("jax2onnx finished")

    # Save the model
    onnx.save_model(onnx_model, f"{save_path}/my_callable.onnx")
    print("save onnx finished")
    import sys
    sys.exit(0)
    
    #method2: Convert to tensorflow and then convert to onnx
    tf_fn = jax2tf.convert(
        inference_fn,
        enable_xla=False,
        with_gradient=False,
        polymorphic_shapes=None  # 保持静态形状以便更好的优化
    )
    
    tf_func = tf.function(tf_fn, input_signature=input_specs, autograph=False)
    print("jax2tf finished")
    
    # 转换为 ONNX
    import tf2onnx
    model_proto, _ = tf2onnx.convert.from_function(
        tf_func,
        input_signature=input_specs,
        opset=13,
    )
    print("tf2onnx finished")
    
    # 保存 ONNX 模型
    with open(save_path, 'wb') as f:
        f.write(model_proto.SerializeToString())

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
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float32))
    
    # 导出为 ONNX
    export_to_onnx(model, output_path, batch_size=1)
