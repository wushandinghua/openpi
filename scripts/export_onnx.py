import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
import tensorflow as tf
import flax.nnx as nnx
from openpi.models import Pi0
from openpi.models import model as _model
from openpi.training import config as _config

def create_inference_fn(model: Pi0):
    """Creates an inference function that handles module_jit correctly."""
    
    # 从模型中提取图定义和状态
    graphdef, frozen_state = nnx.split(model)
    
    # 创建内部函数用于 jit 编译
    @jax.jit
    def jitted_fn(rng, obs, num_steps=10):
        module = nnx.merge(graphdef, frozen_state)
        return module.sample_actions(rng, obs, num_steps=num_steps)
    
    def inference_fn(
        rng,
        images_base,
        images_left,
        images_right,
        state,
        tokens,
        token_mask,
    ):
        # 重构 Observation 对象
        obs = _model.Observation(
            images={
                "base_0_rgb": images_base,
                "left_wrist_0_rgb": images_left,
                "right_wrist_0_rgb": images_right,
            },
            image_masks={
                "base_0_rgb": jnp.ones((images_base.shape[0],), dtype=bool),
                "left_wrist_0_rgb": jnp.ones((images_left.shape[0],), dtype=bool),
                "right_wrist_0_rgb": jnp.ones((images_right.shape[0],), dtype=bool),
            },
            state=state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_mask
        )
        
        # 使用 jitted 函数
        return jitted_fn(rng, obs)
    
    return inference_fn

def export_to_onnx(model: Pi0, save_path: str, batch_size: int = 1):
    """Export Pi0 model to ONNX format, preserving module_jit behavior."""
    
    inference_fn = create_inference_fn(model)
    
    # 创建输入规范，注意数据类型要与实际使用匹配
    input_specs = [
        tf.TensorSpec([2], tf.float32),  # rng
        tf.TensorSpec([batch_size, 256, 256, 3], tf.float32),  # base image
        tf.TensorSpec([batch_size, 256, 256, 3], tf.float32),  # left image
        tf.TensorSpec([batch_size, 256, 256, 3], tf.float32),  # right image
        tf.TensorSpec([batch_size, model.action_dim], tf.float32),  # state
        tf.TensorSpec([batch_size, model.max_token_len], tf.int32),  # tokens
        tf.TensorSpec([batch_size, model.max_token_len], tf.bool),  # token mask
    ]
    
    # 转换为 TF 函数，使用与 module_jit 相同的配置
    tf_fn = jax2tf.convert(
        inference_fn,
        enable_xla=True,
        polymorphic_shapes=None  # 保持静态形状以便更好的优化
    )
    
    tf_func = tf.function(tf_fn, input_signature=input_specs)
    
    # 转换为 ONNX
    import tf2onnx
    model_proto, _ = tf2onnx.convert.from_function(
        tf_func,
        input_names=['rng', 'image_base', 'image_left', 'image_right', 
                    'state', 'tokens', 'token_mask'],
        output_names=['actions'],
        opset=13,
    )
    
    # 保存 ONNX 模型
    with open(save_path, 'wb') as f:
        f.write(model_proto.SerializeToString())

if __name__ == "__main__":
    checkpoint_config = ""
    checkpoint_dir = ""
    output_path = ""
    train_config = _config.get_config(checkpoint_config)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float16))
    
    # 导出为 ONNX
    export_to_onnx(model, output_path, batch_size=1)