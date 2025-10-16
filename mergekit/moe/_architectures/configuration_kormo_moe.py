# <저장된_모델_경로>/configuration_kormo_moe.py

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class KORMoMoeConfig(PretrainedConfig):
    model_type = "kormo_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size=112576,
        hidden_size=6144,
        intermediate_size=21504,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        attention_bias=False,
        attention_dropout=0.0,
        rope_scaling=None,
        mlp_bias=False,
        head_dim=128,
        # MoE specific
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=None,
        shared_expert_intermediate_size=None,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.mask_type = None
        
        # MoE specific
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )