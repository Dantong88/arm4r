from typing import Union
from arm4r.util.args import ModelConfig, SharedConfig, VisionEncoderConfig, PolicyConfig
from arm4r.models.backbones.encoders import VisionEncoder, VisionEncoderCNN

def vision_encoder_constructor(
    vision_encoder_cfg : VisionEncoderConfig,
) -> Union[VisionEncoder, VisionEncoderCNN]: 
    """Instantiate a vision encoder based on the vision_encoder_cfg 

    Return the vision encoder instance
    """
    if vision_encoder_cfg.vision_unfreeze_all:
        vision_finetune = "all"

    elif vision_encoder_cfg.vision_unfreeze_last_n > 0:
        vision_finetune = vision_encoder_cfg.vision_unfreeze_last_n
    
    elif vision_encoder_cfg.vision_lora:
        vision_finetune = "lora"
    
    else:
        vision_finetune = False
    
    vision_pretrained = not vision_encoder_cfg.vision_nonpretrained

    vision_encoder_name = vision_encoder_cfg.vision_encoder
    print(f"using vision encoder {vision_encoder_name}")
    vision_encoder_cls = VisionEncoderCNN if "resnet" in vision_encoder_name.lower() else VisionEncoder
    vision_encoder = vision_encoder_cls(
        vision_encoder_name, pretrained=vision_pretrained, 
        global_pool="", finetune=vision_finetune, lora_rank=vision_encoder_cfg.vision_lora_rank,
    )

    return vision_encoder

def policy_constructor(
    policy_cfg : PolicyConfig, 
    shared_config : SharedConfig,
    vision_encoder : Union[VisionEncoder, VisionEncoderCNN], 
    train : bool = True, 
):
    """
    Instantiate a policy based on the policy_cfg
    """
    # # proprio and action dim
    # proprio_dim = 10 if shared_config.rot_6d else 8
    # action_dim = 11 if shared_config.rot_6d else 8

    # proprio and action dim
    proprio_dim = policy_cfg.proprio_dim
    action_dim = policy_cfg.action_dim

    # find the phase of the training
    if policy_cfg.task == 'robotics':
        from arm4r.models.policy.arm4r import ARM4R
    else:
        assert policy_cfg.task == '3dpoints'
        from arm4r.models.policy.arm4r_rlbench_pretrain_mask import ARM4R
    model = ARM4R(
        llama_ckpt_dir=policy_cfg.llama_ckpt_dir, 
        vision_encoder=vision_encoder,
        phase=policy_cfg.phase, 
        num_cameras=shared_config.num_cameras, 
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        adapter_mlp_ratio=policy_cfg.adapter_mlp_ratio, 
        adapter_num_heads=policy_cfg.adapter_num_heads,
        multikv_attn_pool=policy_cfg.multikv_attn_pool,
        loss_w_action=policy_cfg.loss_w_action,    
        lora_rank=policy_cfg.lora_rank,
        camera_pos_emb=policy_cfg.camera_pos_emb,
        modality_pos_emb=policy_cfg.modality_pos_emb,
        separate_camera_adapter=policy_cfg.separate_camera_adapter, 
        seq_length=shared_config.seq_length,
        rot_6d=shared_config.rot_6d,
        train=train,
        max_batch_size=shared_config.batch_size,
        num_pred_steps=shared_config.num_pred_steps,
        pred_action_only=policy_cfg.pred_action_only,
        remove_proprio=policy_cfg.remove_proprio,
        no_prompt_loss=policy_cfg.no_prompt_loss,
        decoder_pred_head=policy_cfg.decoder_pred_head,
        use_delta_action=shared_config.use_delta_action,
        kl_div_loss=policy_cfg.kl_div_loss,
        scale_loss=policy_cfg.scale_loss,
        load_llama=policy_cfg.load_llama,
        step_weight=policy_cfg.step_weight,
        scratch_llama_config=policy_cfg.scratch_llama_config,
        num_train_diffusion_steps=policy_cfg.num_train_diffusion_steps,
        num_inference_diffusion_steps=policy_cfg.num_inference_diffusion_steps,
        scale_action=shared_config.scale_action,
    )
    return model 

def model_constructor(
    model_config : ModelConfig, 
    shared_config : SharedConfig,
    train : bool = True, 
):
    vision_encoder = vision_encoder_constructor(model_config.vision_encoder_cfg)
    policy = policy_constructor(model_config.policy_cfg, shared_config, vision_encoder, train=train)
    return policy