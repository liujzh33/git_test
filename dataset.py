# Dataset Factory
# Simple factory to create different types of datasets

from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
import torch


def create_dataset(config: OmegaConf, val: bool = False):
    """
    Create dataset based on config.
    
    Args:
        config: Configuration object
        val: Whether to create validation dataset
        
    Returns:
        Dataset instance
    """
    dataset_type = config.dataset.get('type', 'robotwin')  # Default to robotwin
    
    if dataset_type == 'robotwin':
        from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'randomized_limit_per_task'):
            params['randomized_limit_per_task'] = config.dataset.randomized_limit_per_task

        # Subtask supervision ratio: fraction of each phase that gets subtask text labels
        if hasattr(config.dataset, 'subtask_supervision_ratio'):
            params['subtask_supervision_ratio'] = config.dataset.subtask_supervision_ratio
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return RobotWinTaskDataset(**params)
    

    elif dataset_type == 'robotwin_dim16':
        from .robotwin2.robotwin_agilex_dataset_dim16 import RobotWinTaskDataset as RobotWinTaskDatasetDim16

        params = {}

        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'task_names'):
            task_names_cfg = config.dataset.task_names
            if task_names_cfg is not None:
                params['task_names'] = list(task_names_cfg)
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val
        if hasattr(config.dataset, 'randomized_limit_per_task'):
            params['randomized_limit_per_task'] = config.dataset.randomized_limit_per_task

        # Subtask supervision ratio: fraction of each phase that gets subtask text labels
        if hasattr(config.dataset, 'subtask_supervision_ratio'):
            params['subtask_supervision_ratio'] = config.dataset.subtask_supervision_ratio

        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return RobotWinTaskDatasetDim16(**params)

    elif dataset_type == 'ac_one':
        from .ac_one.ac_one_dataset import ACOneDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return ACOneDataset(**params)

    elif dataset_type == 'latent_action':
        from .latent_action.latent_action_dataset import LatentActionDataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        if hasattr(config.dataset, 'dataset_dir'):
            dataset_dir = list(config.dataset.dataset_dir)
            params['dataset_dir'] = [str(p) for p in dataset_dir]
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Optional VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Optional additional params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return LatentActionDataset(**params)

    elif dataset_type == 'aloha_agilex_2':
        from .aloha_agilex_2.aloha_agilex2_dataset import AlohaAgilex2Dataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return AlohaAgilex2Dataset(**params)

    elif dataset_type == 'lerobot':
        from .lerobot.lerobot_dataset import LeRobotMotusDataset

        # Get all parameters from config
        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag
        params['val'] = val

        return LeRobotMotusDataset(**params)

    elif dataset_type == 'rc2':
        from .rc2.rc2_dataset import RC2Dataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'robot_type'):
            params['robot_type'] = config.dataset.robot_type
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Additional params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return RC2Dataset(**params)

    elif dataset_type == 'dobot':
        from .dobot.dobot_dataset import DobotDataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        params['val'] = val

        return DobotDataset(**params)

    elif dataset_type == 'multi_source_pretrain':
        from .multi_source_pretrain_dataset import MultiSourcePretrainDataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
                'max_action_dim': config.common.get('max_action_dim', 16),
            })

        # Multi-source parameters
        if hasattr(config.dataset, 'sources'):
            params['sources'] = OmegaConf.to_object(config.dataset.sources)
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        params['val'] = val

        return MultiSourcePretrainDataset(**params)

    elif dataset_type == 'g1':
        from .g1.g1_dataset import G1Dataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # G1-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        params['val'] = val

        return G1Dataset(**params)

    # Example: Add more dataset types here
    # elif dataset_type == 'bridge':
    #     from .bridge_dataset import BridgeDataset
    #     return BridgeDataset(**params)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: robotwin, ac_one, aloha_agilex_2, latent_action, lerobot, rc2, dobot, multi_source_pretrain, g1")


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    
    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask
            
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    if not language_embeddings or all(emb is None for emb in language_embeddings):
        return None

    padded_embeddings = []

    for emb in language_embeddings:
        if emb is None:
            # Determine expected dimension from first valid embedding
            valid_emb = next(e for e in language_embeddings if e is not None)
            padded = torch.zeros(text_len, valid_emb.shape[-1], dtype=valid_emb.dtype)
        elif emb.shape[0] < text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)

    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.

    Supports two text embedding fields:
    - text_embedding: Pre-computed embeddings (e.g., Cosmos) with any dimension
    - language_embedding: UMT5/T5 embeddings with 1024 dimension

    Args:
        batch: List of sample dictionaries (may contain None)

    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    # Stack tensors（支持无 initial_state 的样本）
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_frames = torch.stack([sample['video_frames'] for sample in batch])
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch])
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None

    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)

    # Process text_embedding (pre-computed, any dimension like 3584 for Cosmos)
    text_embeddings = [sample.get('text_embedding') for sample in batch]
    processed_text_embeddings = _process_language_embeddings_batch(text_embeddings)

    # Process language_embeddings (legacy, 1024 dimension for UMT5/T5)
    language_embeddings = [sample.get('language_embedding') for sample in batch]
    processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)

    # Collect text instructions (for online encoding)
    text_instructions = [sample.get('text_instruction') for sample in batch]
    processed_text_instructions = text_instructions if any(t is not None for t in text_instructions) else None

    # Collect subtask prompts
    subtask_prompts = [sample.get('subtask_prompt') for sample in batch]
    processed_subtask_prompts = subtask_prompts if any(s is not None for s in subtask_prompts) else None

    # Collect subtask_lm_inputs for Qwen3-VL native LM generation
    # Use any() instead of all(): if some samples have subtask and some don't,
    # fill missing ones with all -100 labels (no CE supervision for those samples)
    subtask_lm_inputs_list = [sample.get('subtask_lm_inputs') for sample in batch]
    processed_subtask_lm_inputs = None
    if subtask_lm_inputs_list and any(s is not None for s in subtask_lm_inputs_list):
        valid_list = [s for s in subtask_lm_inputs_list if s is not None]
        processed_subtask_lm_inputs = _process_vlm_inputs_batch(valid_list)
        # Also batch labels with padding (-100) - must match padded input_ids length
        labels_list = [s['labels'] for s in valid_list]
        max_input_len = processed_subtask_lm_inputs['input_ids'].shape[1]
        padded_labels = []
        for l in labels_list:
            if l.shape[1] < max_input_len:
                pad_size = max_input_len - l.shape[1]
                l = torch.cat([l, torch.full((1, pad_size), -100, dtype=l.dtype)], dim=1)
            padded_labels.append(l)
        valid_labels = torch.cat(padded_labels, dim=0)

        # For samples without subtask data, create all -100 labels
        num_valid = len(valid_list)
        num_total = len(subtask_lm_inputs_list)
        if num_valid < num_total:
            dummy_labels = torch.full((num_total - num_valid, max_input_len), -100, dtype=valid_labels.dtype)
            # Interleave dummy labels at the correct positions to maintain batch order
            all_labels = []
            valid_idx = 0
            for s in subtask_lm_inputs_list:
                if s is not None:
                    all_labels.append(padded_labels[valid_idx])
                    valid_idx += 1
                else:
                    all_labels.append(torch.full((1, max_input_len), -100, dtype=valid_labels.dtype))
            processed_subtask_lm_inputs['labels'] = torch.cat(all_labels, dim=0)
            # Also need to pad input_ids and attention_mask for missing samples
            # Use the first valid sample as template for missing ones
            dummy_input_ids = processed_subtask_lm_inputs['input_ids'][0:1].expand(num_total - num_valid, -1)
            dummy_attention_mask = torch.zeros(num_total - num_valid, max_input_len, dtype=processed_subtask_lm_inputs['attention_mask'].dtype)
            # Rebuild full input_ids and attention_mask in correct order
            all_input_ids = []
            all_attention_mask = []
            valid_idx = 0
            for s in subtask_lm_inputs_list:
                if s is not None:
                    all_input_ids.append(processed_subtask_lm_inputs['input_ids'][valid_idx:valid_idx+1])
                    all_attention_mask.append(processed_subtask_lm_inputs['attention_mask'][valid_idx:valid_idx+1])
                    valid_idx += 1
                else:
                    all_input_ids.append(processed_subtask_lm_inputs['input_ids'][0:1].clone())
                    all_attention_mask.append(torch.zeros(1, max_input_len, dtype=processed_subtask_lm_inputs['attention_mask'].dtype))
            processed_subtask_lm_inputs['input_ids'] = torch.cat(all_input_ids, dim=0)
            processed_subtask_lm_inputs['attention_mask'] = torch.cat(all_attention_mask, dim=0)
            # pixel_values and image_grid_thw: use valid samples only, they will be padded by _process_vlm_inputs_batch
        else:
            processed_subtask_lm_inputs['labels'] = valid_labels

    # Collect subtask decoder inputs for mot_decoder mode
    # Use any() instead of all(): fill missing samples with all -100 labels
    subtask_dec_input_ids_list = [sample.get('subtask_input_ids') for sample in batch]
    subtask_dec_labels_list = [sample.get('subtask_labels') for sample in batch]
    processed_subtask_input_ids = None
    processed_subtask_labels = None
    if subtask_dec_input_ids_list and any(s is not None for s in subtask_dec_input_ids_list):
        valid_inputs = [s for s in subtask_dec_input_ids_list if s is not None]
        valid_labels = [s for s in subtask_dec_labels_list if s is not None]
        # Get max length from valid samples
        max_len = max(s.shape[1] for s in valid_inputs)
        # Pad valid samples to max length
        padded_inputs = []
        padded_labels = []
        for inp, lab in zip(valid_inputs, valid_labels):
            if inp.shape[1] < max_len:
                pad_size = max_len - inp.shape[1]
                pad_id = 0  # pad_token_id
                inp = torch.cat([inp, torch.full((1, pad_size), pad_id, dtype=inp.dtype)], dim=1)
                lab = torch.cat([lab, torch.full((1, pad_size), -100, dtype=lab.dtype)], dim=1)
            padded_inputs.append(inp)
            padded_labels.append(lab)
        # For missing samples, create dummy all -100 labels
        num_valid = len(valid_inputs)
        num_total = len(subtask_dec_input_ids_list)
        if num_valid < num_total:
            all_inputs = []
            all_labels = []
            valid_idx = 0
            for s in subtask_dec_input_ids_list:
                if s is not None:
                    all_inputs.append(padded_inputs[valid_idx])
                    all_labels.append(padded_labels[valid_idx])
                    valid_idx += 1
                else:
                    dummy_inp = torch.zeros(1, max_len, dtype=padded_inputs[0].dtype)
                    dummy_lab = torch.full((1, max_len), -100, dtype=padded_labels[0].dtype)
                    all_inputs.append(dummy_inp)
                    all_labels.append(dummy_lab)
            processed_subtask_input_ids = torch.cat(all_inputs, dim=0)
            processed_subtask_labels = torch.cat(all_labels, dim=0)
        else:
            processed_subtask_input_ids = torch.cat(padded_inputs, dim=0)
            processed_subtask_labels = torch.cat(padded_labels, dim=0)

    # Collect progress targets
    progress_targets_list = [sample.get('progress_target') for sample in batch if sample.get('progress_target') is not None]
    processed_progress_targets = torch.stack(progress_targets_list) if progress_targets_list else None

    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W]
        'action_sequence': action_sequences,     # [B, F, D]
        'vlm_inputs': processed_vlm_inputs,
        'text_embedding': processed_text_embeddings,  # [B, seq_len, dim] or None
        'language_embedding': processed_language_embeddings,  # [B, seq_len, 1024] or None
        'text_prompts': processed_text_instructions,  # List[str] or None
        'subtask_prompts': processed_subtask_prompts,  # List[str] or None
        'subtask_lm_inputs': processed_subtask_lm_inputs,  # Dict[str, Tensor] or None
        'subtask_input_ids': processed_subtask_input_ids,  # [B, max_len] or None (for mot_decoder)
        'subtask_labels': processed_subtask_labels,        # [B, max_len] or None (for mot_decoder)
        'progress_targets': processed_progress_targets,  # [B, 1] or None
    }

    if initial_states is not None:
        result['initial_state'] = initial_states

    return result