import torch
import argparse
import os
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_mp_zero_path", required=False, default="/datap/SeleneExperiments/Comparison3b/3b_VocabParallel/3bModel/checkpoints/mp_rank_00/megatron_t5_prompt_tune--val_loss=5.676-step=72000.ckpt", help="path to checkpoint file in mp_rank_00")
    parser.add_argument("--language_model_path_tp2", required=False, default="/datap/misc/Checkpoints/t5_3b/megatron_t5_expanded_vocab_posemb1536.nemo", help="path to language model")
    parser.add_argument("--language_model_path_tp1", required=False, default="/datap/misc/Checkpoints/t5_3b_1tp/megatron_t5_expanded_vocab_posemb1536.nemo", help="path to language model")
    parser.add_argument("--model_class", required=False, default="nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_pretrain_model.MegatronT5SpeechLMModel", help="Speech LM Model Class")
    args = parser.parse_args()
    
    # Eg. checkpoint_path = "/datap/SeleneExperiments/Comparison3b/3b_VocabParallel/3bModel/checkpoints/mp_rank_00/megatron_t5_prompt_tune--val_loss=5.676-step=72000.ckpt"
    checkpoint_path = args.checkpoint_mp_zero_path

    # nemo_base_dir is /datap/SeleneExperiments/Comparison3b/3b_VocabParallel/3bModel/checkpoints/
    nemo_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    # nemo_extracted_dir is /datap/SeleneExperiments/Comparison3b/3b_VocabParallel/3bModel/checkpoints/nemo_extracted
    nemo_extracted_dir = os.path.join(nemo_base_dir, "nemo_extracted")
    rank0_weights_dir = os.path.join(nemo_extracted_dir, "mp_rank_00")
    rank1_weights_dir = os.path.join(nemo_extracted_dir, "mp_rank_01")
    hparams_yaml_path =  os.path.join(os.path.dirname(nemo_base_dir), "version_0/hparams.yaml")
    
    hparams = OmegaConf.load(hparams_yaml_path)
    assert os.path.exists(hparams_yaml_path), f"{hparams_yaml_path} does not exist"
    models_hparams = hparams.cfg
    models_hparams['language_model_path'] = args.language_model_path_tp2
    
    if not os.path.exists(rank0_weights_dir):
        os.makedirs(rank0_weights_dir)
    if not os.path.exists(rank1_weights_dir):
        os.makedirs(rank1_weights_dir)
    
    model_config_save_path = os.path.join(nemo_extracted_dir, "model_config.yaml")
    OmegaConf.save(models_hparams, model_config_save_path)
    
    print("Loading mp rank 0 checkpoint")
    rank0_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Loading mp rank 1 checkpoint")
    rank1_checkpoint = torch.load(checkpoint_path.replace("mp_rank_00", "mp_rank_01"), map_location="cpu")

    rank0_model = rank0_checkpoint["state_dict"]
    rank1_model = rank1_checkpoint["state_dict"]

    print("Saving mp rank 0 checkpoint")
    torch.save(rank0_model, os.path.join(rank0_weights_dir, "model_weights.ckpt"))
    print("Saving mp rank 1 checkpoint")
    torch.save(rank1_model, os.path.join(rank1_weights_dir, "model_weights.ckpt"))
    print("Done")

    target_nemo_filepath = os.path.join(nemo_base_dir, "SpeechLMT5_tp1.nemo")
    command_str = "python examples/nlp/language_modeling/megatron_change_num_partitions.py --model_extracted_dir={} --target_file={} --model_class={} --tensor_model_parallel_size=2 --target_tensor_model_parallel_size=1 --pipeline_model_parallel_size=1 --target_pipeline_model_parallel_size=1 --target_pipeline_model_parallel_split_rank=0 --speech_model --precision=bf16 --tp1_language_model_path {}".format(
        nemo_extracted_dir, 
        target_nemo_filepath, 
        args.model_class,
        args.language_model_path_tp1
        )
    print("TP2 Nemo Directory Setup. Now Run the following command:\n")
    print(command_str)


if __name__ == "__main__":
    main()