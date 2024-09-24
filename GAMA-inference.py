import os
import sys


import torch
import torchaudio
import numpy as np
import fire
import time
import torchvision

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils.prompter import Prompter

import json
import random
import librosa
import argparse
from tqdm import tqdm
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed):
    """
    Set the random seed for all random number generators.

    Args:
        seed: The random seed to use.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """
    Parse command line arguments for inference.

    Args:
        p_dataset (Path): Path to the dataset folder. Defaults to /livingrooms/wcchen/Dynamic_Datasets/.
        p_results (Path): Path to the results folder. Required.
        p_save (Path): Path to the save folder. Required.
        seed (int): The random seed. Defaults to 33.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_dataset', type=Path, default='/livingrooms/wcchen/Dynamic_Datasets/')
    parser.add_argument('--p_save', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=33)
    return parser.parse_args()

def get_pretrained_model(
    model_name: str ="/workspace/GAMA/GAMMA-data-models/fs/nexus-projects/brain_project/Llama-2-7b-chat-hf-qformer",
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
):
    """
    """
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto") #, torch_dtype=torch.bfloat16
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # change it to your model path, this is the GAMMA state 5 model trained 1 epoch as in paper results.
    eval_mdl_path = '/workspace/GAMA/GAMMA-data-models/checkpoint-1100/pytorch_model.bin' #location on singularity image of the bin model.
    state_dict = torch.load(eval_mdl_path, map_location='cpu')
    _ = model.load_state_dict(state_dict, strict=False)

    model.is_parallelizable = True
    model.model_parallel = True

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    return model, prompter, tokenizer

def read_wavs(p_wav):
    """
    Read all the wave files in the same directory as `p_wav`, including itself.
    
    The function assumes that the wave files are named as follows:
    
    - The first wave file is named as `p_wav`.
    - The second wave file is named as `p_wav.stem + "_pair2." + p_wav.suffix`.
    - The third wave file is named as `p_wav.stem + "_pair3." + p_wav.suffix`.
    - And so on.
    
    The function will return a list of strings, where each string is the path to a wave file.
    """
    wavs = [str(p_wav)]
    index = 2
    while True:
        pair_wav_key = '{}_pair{}.{}'.format(p_wav.stem, index, 'wav')
        pair_wav_path = p_wav.parent / pair_wav_key
        if pair_wav_path.exists():
            wavs.append(str(pair_wav_path))
            index += 1
        else:
            break
    return wavs


def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    return fbank


def main():
    args = parse_args()
    set_random_seed(args.seed)

    model_name = '/workspace/GAMA/GAMMA-data-models/fs/nexus-projects/brain_project/Llama-2-7b-chat-hf-qformer'
    model, prompter, tokenizer = get_pretrained_model(model_name)
    temp, top_p, top_k = 0.1, 0.95, 500

    
    for metafile in tqdm(args.p_dataset.glob('*/metadata.json')):
        metadata = json.load(metafile.open('r'))
        taskname = metafile.parent.name
        print('Processing {}'.format(taskname))
        
        savefile = args.p_save / "GAMA_IT" / '{}.json'.format(taskname)
        savefile.parent.mkdir(parents=True, exist_ok=True)

        if savefile.exists():
            continue

        
        if taskname in [
            # 'PhonologicalFeatureClassification_VoxAngeles-Phone',
            'Emergency_traffic_detection_ETD',
        ]:
            print('Skip {}'.format(taskname))
            continue

        for pwav, example in tqdm(metadata.items()):
            p_wav = args.p_dataset / taskname / pwav
            # Read all the wave files
            wavs = read_wavs(p_wav)

            if len(wavs) > 1:
                print("Exiting script because len(wavs) > 1")
                sys.exit()

            ### on the meantime I will process audio one by one but this is inneficient, yet I worry about GPU memory  ###
            cur_audio_input = load_audio(wavs[0]).unsqueeze(0).to("cuda")
            
            if 'text' in example:
                example['instruction'] = 'The text is: {}\n{}'.format(example['text'], example['instruction'])
            
            
            prompt = prompter.generate_prompt(example['instruction'], None)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to("cuda")

            generation_config = GenerationConfig(
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            max_new_tokens=400,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            num_return_sequences=1
            )

            # Without streaming

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids.to(device),
                    audio_input=cur_audio_input,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=400,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)[6:-4]
            output = output[len(prompt):]


            # Record response
            metadata[pwav]['sllm_name'] = "GAMA_IT"
            metadata[pwav]['sllm_response'] = output

        # Save the results
        json.dump(metadata, savefile.open('w'), indent=4, ensure_ascii=False)
        print('Done {}'.format(taskname))


if __name__ == '__main__':
    main()
