# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import jieba
import tqdm

import argparse
import json
import os
import numpy as np

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

prompt_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )

def generate_prompt(instruction, input=None):
        if input:
            instruction = instruction + '\n' + input
        return prompt_input.format_map({'instruction': instruction})



def main():
    parser = argparse.ArgumentParser()
    # ... (same as before)
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--output_file', default='./output.json', type=str)
    
    args = parser.parse_args()
    # ... (same as before)
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=400
    )

    # The prompt template below is taken from llama.cpp
    # and is slightly different from the one used in training.
    # But we find it gives better results
    prompt_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )

    sample_data = ["乙肝和丙肝的区别？"]

    def generate_prompt(instruction, input=None):
        if input:
            instruction = instruction + '\n' + input
        return prompt_input.format_map({'instruction': instruction})

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )

    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()


    model.eval()
    # 加载评估数据集
    with open(args.data_file, 'r') as f:
        data = json.load(f)

    # # 初始化模型和分词器
    # model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    # model = model_class.from_pretrained(
    #     args.base_model,
    #     torch_dtype=torch.float32,  # 使用float32进行评估
    #     trust_remote_code=True,
    # ).to(device)





    def evaluate_model(model, tokenizer, data, device):
        metrics = {
            "BLEU": 0,
            "ROUGE-1": 0,
            "ROUGE-2": 0,
            "ROUGE-L": 0,
        }
        with torch.no_grad():
            print('length of data:', len(data))
            print('data[0]:', data[0])
            for item in tqdm.tqdm(data):
                question = item["instruction"]
                reference_answer = item["output"]
                input_content = item['input']
            
                
                if args.with_prompt is True:
                    input_text = generate_prompt(instruction=question, input=input_content)
                else:
                    input_text = question + input_content if input_content else question



                inputs = tokenizer(input_text, return_tensors="pt")
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    bos_token_id=tokenizer.bos_token_id,
                    # eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)

                if args.with_prompt:
                    generated_response = output.split("### Response:")[1].strip()
                else:
                    generated_response = output
                
                print('generated_response:', generated_response)

                # 移除参考答案中的标记，如果存在
                reference_answer = reference_answer.replace("reference", "")
                # 计算BLEU分数
                reference = ' '.join(jieba.cut(reference_answer))
                hypothesis = ' '.join(jieba.cut(generated_response))
                metrics['BLEU'] += sentence_bleu([reference], hypothesis)
                # 计算ROUGE分数
                rouge = Rouge()
                scores = rouge.get_scores(hypothesis, reference)[0]
                metrics['ROUGE-1'] += scores['rouge-1']['f']
                metrics['ROUGE-2'] += scores['rouge-2']['f']
                metrics['ROUGE-L'] += scores['rouge-l']['f']
            
        metrics['BLEU'] /= len(data)
        metrics['ROUGE-1'] /= len(data)
        metrics['ROUGE-2'] /= len(data)
        metrics['ROUGE-L'] /= len(data)
        return metrics

    # 评估模型
    metrics = evaluate_model(model, tokenizer, data, device)

    # 打印评估结果
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 可选地，将评估结果保存到文件
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()