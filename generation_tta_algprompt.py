import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig,AutoModelForCausalLMWithValueHead
import argparse
import numpy as np
import wandb
import copy
import random
import heapq
import utils
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset
from peft import LoraConfig
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',type=str,default='google/gemma-1.1-7b-it')
    parser.add_argument('--task',type=str,default='generation')
    parser.add_argument('--dataset',type=str,default='squad')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='/mnt/sdb/llm/')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_prompt_length',type=int,default=100)
    parser.add_argument('--train_data_per_labels',type=int,default=10)
    parser.add_argument('--num_example',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I want to give the appropriate instruction to help
                        a friend who needs to look at the input and guess the output.
                        Plase write instruction to help my friends. Here are the input-output pairs:
                        ''',)
    parser.add_argument('--prompt_per_example',type=int,default=4)
    parser.add_argument('--learning_rate',type=float,default=1e-5)
    args = parser.parse_args()
    return args

def main():
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_flash_sdp(False)
    args = parser_args()
    device=  'cuda:0'
    agent_name = args.agent_model.split('/')[-1]
    target_name = args.target_model.split('/')[-1]
    wandb.init(project='tta_' + args.dataset + '_' + agent_name + '_' + target_name, 
               config=args,
               name = args.task + '_' + args.dataset + '_' + args.agent_model + '_' + args.target_model)
    
    

    
    #load dataset
    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        train_dataset,validation_dataset = utils.create_balanced_subset_and_validation(train_dataset,
                                                                                       args.train_data_per_labels * num_labels,
                                                                                       )
    elif args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = train_dataset
    
    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        verbalizer = None
        validation_dataset = train_dataset
    
    else:
        #TODO
        pass
    
    print('Verbalizer : ', verbalizer)        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = 1,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = 1,shuffle = True)
    
    #load agent model
    config = PPOConfig(
        model_name = args.agent_model,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        mini_batch_size= args.batch_size,
        log_with='wandb',
    )
    lora_config = LoraConfig(
        r= 16,
        lora_alpha = 32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,cache_dir = args.cache_dir)
    agent_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    ppo_trainer = PPOTrainer(config,agent_model,ref_model,agent_tokenizer)
    
    #load target model
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,cache_dir = args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,
                                                        torch_dtype=torch.bfloat16,
                                                        cache_dir = args.cache_dir,
                                                        device_map='auto')
    target_model.config.pad_token_id = target_tokenizer.eos_token_id
    target_tokenizer.pad_token = target_tokenizer.eos_token
    
    #generation kwargs setting
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": agent_tokenizer.eos_token_id,
    "max_new_tokens":args.max_prompt_length,
    "min_length": -1,
    }
    
    
    #setting verbalizer ids
    verbalizer_ids=  []
    if verbalizer is not None:
        for i in range(len(verbalizer)):
            verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    
    
    
    test_acc = 0
    test_total = 0
    print('start test')
    # 랜덤하게 5개의 배치 인덱스를 선택합니다.
    random_batches = random.sample(range(len(test_dataloader)), 5)
    batch_count = 0
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
            query_text = [
                {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                {"role":"assistant","content" : "Sure. Let me see what input friend sees "},
                {"role" : "user", "content" : "The input that friend sees : " + inputs[0]},
                {"role": "assistant","content" : "The Instruction is : '"}
            ]
            query_encoded = agent_tokenizer.apply_chat_template(
                query_text,
                return_tensors='pt'
            ).view(-1).to('cuda:0')
            response_tensors = ppo_trainer.generate(
                query_encoded,
                **generation_kwargs,
                return_prompt=False,
                num_return_sequences = 1
            )
            used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            prompt = used_prompt[0]
            template = prompt + "\nInput : " + inputs[0] + "Output : "
            # 선택된 배치 인덱스일 때만 template을 출력합니다.
            if batch_count in random_batches:
                print(template)
            prompt_encoded = target_tokenizer(template,return_tensors='pt').to(device)
            outputs = target_model(**prompt_encoded)
            logits = outputs.logits
            verbalizer_logits = logits[:, -1, verbalizer_ids]
            label= labels
            if torch.argmax(verbalizer_logits).item() == label:
                test_acc += 1
            test_total += 1
            batch_count += 1
    print('Test Accuracy : ', test_acc / test_total)
    wandb.log({
        'test_acc' : test_acc / test_total
    })
    
    
    #start training
    for ep in tqdm(range(args.epochs)):
        max_total_loss = 0
        min_total_loss = 0
        mean_total_loss = 0
        sum_total_loss = 0
        
        
        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
            query_text = [
                {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                {"role":"assistant","content" : "Sure. Let me see what input friend sees "},
                {"role" : "user", "content" : "The input that friend sees : " + inputs[0]},
                {"role": "assistant","content" : "The Instruction is : "}
            ]
            
            query_encoded = agent_tokenizer.apply_chat_template(
                query_text,
                return_tensors='pt'
            ).view(-1)
            
            response_tensors =ppo_trainer.generate(
                query_encoded.to(device),
                **generation_kwargs,
                return_prompt=False,
                num_return_sequences = args.prompt_per_example
            )
            
            used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            
            #나온 프롬프트 중 너무 길이가 짧은게 많으면 종료
            if sum([len(p) for p in used_prompt]) < args.prompt_per_example * 10:
                break
            
            rewards = []
            losses = []
            with torch.no_grad(): 
                for prompt in used_prompt:
                    template = prompt + "Input : " + inputs[0] + "Output : "
                    prompt_encoded = target_tokenizer(template,return_tensors='pt').to(device)
                    outputs = target_model.generate(**prompt_encoded,max_length=30,do_sample=True)
                    outputs = target_tokenizer.decode(outputs[0],skip_special_tokens=True)
                    f1_score = utils.f1_score(outputs,labels['text'])      
                    loss = f1_score
                    rewards.append(loss - len(prompt) * 0.00001)
                    losses.append(loss)
            np_rewards = np.array(rewards)
            pt_rewards = [torch.tensor(reward_) for reward_ in rewards]
            bs = len(pt_rewards)
            stats = ppo_trainer.step(
                [query_encoded] * bs,
                [response for response in response_tensors],
                pt_rewards,
            )
            max_total_loss += max(losses)
            min_total_loss += min(losses)
            mean_total_loss += np.mean(losses)
            sum_total_loss += sum(losses)
            #print(losses,used_prompt)
        print('Max Total Loss : ', max_total_loss)
        print('Min Total Loss : ', min_total_loss)
        print('Mean Total Loss : ', mean_total_loss)
        print('Sum Total Loss : ', sum_total_loss)
        wandb.log({
            'max_total_loss' : max_total_loss,
            'min_total_loss' : min_total_loss,
            'mean_total_loss' : mean_total_loss,
            'sum_total_loss' : sum_total_loss
        })
        
        #start evaluation
        if ep % 1 == 0:
            test_acc = 0
            test_total = 0
            print('start test')
            # 랜덤하게 5개의 배치 인덱스를 선택합니다.
            random_batches = random.sample(range(len(test_dataloader)), 5)
            batch_count = 0
            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    inputs = batch['text']
                    labels = batch['label']
                    examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
                    query_text = [
                        {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                        {"role":"assistant","content" : "Sure. Let me see what input friend sees "},
                        {"role" : "user", "content" : "The input that friend sees : " + inputs[0]},
                        {"role": "assistant","content" : "The Instruction is : "}
                    ]
                    query_encoded = agent_tokenizer.apply_chat_template(
                        query_text,
                        return_tensors='pt'
                    ).view(-1)
                    response_tensors = ppo_trainer.generate(
                        query_encoded,
                        **generation_kwargs,
                        return_prompt=False,
                        num_return_sequences = 1
                    )
                    used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
                    prompt = used_prompt[0]
                    template = prompt + "\nInput : " + inputs[0] + "Output : "
                    # 선택된 배치 인덱스일 때만 template을 출력합니다.
                    if batch_count in random_batches:
                        print(template)
                    prompt_encoded = target_tokenizer(template,return_tensors='pt').to(device)
                    outputs = target_model.generate(**prompt_encoded,max_length=30,do_sample=True)
                    outputs = target_tokenizer.decode(outputs[0],skip_special_tokens=True)
                    f1_score = utils.f1_score(outputs,labels['text'])                    
                    if torch.argmax(verbalizer_logits).item() == label:
                        test_acc += f1_score
                    test_total += 1
                    batch_count += 1
            print('Test Mean F1 : ', test_acc / test_total)
            wandb.log({
                'Test Mean F1' : test_acc / test_total
            })
        
            
if __name__ == '__main__':
    main()
                
                    
                    
    
    
    