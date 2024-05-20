import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader,TensorDataset
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
from PIL import Image
import os
from peft import LoraConfig
import warnings
import numpy as np
import wandb
import copy
from collections import deque
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import random
import torch
import heapq
import torch.nn.functional as F

class TopAccuracyTextsNoDuplicates:
    def __init__(self, max_size=5):
        self.heap = []
        self.text_map = {}  # 텍스트를 키로, (힙 내 위치, 생성 시점)을 값으로 하는 딕셔너리
        self.max_size = max_size
        self.only_text = []

    def add(self, accuracy, text,ep):
        #print(accuracy,text,ep)
        if text in self.only_text:
            # 이미 존재하는 텍스트의 정확도와 생성 시점 업데이트 (더 높은 정확도로)
            print('already exist')
        else:
            # 새로운 텍스트 추가
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (accuracy, len(text), text, ep))
                self.text_map[text] = (len(self.heap) - 1, ep)
            elif accuracy > self.heap[0][0]:
                # 현재 힙의 최소 정확도보다 높은 경우에만 추가
                removed_text = heapq.heappop(self.heap)[2]
                if removed_text in self.text_map:
                    self.text_map.pop(removed_text)  # 제거된 텍스트를 딕셔너리에서 삭제
                heapq.heappush(self.heap, (accuracy, len(text), text, ep))
                self.text_map[text] = (len(self.heap) - 1, ep)
                self.only_text.append(text)
                return True
        return False

    def get_top_texts(self):
        # 정확도가 높은 순서로 정렬하여 텍스트와 생성 시점을 반환
        return sorted([(accuracy, text, ep) for accuracy, _, text, ep in self.heap], reverse=True)


class TopAccuracyTextsScore:
    def __init__(self, max_size=5):
        self.heap = []
        self.text_map = {}  # 텍스트를 키로, (힙 내 위치, 생성 시점)을 값으로 하는 딕셔너리
        self.max_size = max_size
        self.only_text = []

    def add(self, accuracy, text, ep, score):
        if text in self.only_text:
            print('already exist')
        else:
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (accuracy, len(text), text, ep, score))
                self.text_map[text] = (len(self.heap) - 1, ep)
            elif accuracy > self.heap[0][0]:
                removed_text = heapq.heappop(self.heap)[2]
                if removed_text in self.text_map:
                    self.text_map.pop(removed_text)
                heapq.heappush(self.heap, (accuracy, len(text), text, ep, score))
                self.text_map[text] = (len(self.heap) - 1, ep)
                self.only_text.append(text)
                return True
        return False

    def get_top_texts(self):
        return sorted([(accuracy, text, ep, score) for accuracy, _, text, ep, score in self.heap], reverse=True)

    
        
    



#전체 테스트 셋에 대해서 테스트
def evaluation_full(prompts,imdb,model,tokenizer,device,verbalizer = ['Yes','No'],side='Last'):
    accs=  []
    for prompt in prompts:
        model.eval()
        subset_indices = random.sample(range(len(imdb["test"])), 100)

        # 서브셋 생성
        imdb_subset = Subset(imdb["test"], subset_indices)

        # DataLoader 설정 (서브셋 사용)
        dl = DataLoader(imdb["test"], batch_size=1, shuffle=True)


        tp = 0  # True Positive
        tn = 0  # True Negative
        fp = 0  # False Positive
        fn = 0  # False Negative
        # 배치 처리
        correct = 0
        total = 0

        yes_token_id = tokenizer.encode(verbalizer[0], add_special_tokens=False)[0]
        no_token_id = tokenizer.encode(verbalizer[1], add_special_tokens=False)[0]

        yes_answer_num = 0
        no_answer_num = 0
        yes_predictioon_num = 0
        no_prediction_num = 0

        for batch in tqdm(dl):
            # 텍스트 인코딩
            if side != 'First':
                input_ids = tokenizer(batch['text'][0] + '\n' + prompt, return_tensors='pt',truncation=True).input_ids.to(device)
            else:
                input_ids = tokenizer(prompt + '\n' + batch['text'][0] , return_tensors='pt',truncation=True).input_ids.to(device)
            # 모델 실행
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits

            # 'Yes'와 'No'의 첫 번째 토큰에 대한 로짓 비교
            yes_logits = logits[0, -1, yes_token_id]
            no_logits = logits[0, -1, no_token_id]

            prediction = 'Yes' if yes_logits > no_logits else 'No'
            correct_label = 'Yes' if batch['label'][0] == 1 else 'No'
            if correct_label == 'Yes':
                yes_answer_num += 1
            else:
                no_answer_num += 1
            if prediction == 'Yes':
                yes_predictioon_num += 1
            else:
                no_prediction_num += 1
            # 정답 레이블과 비교
            if prediction == 'Yes' and correct_label == 'Yes':
                tp += 1
            elif prediction == 'No' and correct_label == 'No':
                tn += 1
            elif prediction == 'Yes' and correct_label == 'No':
                fp += 1
            elif prediction == 'No' and correct_label == 'Yes':
                fn += 1

        # 성능 지표 계산
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = sensitivity  # 재현율은 민감도와 동일
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accs.append(accuracy)
    return accs


def induction_soft(
    prompts,
    inputs,
    targets,
    model,
    tokenizer,
    device,
    ):
    rewards = []
    for prompt in prompts:
        reward = 0
        for i in range(len(targets)):
            question = prompt +'\nInput : '+ inputs[i]
            answer = targets[i]
            encoded_input = tokenizer(question,answer, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded_input,labels = encoded_input['input_ids'])
                logits = outputs.logits
                labels = encoded_input['input_ids']
                log_probs = -torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
                log_probs = log_probs.view(labels.size())

                # 정답의 log probability 합산
                answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
                answer_log_probs = log_probs[0, -len(answer_tokens) - 1:-1].sum().item()
                reward += answer_log_probs
        rewards.append(reward)
    return rewards, [0.0 for i in range(len(prompts))]
            
            

def language_feedback(
    queue,
    ppo_trainer,
    ppo_tokenizer,
    args,
    example,
    generation_kwargs,
    device,
):
    li = queue.get_top_texts()
    used_prompt = [i[1] for i in li]
    accuracys = [i[0] for i in li]
    feedback = ''
    for prompt, accuracy in zip(used_prompt, accuracys):
        feedback += 'Instruction: {prompt}\nAccuracy: {accuracy:.2f}\n'.format(prompt=prompt, accuracy=accuracy)
    feedback_query = [
        {"role": "user", "content": """This is output instruction and it's accuracy. Please describe the problems of the instruction generated to solve this problem. \nTasks : """ + args.meta_question + '\n' + example+ feedback},
        {"role": "assistant", "content" : " Problems : "}
    ]
    feedback_output = ppo_trainer.generate(ppo_tokenizer.apply_chat_template(feedback_query, return_tensors='pt').view(-1).to(device),**generation_kwargs,return_prompt=False, num_return_sequences=1)
    feedback_output = ppo_tokenizer.decode(feedback_output.squeeze(),skip_special_tokens=True)
    update_query = [
        {"role": "user", "content": 'Please write instructions to fix this problem.\n' + feedback_output},
        {"role": "assistant", "content" : "Instruction : "}
    ]
    update_output = ppo_trainer.generate(ppo_tokenizer.apply_chat_template(update_query, return_tensors='pt').view(-1).to(device),**generation_kwargs,return_prompt=False, num_return_sequences=1)
    update_output = ppo_tokenizer.decode(update_output.squeeze(),skip_special_tokens=True)
    updated_query_text = [
        {"role": "user", "content": args.meta_question + '\n' + example + '\n Here is some tips for writing instruction :' + update_output},
        {"role": "assistant", "content" : "Instruction : "}
    ]
    return updated_query_text

def evaluation_roberta(prompts,
                       test_dataset,
                       model,
                       tokenizer,
                       device,
                       verbalizer,
                       debug=False,
                       side='First',
                       batch_size= 64
                        ):
    def _format_prompts(prompts,inputs,side):
        if side=='Last':
            template = "{sentence_1} {prompt} <mask> ."
        else:
            template = "{prompt} Input : {sentence_1}  Output : <mask> ."
        return [template.format(sentence_1=s_1, prompt=prompt)
            for s_1, prompt in zip(inputs, prompts)]
    def _get_mask_token_index(input_ids, tokenizer):
        mask_token_index = []
        for ids in input_ids:
            # 마스크 토큰 위치 찾기
            mask_positions = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                # 마스크 토큰이 없는 경우, 마지막 인덱스 사용
                mask_token_index.append(len(ids) - 2)
            else:
                # 마스크 토큰이 있는 경우, 첫 번째 마스크 토큰 위치 사용
                mask_token_index.append(mask_positions[0].item())

        return torch.tensor(mask_token_index)

    def _get_logits(texts,tokenizer,model,device):
        batch_size = len(texts)
        
        encoded_inputs = tokenizer(texts, 
                                padding='longest', 
                                truncation=True, return_tensors="pt",add_special_tokens=True)
        token_logits = model(**encoded_inputs.to(device)).logits
        mask_token_indices= \
            _get_mask_token_index(encoded_inputs['input_ids'],tokenizer)
        out_logits = token_logits[range(batch_size), mask_token_indices, :]
        return out_logits
    
    accuracys = []
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
    for prompt in prompts:
        num_of_examples = dataloader.dataset.__len__()
        correct_sum = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if 'text' in batch.keys() :
                    inputs = batch['text']
                else :
                    inputs = batch['sentence']
                targets = batch['label']
                batch_size = targets.size(0)
                current_prompts = [prompt for _ in range(batch_size)]
                formatted_templates = _format_prompts(current_prompts,inputs,side=side)
                all_logits=  _get_logits(formatted_templates,tokenizer,model,device)
                class_probs = torch.softmax(all_logits[:,verbalizer_ids],-1)
                predicted_labels = torch.argmax(class_probs,-1)
                label_agreement = torch.where(
                    targets.to(device) == predicted_labels,1,0
                )
                correct_sum += label_agreement.sum()
        accuracy = correct_sum / num_of_examples
        accuracys.append(accuracy.cpu())
    return accuracys
    
def evaluation_roberta_soft(prompts,
                       inputs
                       ,targets,
                       model,
                       tokenizer,
                       device,
                       verbalizer,
                       debug=False,
                       side='First',
                       Fail_coefficient=1,
                       Success_coefficient=1,
                       return_reward = False, 
                    ):
    def _format_prompts(prompts,inputs,side):
        if side=='Last':
            template = "{sentence_1} {prompt} <mask> ."
        else:
            template = "{prompt} Input : {sentence_1}  Output : <mask> ."
        return [template.format(sentence_1=s_1, prompt=prompt)
            for s_1, prompt in zip(inputs, prompts)]
    def _get_mask_token_index(input_ids, tokenizer):
        mask_token_index = []
        for ids in input_ids:
            # 마스크 토큰 위치 찾기
            mask_positions = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                # 마스크 토큰이 없는 경우, 마지막 인덱스 사용
                mask_token_index.append(len(ids) - 2)
            else:
                # 마스크 토큰이 있는 경우, 첫 번째 마스크 토큰 위치 사용
                mask_token_index.append(mask_positions[0].item())

        return torch.tensor(mask_token_index)

    def _get_logits(texts,tokenizer,model,device):
        batch_size = len(texts)
        
        encoded_inputs = tokenizer(texts, 
                                padding='longest', 
                                truncation=True, return_tensors="pt",add_special_tokens=True)
        token_logits = model(**encoded_inputs.to(device)).logits
        mask_token_indices= \
            _get_mask_token_index(encoded_inputs['input_ids'],tokenizer)
        out_logits = token_logits[range(batch_size), mask_token_indices, :]
        return out_logits
    accuracies = []
    model.eval()
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)

    batch_size = targets.size(0)
    rewards = []
    with torch.no_grad():
        for prompt in prompts:
            current_prompts = [prompt for _ in range(batch_size)]
            formatted_templates = _format_prompts(current_prompts, inputs,side=side)
            all_logits = _get_logits(formatted_templates, tokenizer, model, device)
            verbalizer_logits = all_logits[:, verbalizer_ids]
            
            # 추출된 로그 확률에 소프트맥스를 적용합니다.
            log_probs = F.softmax(verbalizer_logits, dim=1)
            
            #get accuracy
            preds = torch.argmax(log_probs, dim=1).cpu()
            correct_predictions = torch.sum(preds == targets)
            accuracy = correct_predictions.item() / batch_size
            accuracies.append(accuracy)
            
            #get reward
            reward = get_reward(all_logits, targets,Fail_coefficient=Fail_coefficient, Success_coefficient=Success_coefficient)
            mean_reward = reward.mean().cpu()
            rewards.append(mean_reward)
    z_scaled_reward = get_z_scaled_reward(rewards)
    if return_reward:
        return z_scaled_reward, accuracies, rewards
    else:
        return z_scaled_reward, accuracies
    

def create_balanced_subset(dataset, subset_size, label_key='label'):
    # Group dataset by label
    by_label = {}
    for item in dataset:
        label = item[label_key]
        if label in by_label:
            by_label[label].append(item)
        else:
            by_label[label] = [item]
    
    # Calculate the number of samples per class
    per_class = subset_size // len(by_label)
    
    # Create the subset
    subset = []
    for label, items in by_label.items():
        subset.extend(random.sample(items, min(per_class, len(items))))
    
    # In case subset_size is not perfectly divisible by the number of labels,
    # add random items from any class until the subset reaches the desired size
    while len(subset) < subset_size:
        label = random.choice(list(by_label.keys()))
        subset.append(random.choice(by_label[label]))
    
    random.shuffle(subset) # Shuffle the final subset to mix labels
    return subset

# Create a balanced random subset
def create_balanced_subset_and_validation(dataset, subset_size, label_key='label'):
    # Group dataset by label
    by_label = {}
    for item in dataset:
        label = item[label_key]
        by_label.setdefault(label, []).append(item)

    # Calculate the number of samples per class for subset and validation
    per_class = subset_size // len(by_label)

    # Create the subset and validation set
    subset = []
    validation_set = []
    for label, items in by_label.items():
        # Ensure the items list is shuffled
        random.shuffle(items)

        # Split items into two halves for subset and validation
        subset_items = items[:len(items)//2]
        validation_items = items[len(items)//2:]

        # Add samples to the subset and validation set
        subset.extend(subset_items[:min(per_class, len(subset_items))])
        validation_set.extend(validation_items[:min(per_class, len(validation_items))])

    # Handle the case where subset_size is not perfectly divisible
    while len(subset) < subset_size:
        label = random.choice(list(by_label.keys()))
        subset.append(random.choice(by_label[label][:len(by_label[label])//2]))

    while len(validation_set) < subset_size:
        label = random.choice(list(by_label.keys()))
        validation_set.append(random.choice(by_label[label][len(by_label[label])//2:]))

    # Shuffle the final subsets
    random.shuffle(subset)
    random.shuffle(validation_set)

    return subset, validation_set
    

def evaluation(prompts,
               dataset,
               model,
               tokenizer,
               device,
               verbalizer=['Yes', 'No', 'Maybe'],
               dataset_size=100,
               debug=False,
               side='First',
               MaskLM=False,
               batch_size =16,
               return_wrong = False,
               ):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    accuracys = []
    for prompt in tqdm(prompts):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                if 'text' in batch.keys():
                    inputs = batch['text']
                else:
                    inputs = batch['sentence']
                targets = batch['label']
                _,acc = evaluation_soft(
                    [prompt],
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer,
                    side=side,
                )
                batch_size = len(targets)
                correct += acc[0] * batch_size
                total += batch_size
        accuracy = correct / total
        accuracys.append(torch.Tensor([accuracy]))
    return accuracys
    
    
    
def evaluation_cot_full(prompts,
               dataset,
               model,
               tokenizer,
               device,
               verbalizer=['Yes', 'No', 'Maybe'],
               dataset_size=100,
               debug=False,
               side='First',
               MaskLM=False,
               batch_size =16,
               return_wrong = False,
               ):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    model.eval()
    accuracys = []
    for prompt in tqdm(prompts):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if 'text' in batch.keys():
                    inputs = batch['text']
                else:
                    inputs = batch['sentence']
                targets = batch['label']
                _,acc = evaluation_cot(
                    [prompt],
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer,
                    side=side,
                )
                batch_size = len(targets)
                correct += acc[0] * batch_size
                total += batch_size
        accuracy = correct / total
        accuracys.append(torch.Tensor([accuracy]))
    return accuracys
    
    
def evaluation_apo(prompts,
               dataset,
               model,
               tokenizer,
               device,
               verbalizer=['Yes', 'No', 'Maybe'],
               dataset_size=100,
               debug=False,
               side='First',
               MaskLM=False,
               batch_size =16,
               return_wrong = False,
               ):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    accuracys = []
    incorrects = []
    for prompt in tqdm(prompts):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                if 'text' in batch.keys():
                    inputs = batch['text']
                else:
                    inputs = batch['sentence']
                targets = batch['label']
                _,acc = evaluation_soft(
                    [prompt],
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer,
                    side=side,
                )
                
                incorrect = evaluation_soft_apo(
                    [prompt],
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer,
                    side=side,
                    debug=True,
                )
                batch_size = len(targets)
                correct += acc[0] * batch_size
                total += batch_size
                incorrects.append(incorrect)
        accuracy = correct / total
        accuracys.append(torch.Tensor([accuracy]))
    if len(incorrects) > 3 :
        incorrects = incorrects[:3]
    return accuracys,incorrects
    
def evaluation_soft_apo(prompts, inputs, targets, model, tokenizer, device, verbalizer, debug=False, Fail_coefficient=1, Success_coefficient=1, return_reward=False, side='First'):
    def _format_prompts(prompts, inputs, side):
        if side == 'First':
            template = "{prompt} Input : {sentence_1} Output:"
        else:
            template = "{sentence_1} {prompt}"
        return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]

    def _get_next_token_index(input_ids):
        return input_ids.shape[1] - 1

    def _get_logits(texts, tokenizer, model, device):
        encoded_inputs = tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
        token_logits = model(**encoded_inputs.to(device)).logits
        next_token_indices = _get_next_token_index(encoded_inputs['input_ids'])
        out_logits = token_logits[range(len(texts)), next_token_indices, :]
        return out_logits

    incorrect_pairs = []
    model.eval()
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
    for prompt in prompts:
        formatted_templates = _format_prompts([prompt] * len(inputs), inputs, side=side)
        all_logits = _get_logits(formatted_templates, tokenizer, model, device)
        
        verbalizer_logits = all_logits[:, verbalizer_ids]
        log_probs = F.softmax(verbalizer_logits, dim=1)
        preds = torch.argmax(log_probs, dim=1).cpu()

        # 잘못된 예측을 찾습니다.
        incorrect_indices = (preds != targets).nonzero(as_tuple=True)[0]
        for idx in incorrect_indices:
            incorrect_pairs.append((inputs[idx], preds[idx].item(), targets[idx].item()))

    if debug:
        return incorrect_pairs
    else:
        # 기존 함수의 반환 값 유지를 위해 조정이 필요한 경우 이 부분을 수정합니다.
        return None




def evaluation_cot(prompts,
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer, 
                    debug=False,
                    Fail_coefficient=1,
                    Success_coefficient=1,
                    return_reward= False,
                    side = 'First',
                    ):
    
    
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens":128,
    "min_length": -1,
    }
    
    def _format_prompts(prompts, inputs,side):
        if side == 'First':
            template = "{prompt} Input : {sentence_1} Output:"
        else:
            template = "{sentence_1} {prompt}"
        return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]
    
    def _get_next_token_index(input_ids):
        # 입력의 마지막 토큰 다음 위치 반환
        return input_ids.shape[1] - 1
    def _chat_format_prompts(prompts,inputs):
        #print(inputs)
        query_text =[
            {'role':"user",'content':prompt+". Let's Think step by step." + '\nInput : ' + inputs[0] },
            {'role':"assistant",'content':'Step-by-Step Answer: '}
            
        ]
        return query_text
    def _get_logits(texts, tokenizer, model, device):
        batch_size = len(texts)
        encoded_inputs = tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
        token_logits = model(**encoded_inputs.to(device)).logits
        next_token_indices = _get_next_token_index(encoded_inputs['input_ids'])
        #print(token_logits.size(),next_token_indices)
        out_logits = token_logits[range(batch_size), next_token_indices, :]
        #print(out_logits.size())
        return out_logits
    
    def _get_answer(texts,tokenizer,model,device):
        encoded_texts = tokenizer.apply_chat_template(texts,return_tensors='pt').to(device)
        generated_output = model.generate(encoded_texts,**generation_kwargs)
        generated_output = tokenizer.decode(generated_output.squeeze(),skip_special_tokens=True)
        #print(generated_output)
        return generated_output
    
    accuracies = []
    rewards = []
    model.eval()
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
    batch_size = targets.size(0)
    correct_all = 0
    for prompt in prompts:
        #Get logits
        
        
        current_prompts = [prompt for _ in range(batch_size)]
        chat_template_input = _chat_format_prompts(current_prompts,inputs)
        middel_answer = _get_answer(chat_template_input,tokenizer,model,device)
        print(current_prompts)
        middel_answer += '\n' + 'Therefore, the answer is :'
        print(middel_answer)
        all_logits = _get_logits([middel_answer], tokenizer, model, device)
        #print(all_logits.size())
        #print(verbalizer_ids)

        #Get verbalizer logits
        verbalizer_logits = all_logits[:, verbalizer_ids]
        log_probs = F.softmax(verbalizer_logits, dim=1)
        #print(log_probs)
        #Get accuracy
        preds = torch.argmax(log_probs, dim=1).cpu()
        correct_predictions = torch.sum(preds == targets)
        accuracy = correct_predictions.item() / batch_size
        correct_all += accuracy
        print(correct_all)
        accuracies.append(accuracy)
        
        #Get reward
        reward = get_reward(all_logits, targets, Fail_coefficient=Fail_coefficient, Success_coefficient=Success_coefficient)
        mean_reward = reward.mean().cpu()
        rewards.append(mean_reward)

    z_scaled_reward = get_z_scaled_reward(rewards)

    if return_reward:
        return z_scaled_reward, accuracies, rewards
    else:
        return z_scaled_reward, accuracies





def evaluation_soft(prompts,
                    inputs,
                    targets,
                    model,
                    tokenizer,
                    device,
                    verbalizer, 
                    debug=False,
                    Fail_coefficient=1,
                    Success_coefficient=1,
                    return_reward= False,
                    side = 'First',
                    ):
    def _format_prompts(prompts, inputs,side):
        if side == 'First':
            template = "{prompt} Input : {sentence_1} Output:"
        else:
            template = "{sentence_1} {prompt}"
        return [template.format(sentence_1=s_1, prompt=prompt) for s_1, prompt in zip(inputs, prompts)]
    
    def _get_next_token_index(input_ids):
        # 입력의 마지막 토큰 다음 위치 반환
        return input_ids.shape[1] - 1

    def _get_logits(texts, tokenizer, model, device):
        batch_size = len(texts)
        encoded_inputs = tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True)
        token_logits = model(**encoded_inputs.to(device)).logits
        next_token_indices = _get_next_token_index(encoded_inputs['input_ids'])
        out_logits = token_logits[range(batch_size), next_token_indices, :]
        return out_logits

    accuracies = []
    rewards = []
    model.eval()
    verbalizer_ids = tokenizer.convert_tokens_to_ids(verbalizer)
    batch_size = targets.size(0)
    for prompt in prompts:
        #Get logits
        current_prompts = [prompt for _ in range(batch_size)]
        formatted_templates = _format_prompts(current_prompts, inputs,side=side)
        all_logits = _get_logits(formatted_templates, tokenizer, model, device)
        
        #Get verbalizer logits
        verbalizer_logits = all_logits[:, verbalizer_ids]
        log_probs = F.softmax(verbalizer_logits, dim=1)
        #print(log_probs)
        #Get accuracy
        preds = torch.argmax(log_probs, dim=1).cpu()
        correct_predictions = torch.sum(preds == targets)
        accuracy = correct_predictions.item() / batch_size
        accuracies.append(accuracy)
        
        #Get reward
        reward = get_reward(all_logits, targets, Fail_coefficient=Fail_coefficient, Success_coefficient=Success_coefficient)
        mean_reward = reward.mean().cpu()
        rewards.append(mean_reward)

    z_scaled_reward = get_z_scaled_reward(rewards)

    if return_reward:
        return z_scaled_reward, accuracies, rewards
    else:
        return z_scaled_reward, accuracies
    
    
    
def get_z_scaled_reward(rewards):
    all_rewards = torch.stack(rewards,dim=0)
    all_mean = torch.mean(all_rewards)
    all_std = torch.std(all_rewards)
    z_scaled_reward = (all_rewards - all_mean) / all_std
    z_scaled_reward = [reward for reward in z_scaled_reward]
    return z_scaled_reward

#이전 대화 항목을 제거하고 프롬프트로 사용
def extract_text_after_colon(text, key = 'AI:'):
    # ':' 문자의 위치를 찾습니다.
    colon_index = text.find(key)

    # ':' 문자가 없으면, 원본 텍스트를 반환합니다.
    if colon_index == -1:
        return text

    # ':' 다음의 문자부터 문자열 끝까지 반환합니다.
    return text[colon_index + len(key):]


import random
from dataset_utils import dataset_dicts



def get_reward(
    logits,
    labels,
    Fail_coefficient=1,
    Success_coefficient=1,
):
    #TODO
    # Inputs :
    #     logits : 모델의 출력 logits
    #     targets : 정답 레이블
    # Outputs :
    #     reward : 정답 로짓 - 최대 로짓의 값 * 성공/실패 계수
    with torch.no_grad():
        labels = labels.to('cpu')
        logits = logits.to('cpu')
        correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        # 정답 레이블의 로짓 값을 0으로 만들어 최대값 계산에 영향을 주지 않도록 합니다.
        mask = torch.ones_like(logits)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        masked_logits = logits * mask

        # 마스킹된 로짓 중 최대값을 찾습니다.
        max_other_logits = masked_logits.max(dim=1)[0]
        differences = correct_logits - max_other_logits
        
    reward = torch.where(differences > 0, differences * Success_coefficient, differences * Fail_coefficient)
    return reward


def remove_text_after_key(text, key='AI:'):
    # 키워드의 위치를 찾습니다.
    key_index = text.find(key)

    # 키워드가 없으면, 원본 텍스트를 반환합니다.
    if key_index == -1:
        return text
    while key_index == 0:
        # 다음 키워드 위치 찾기 (현재 위치 + 키 길이 부터 시작)
        key_index = text.find(key, key_index + len(key))
        
        # 다음 키워드가 없으면, 원본 텍스트를 반환합니다.
        if key_index == -1:
            return text

    # 키워드 이전의 문자부터 문자열 시작까지 반환합니다.
    return text[:key_index]
def got_example(dataset,dataset_dict,shot=5,label_key='label'):
    examples =''
    for i in range(shot):
        idx = random.randint(0,len(dataset)-1)
        example = dataset[idx]
        if example[label_key] == -1:
            continue
        if 'text' in example.keys():
            a = 'Input : ' + example['text']+ '\nOutput : '+ dataset_dict[example[label_key]] + '\n'
            examples += a 
        else:
            a= 'Input : ' + example['sentence']+ '\nOutput : '+ dataset_dict[example[label_key]] + '\n'
            #examples.append(a)
            examples += a
            
    return examples

def got_example_input(dataset,dataset_dict,shot=5,in_sentence=False):
    examples =[]
    for i in range(shot):
        idx = random.randint(0,len(dataset)-1)
        example = dataset[idx]
        if example['label'] == -1:
            continue
        if 'text' in example.keys():
            a = 'Input : ' + example['text']+ '\nOutput : '+ dataset_dict[example['label']]
            examples.append(a)
        else:
            a= 'Input :' + example['sentence']+ '\nOutput : '+ dataset_dict[example['label']]
            examples.append(a)
    if in_sentence:
        return ' '.join(examples)
    else:
        return examples


def evaluate_openai(
    prompts : list,
    datasets : TensorDataset,
    target_model = 'davinci-002',
):
    #TODO
    # Inputs :
    #     prompts : 텍스트로 된 prompt들의 리스트 (예: ['Is this reveiw positive?', 'Is this reveiw negative?'])
    #     datasets : TensorDataset 형태로 된 데이터셋 (예: TensorDataset(inputs, labels))
    #     target_model : OpenAI API에 사용할 모델 이름 (예: 'davinci-002')
    # Outputs :
    #     accuracy : 각 prompt에 대한 정확도 (예: [0.95, 0.92])
    return 0.00
    
    
def reward_openai(
    prompts :list,
    inputs,
    labels,
    target_model = 'davinci-002',
):
    #TODO
    # Inputs :
    #     prompts : 텍스트로 된 prompt들의 리스트 (예: ['Is this reveiw positive?', 'Is this reveiw negative?'])
    #     inputs : 이번 배치의 입력 텍스트 (예: ['This movie is great!', 'This movie is bad!'])
    #     labels : 정답 레이블들의 리스트 (예: [1, 0])
    #     target_model : OpenAI API에 사용할 모델 이름 (예: 'davinci-002')
    # Outputs :
    #     rewards : 각 입력에 대한 보상. 아마 log probability가 될 것 같음 (예: [0.95, 0.92])
    #     accuracy : 각 prompt에 대한 정확도 (예: [0.95, 0.92])
    return [torch.Tensor(0.00) for i in range(len(prompts))], [torch.Tensor(0.00) for i in range(len(prompts))]