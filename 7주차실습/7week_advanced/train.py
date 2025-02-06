from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
import wandb
import re

# wandb 초기화
wandb.init(project="instruction-tuning-korea-travel")  # wandb 프로젝트 및 엔터티 설정
wandb.run.name = 'korea-travel-gpt-finetuning'

# 데이터셋 불러오기
raw_datasets = load_dataset('json', data_files='corpus.json')

# 'test' 키가 없다면 train 데이터를 split하여 test 데이터를 생성
if 'test' not in raw_datasets:
    raw_datasets = raw_datasets['train'].train_test_split(test_size=0.2)
    train_dataset = raw_datasets['train']
    val_dataset = raw_datasets['test']
else:
    train_dataset = raw_datasets['train']
    val_dataset = raw_datasets['test']

# 데이터셋에 결측값이 있는지 확인하는 함수
def check_missing_values(dataset):
    for idx, example in enumerate(dataset):
        if example.get('instruction') is None or example.get('response') is None:
            print(f"Missing value found at index {idx}: {example}")

# 데이터셋 확인 (결측값을 찾기)
check_missing_values(train_dataset)
check_missing_values(val_dataset)

# 마크다운 포맷을 텍스트로 변환하는 함수
def clean_markdown(text):
    """
    마크다운 구문 제거 (헤더, 굵게 표시된 텍스트 등)
    """
    text = re.sub(r'##\s*', '', text)  # ## 헤더 제거
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **굵게** 처리된 부분 제거
    text = re.sub(r'\n+', ' ', text)  # 여러 줄 바꿈을 공백으로 처리
    return text

# 데이터의 'instruction'과 'response' 필드가 제대로 존재하는지 체크
def filter_invalid_examples(example):
    """
    'instruction'이나 'response'가 None이거나 빈 문자열인 경우 필터링
    """
    return example.get('instruction') and example.get('response')

# 'instruction'과 'response'가 유효한 데이터만 남기기
train_dataset = train_dataset.filter(filter_invalid_examples)
val_dataset = val_dataset.filter(filter_invalid_examples)

# 'instruction'과 'response' 텍스트에서 마크다운 포맷을 제거
train_dataset = train_dataset.map(lambda x: {'instruction': clean_markdown(x['instruction']), 'response': clean_markdown(x['response'])})
val_dataset = val_dataset.map(lambda x: {'instruction': clean_markdown(x['instruction']), 'response': clean_markdown(x['response'])})

# 토크나이저 불러오기 (GPT-2 사용)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# padding 토큰 설정 (필요시 추가)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# 토큰화 함수 정의
def tokenize_function(examples):
    """
    Instruction과 response를 모두 토큰화하고,
    Instruction과 response 토큰을 합쳐 새로운 input을 생성하는 함수

    Args:
        examples: instruction과 response를 포함하는 데이터셋 딕셔너리

    Returns:
        inputs: input_ids와 attention_mask를 포함하는 딕셔너리
    """
    # Instruction과 response를 모두 토큰화
    instructions = tokenizer(
        examples['instruction'],
        truncation=True,
        padding="max_length",
        max_length=1024,  # 적절한 max_length 값으로 수정
        return_tensors='pt'
    )
    responses = tokenizer(
        examples['response'],
        truncation=True,
        padding="max_length",
        max_length=1024,  # 적절한 max_length 값으로 수정
        return_tensors='pt'
    )

    # Instruction과 response 토큰을 합쳐 새로운 input 생성
    inputs = {
        "input_ids": [],
        "attention_mask": []
    }

    for instruction_ids, response_ids in zip(instructions['input_ids'], responses['input_ids']):
        # None 값 처리: instruction_ids나 response_ids가 None이면 건너뜀
        if instruction_ids is None or response_ids is None:
            continue

        input_ids = [tokenizer.bos_token_id] + instruction_ids.tolist() + \
                    [tokenizer.sep_token_id] + response_ids.tolist() + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)  # attention_mask는 1로 설정 (패딩이 아니면 1)

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)

    # 비어 있는 데이터가 없도록 필터링
    if len(inputs["input_ids"]) == 0:
        return None

    return inputs


# 토큰화 적용
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# None을 포함하는 항목을 필터링
tokenized_train_dataset = tokenized_train_dataset.filter(lambda x: x['input_ids'] is not None)
tokenized_val_dataset = tokenized_val_dataset.filter(lambda x: x['input_ids'] is not None)

# 모델 불러오기 (GPT-2 사용)
model = AutoModelForCausalLM.from_pretrained('gpt2')

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # GPU 메모리에 따라 조절
    per_device_eval_batch_size=4,  # GPU 메모리에 따라 조절
    learning_rate=2e-5,  # 적절한 learning_rate 값으로 수정
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="wandb",  # wandb 로깅 설정
    eval_strategy="epoch",  # 각 에포크마다 평가
    save_strategy="epoch",  # 각 에포크마다 모델 저장
    run_name="korea-travel-gpt-finetuning",  # run_name을 지정
    load_best_model_at_end=True,  # 최적의 모델을 로드
    metric_for_best_model="loss",  # loss를 기준으로 최적의 모델 선택
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # early stopping 적용
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained("./korea-travel-gpt")  # 모델 저장 경로 명시

# wandb 로그 링크 출력
print("wandb 로그 링크:", wandb.run.get_url())
