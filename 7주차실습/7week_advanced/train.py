import logging

import wandb
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# logging 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# wandb 초기화
wandb.init(project="instruction-tuning-korea-travel")  # wandb 프로젝트 및 엔터티 설정
wandb.run.name = 'korea-travel-gpt-finetuning'

# 1. corpus.json 파일 불러오기
try:
    with open("corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
        logging.debug("파일 로드 성공")

        for example in corpus:
            logging.debug(f"instruction 길이: {len(example['instruction'])}, response 길이: {len(example['response'])}")
            logging.debug(f"instruction 내용: {example['instruction']}")

except Exception as e:
    logging.error(f"파일 로드 실패: {e}")

# 2. 데이터셋 생성
dataset = load_dataset("json", data_files={"train": "corpus.json"})

# 3. train/validation 데이터 분할 (8:2)
train_valid_dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_valid_dataset["train"]
valid_dataset = train_valid_dataset["test"]

# 4. 토크나이저 불러오기
model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# padding 토큰 설정 (필요시 추가)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 5. 데이터 전처리 함수 (labels 추가)
def preprocess_function(examples):
    texts = [f"Instruction: {instruction} Response: {response}" for instruction, response in
             zip(examples["instruction"], examples["response"])]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()

    # input_ids 범위 검사 및 수정
    tokenized["input_ids"] = [
        [token if token < tokenizer.vocab_size else tokenizer.pad_token_id for token in ids]
        for ids in tokenized["input_ids"]
    ]

    # Debugging: Check max token id for a sample
    max_token_id = max(max(ids) for ids in tokenized["input_ids"])
    logging.debug(f"Max token id in input_ids: {max_token_id}")

    return tokenized


# 6. 데이터셋에 전처리 적용 (labels 유지)
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "response"],
)
valid_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "response"],
)

# 7. 모델 불러오기
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    logging.error(f"모델 로드 실패: {e}")

# 8. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir='./logs',
    logging_steps=500,
    report_to="wandb",
    per_device_train_batch_size=8,  # 배치 크기 조정 가능
    per_device_eval_batch_size=8,  # 배치 크기 조정 가능
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",  # Deprecated된 `evaluation_strategy` 대신 `eval_strategy` 사용
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=False,  # CPU 학습 시에는 fp16 사용하지 않음
    no_cuda=True,  # CPU 사용
)

# 9. Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# 모델 저장
model.save_pretrained("./korea-travel-gpt")  # 모델 저장 경로 명시

# 10. 모델 학습
trainer.train()

# Train and evaluate
trainer.evaluate()

# 손실 값 로깅 예시
wandb.log({"train_loss": trainer.state.log_history[-1]['loss']})