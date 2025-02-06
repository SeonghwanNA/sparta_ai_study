import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field


from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# Weights & Biases 초기화 및 프로젝트 이름 설정
wandb.init(project='7week_basic_nabakgood_ai')
wandb.run.name = 'gpt-finetuning'


# 로깅 설정
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)  # 사용할 모델의 경로 또는 이름
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # 데이터 타입 설정
    dataset_name: Optional[str] = field(default=None)  # 사용할 데이터셋 이름
    dataset_config_name: Optional[str] = field(default=None)  # 데이터셋 설정 이름
    block_size: int = field(default=1024)  # 토큰 길이 설정
    num_workers: Optional[int] = field(default=None)  # 데이터 로딩을 위한 워커 수 설정

# 명령행 인자를 받아서 파싱
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

logger.info(f"Training/evaluation parameters: {training_args}")

# 데이터셋 로드
logger.info("Loading dataset...")
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)
logger.info(f"Dataset loaded. Keys: {raw_datasets.keys()}")
logger.info(f"Train dataset size: {len(raw_datasets['train'])}")

# 데이터셋 컬럼 이름 출력 (디버깅용)
logger.info(f"Train dataset columns: {raw_datasets['train'].column_names}")

def filter_long_texts(example):
    # 텍스트 길이가 512 토큰을 넘는 경우 제외
    return len(example['text'].split()) <= 1024

filtered_datasets = raw_datasets.filter(filter_long_texts)

# 데이터 크기를 제한합니다 (1000개 샘플로 제한)
max_samples = 1000  # 제한할 최대 샘플 수

def limit_dataset(dataset, max_samples):
    """데이터셋을 최대 max_samples만큼 제한합니다."""
    if len(dataset) > max_samples:
        return dataset.select(range(max_samples))
    return dataset

# 제한된 데이터셋 생성
raw_datasets["train"] = limit_dataset(raw_datasets["train"], max_samples)
if "validation" in raw_datasets:
    raw_datasets["validation"] = limit_dataset(raw_datasets["validation"], max_samples)

# 모델과 토크나이저 로드
logger.info("Loading model and tokenizer...")
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# 패딩 토큰 설정 (eos_token을 pad_token으로 사용)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Pad token set to: {tokenizer.pad_token}")

# 데이터 토큰화
def tokenize_function(examples):
    text_column = "text"  # 이 부분을 실제 텍스트 컬럼 이름으로 수정
    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=args.block_size)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=args.num_workers,
    remove_columns=raw_datasets["train"].column_names,
    writer_batch_size=1000  # 기본값 1000에서 100으로 낮춤
)

# validation 데이터가 없으면 자동으로 10%를 분리
if "validation" not in tokenized_datasets:
    logger.info("Splitting dataset into train and validation...")
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)

train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["test"]

logger.info(f"Final Train dataset size: {len(train_dataset)}")
logger.info(f"Final Validation dataset size: {len(val_dataset)}")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # labels가 없으면 input_ids를 복사해서 사용
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
        print(inputs.keys())
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Loss 계산
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)



# Training 시작
logger.info("Starting training...")
train_result = trainer.train()
trainer.save_model()

wandb.log({"train/loss": train_result.training_loss})

# Train/Eval loss를 WandB에 로깅
train_loss = train_result.training_loss
eval_result = trainer.evaluate()
logging.debug("Eval keys:", eval_result)  # eval_result에 어떤 키들이 있는지 출력
eval_loss = eval_result.get("eval_loss") or eval_result.get("eval_perplexity") or eval_result.get("eval_accuracy") or eval_result.get("eval_runtime") or eval_result.get("eval_samples_per_second")
if eval_loss is None:
    raise logging.debug("No valid evaluation key found (eval_loss, eval_perplexity, eval_accuracy).")
else:
    logging.debug(f"Eval Metric: {eval_loss}")

if eval_loss is not None:
    wandb.log({"train/loss": train_loss, "eval/loss": eval_loss})

logger.info(f"Train loss: {train_loss}")
logger.info(f"Eval loss: {eval_loss}")