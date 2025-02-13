import logging
import wandb
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, LlamaTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

# Wandb callback
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            wandb.log({"train_loss": logs["loss"]})

# logging 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# wandb 초기화
wandb.init(project="instruction-tuning-korea-travel-lora")
wandb.run.name = 'korea-travel-gpt-finetuning-lora'

# 1. corpus.json 파일 불러오기
try:
    with open("corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
        logging.debug("파일 로드 성공")

        for example in corpus:
            logging.debug(f"instruction 길이: {len(example['instruction'])}"
                          f", response 길이: {len(example['response'])}")
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
model_name = "openlm-research/open_llama_3b"  # OpenLLM 모델 경로
tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=False)  # LlamaTokenizer 사용

# padding 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. 데이터 전처리 함수
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

    max_token_id = max(max(ids) for ids in tokenized["input_ids"])
    logging.debug(f"Max token id in input_ids: {max_token_id}")
    return tokenized

# 6. 데이터셋에 전처리 적용
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
model = None  # 모델 변수 초기화

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # token="YOUR_HUGGING_FACE_TOKEN"  # 필요한 경우 사용, huggingface-cli login 권장
    )
    print("모델 구조 출력")
    print(model)

    # PEFT 설정 (LoRA)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # PEFT 모델 생성
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

except Exception as e:
    logging.error(f"모델 로드 또는 PEFT 적용 실패: {e}")


# 8. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    logging_steps=500,
    report_to="wandb",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=False,
    no_cuda=True,
)

# 9. Trainer 정의 (model이 None이 아닌 경우에만)
if model is not None:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[WandbCallback]
    )

    # 10. 모델 학습
    trainer.train()

    # Train and evaluate
    trainer.evaluate()

    # 손실 값 로깅
    if len(trainer.state.log_history) > 0 and "loss" in trainer.state.log_history[-1]:
        wandb.log({"train_loss": trainer.state.log_history[-1]["loss"]})
    else:
        logging.warning("손실 값이 없습니다.")

    # 모델 저장
    model.save_pretrained("./korea-travel-llama")
else:
    logging.error("모델 로드에 실패하여 Trainer를 초기화하지 않았습니다.")