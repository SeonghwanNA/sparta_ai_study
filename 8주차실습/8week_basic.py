import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb

# wandb 초기화
wandb.init(project="LoRa test-8week-basic")

# 모델 및 데이터셋 준비
model_name = "facebook/opt-350m"
dataset_name = "lucasmccabe-lmi/CodeAlpaca-20k"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터셋 로드
dataset = load_dataset(dataset_name)

# 데이터셋 컬럼 확인
print(dataset['train'].column_names)  # 'instruction', 'input', 'output' 확인

# LoRA Config 설정 (rank 값을 8, 128, 256로 변경하며 반복)
lora_r_values = [8, 128, 256]


def tokenize_function(examples):
    # 'instruction'과 'input'이 리스트일 수 있으므로 각각 처리하고 합쳐야 함
    instruction = " ".join(examples['instruction']) if isinstance(examples['instruction'], list) else examples['instruction']
    input_text = " ".join(examples['input']) if isinstance(examples['input'], list) else examples['input']

    # 두 텍스트를 합쳐서 토크나이징 (return_tensors="pt"로 PyTorch 텐서 형식으로 반환)
    return tokenizer(
        instruction + " " + input_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"  # 반환 형식은 텐서로 설정
    )


# 텍스트 포맷팅 함수 정의
def formatting_prompts_func(examples):
    # 예시들이 리스트로 주어짐
    formatted_prompts = []

    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i] if isinstance(examples['instruction'], list) else examples[
            'instruction']
        input_text = examples['input'][i] if isinstance(examples['input'], list) else examples['input']
        output_text = examples['output'][i] if isinstance(examples['output'], list) else examples['output']

        # 텍스트 포맷팅
        formatted_prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
        formatted_prompts.append(formatted_prompt)  # 리스트에 추가

    return formatted_prompts  # 여러 예시를 담은 리스트 반환


#for lora_r in lora_r_values:
print(f"Training with LoRA rank: {128}")
wandb.run.name = f'LoRa test rank {128}'

# LoRA 설정
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=128,
    lora_alpha=16,
    lora_dropout=0.1
)

model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora_config)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],  # 데이터셋에서 train 데이터를 사용
    args=SFTConfig(
        output_dir=f"./lora_rank_{8}",
        max_seq_length=128,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=100,
        logging_steps=10
    ),
    formatting_func=formatting_prompts_func,  # 텍스트 포맷팅 함수 추가
    data_collator=None,  # 필요시 설정
)

# 데이터셋에 토크나이징 적용
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output"])

# 학습 시작
trainer.train()

# 메모리 사용량 기록
print('Max Alloc:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')

# 학습 중 loss와 memory 사용량 로그 기록
def log_metrics():
    loss = trainer.state.best_metric  # 예시로 best_metric을 사용. 다른 변수로 변경 가능
    memory_alloc = torch.cuda.max_memory_allocated(0) / 1024**3  # GB 단위로 변환
    # Log metrics with step
    wandb.log({'loss': loss, 'memory_allocated': memory_alloc}, step=trainer.global_step)


# 매 훈련 epoch 마다 log_metrics 호출
log_metrics()

# wandb 종료
wandb.finish()