LoRA Rank에 따른 학습 성능 비교 (8주차 기본 과제)
본 문서는 LoRA (Low-Rank Adaptation)에서 rank 값을 변경하면서 학습을 진행했을 때 발생하는 성능, 학습 속도, 메모리 사용량의 변화를 분석하고, LoRA의 장단점을 논의합니다.
실험 설정
* 모델: facebook/opt-350m
* 데이터셋: lucasmccabe-lmi/CodeAlpaca-20k
* LoRA rank (lora_r): 8, 128, 256
* Deepspeed: 사용 안 함
* SFTTrainer 설정:

wandb log : https://wandb.ai/ainabakgood-personal/LoRa%20test-8week-basic?nw=nwuserainabakgood&panelDisplayName=train%2Floss&panelSectionName=train


실험 결과
LoRA rank : 8
{'train_runtime': 2979.7306, 'train_samples_per_second': 20.158, 'train_steps_per_second': 2.52, 'train_loss': 1.7972377590911495, 'epoch': 3.0}
Max Alloc: 3.1 GB


LoRA rank : 128
{'train_runtime': 3150.6706, 'train_samples_per_second': 19.065, 'train_steps_per_second': 2.383, 'train_loss': 1.7855897097727924, 'epoch': 3.0}
Max Alloc: 3.2 GB

LoRA rank : 256
{'train_runtime': 3377.9271, 'train_samples_per_second': 17.782, 'train_steps_per_second': 2.223, 'train_loss': 1.7853196822687287, 'epoch': 3.0}
Max Alloc: 3.4 GB


결과 분석
Loss: Rank가 증가함에 따라 train_loss가 소폭 감소하는 경향을 보입니다. Rank 128과 256에서는 loss 값에 큰 차이가 없습니다. 이는 더 높은 rank가 모델의 표현력을 증가시켜 데이터에 더 잘 적합(fit)될 수 있음을 시사하지만, 일정 수준 이상에서는 그 효과가 미미해짐을 의미합니다.

학습 속도: Rank가 증가할수록 학습 속도 (train_samples_per_second, train_steps_per_second)가 감소합니다. 이는 rank가 커질수록 업데이트해야 할 파라미터 수가 증가하여 연산량이 늘어나기 때문입니다.

메모리 점유율: Rank가 증가할수록 최대 메모리 점유율 (Max Alloc)이 증가합니다.  LoRA는 추가적인 파라미터를 도입하므로, rank가 커질수록 메모리 사용량이 늘어나는 것은 당연한 결과입니다.

LoRA의 장단점
장점:

효율적인 파라미터: LoRA는 전체 모델 파라미터를 fine-tuning 하는 대신, 훨씬 적은 수의 추가 파라미터 (rank에 의해 결정)만을 학습합니다. 이는 메모리 사용량과 학습 시간을 줄여줍니다. 특히, 본 실험에서는 Full fine-tuning에 비해 훨씬 적은 메모리로 학습이 가능함을 확인했습니다.
빠른 Fine-tuning: Pretrained 모델의 대부분은 그대로 두고, 저차원 행렬(low rank matrices)만을 훈련함으로, Full Fine Tuning에 비해 학습 속도가 빠릅니다.
Fine-tuning 된 모델 대비 좋은 성능: LoRA는 작은 수의 파라미터만으로도 기존 fine-tuning과 유사하거나 더 나은 성능을 달성할 수 있습니다.
단점:

Rank 선택의 어려움: 적절한 rank 값을 찾는 것이 중요합니다. 너무 낮은 rank는 모델의 표현력을 제한하여 성능 저하를 유발할 수 있고, 너무 높은 rank는 LoRA의 효율성을 감소시키고 과적합(overfitting)을 야기할 수 있습니다.
추가적인 메모리 소모: Rank가 커질수록 메모리 사용량이 증가합니다. 비록 full fine-tuning보다는 적지만, rank에 따라 추가적인 메모리가 필요합니다.
학습 속도 감소: Rank가 증가함에 따라 학습 속도가 느려집니다.
결론
LoRA는 제한된 컴퓨팅 자원으로 대규모 언어 모델을 효과적으로 fine-tuning 할 수 있는 강력한 기법입니다.  Rank 값은 성능, 속도, 메모리 사용량 간의 균형을 조절하는 중요한 하이퍼파라미터입니다.  본 실험에서는 rank 128이 loss와 메모리 사용량 측면에서 적절한 균형을 제공하는 것으로 보입니다. 하지만 최적의 rank는 task와 데이터셋에 따라 달라질 수 있으므로, 실험을 통해 찾아야 합니다.
