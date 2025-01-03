심화과제: Pre-trained 모델로 효율적인 NLP 모델 학습하기

Q1) 어떤 task를 선택하셨나요?
Task: Named Entity Recognition (NER)

NER은 문장에서 개체명(예: 사람, 장소, 조직 등)을 인식하는 문제입니다. 감정 분석과 같은 문장 분류가 아니라 각 토큰(단어)을 분류하는 Token Classification 문제입니다. 이 과제에서는 주어진 NER 데이터셋을 사용하여 DistilBERT 모델을 Fine-tuning하여 NER 문제를 해결하고자 했습니다.

Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
모델 설계:

입력 형태: 모델의 입력은 문장으로 주어지며, 각 문장은 단어 단위로 토큰화됩니다. 문장은 DistilBERTTokenizer를 사용하여 토큰화된 후, input_ids와 attention_mask로 변환됩니다.
출력 형태: 모델의 출력은 각 단어에 대한 개체명 태그입니다. 각 토큰에 대해 예측된 클래스(개체명)의 레이블이 출력됩니다. 출력 크기는 입력 문장의 토큰 수와 동일하며, 각 토큰에 대해 하나의 label(예: 'O', 'B-PER', 'I-PER' 등)을 할당합니다.

모델 구조:

Input: Tokenized words of a sentence (input_ids, attention_mask)

Output: Label corresponding to each token in the input sentence (e.g., 'O', 'B-PER', 'I-PER', etc.)

입력 예시:

Sentence: "Barack Obama was born in Hawaii."

Input: ["Barack", "Obama", "was", "born", "in", "Hawaii"]

Output (Labels): ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC']


Q3) 어떤 pre-trained 모델을 활용하셨나요?
Pre-trained 모델: DistilBERT

영어 텍스트를 이해하는 능력을 제공하는 DistilBERT는 BERT의 경량화된 버전으로, 빠르고 효율적으로 동작하며, 제한된 리소스에서 잘 작동합니다. 이 모델을 사용하여 NER 작업을 수행하였고, token classification을 위해 fine-tuning을 진행했습니다.

DistilBERT 모델 선택 이유:
빠른 학습과 예측 속도
GPU 리소스가 제한적인 환경에서 잘 작동(m1 pro)
BERT 기반의 모델로서 높은 성능을 유지하면서도 작은 모델 크기

Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요?

1. Loss Curve:
Pre-trained 모델 (DistilBERT)을 fine-tuning했을 때, Epoch 1에서 Train Loss: 0.0049로 상대적으로 높은 손실을 보였으며, Test Loss: 0.0001로 낮은 정확도를 보였습니다.
하지만, 이후 Epoch 2와 Epoch 3에서 손실이 많이(?) 감소하며 Train Loss와 Test Loss가 모두 거의 0에 가까운 값으로 수렴했습니다.

Pre-trained 모델의 손실은 첫 번째 에폭에서 상대적으로 높았지만, 후속 에폭에서 급격히 개선되었으며, 빠르게 수렴하는 양상을 보였습니다.
Non-pretrained 모델을 학습했을 때는 초기 Epoch 1에서 Train Loss: 0.0017로 상대적으로 낮은 손실을 보였고, 이후 Epoch 2와 Epoch 3에서 손실이 급격히 감소하면서 Train Loss와 Test Loss가 모두 0에 가까워졌습니다.

Non-pretrained 모델은 상대적으로 초기 손실이 낮고, 수렴 속도가 빨랐습니다.

사실 너무 빠르게 0으로 나와 이게 맞는건지 궁금합니다.
1 epoch 진행하는데 몇 시간이 걸리다보니 많이 돌려보지 못했습니다.

2. 성능 비교:

Pretrained Epoch 1/3
Train Loss: 0.0049, Train Accuracy: 99.97%
Test Loss: 0.0001, Test Accuracy: 100.00%

Pretrained Epoch 2/3
Train Loss: 0.0001, Train Accuracy: 100.00%
Test Loss: 0.0000, Test Accuracy: 100.00%

Pretrained Epoch 3/3
Train Loss: 0.0000, Train Accuracy: 100.00%
Test Loss: 0.0000, Test Accuracy: 100.00%

Non-pretrained Epoch 1/3
Train Loss: 0.0017, Train Accuracy: 99.94%
Test Loss: 0.0000, Test Accuracy: 100.00%

Non-pretrained Epoch 2/3
Train Loss: 0.0000, Train Accuracy: 100.00%
Test Loss: 0.0000, Test Accuracy: 100.00%

Non-pretrained Epoch 3/3
Train Loss: 0.0000, Train Accuracy: 100.00%
Test Loss: 0.0000, Test Accuracy: 100.00%

3. 결과 분석:
솔직히 말하자면, pre-trained 모델과 non-pre-trained 모델 간의 차이가 그렇게 크게 나지 않았습니다.
훈련 정확도와 테스트 정확도가 각각 99.97%와 99.94%로 거의 비슷하게 나왔고, 손실도 그 차이가 크지 않았습니다.
사실, 99.97%와 99.94%는 차이가 매우 적다고 느껴졌고, 모델이 잘 학습된 것인지, 아니면 데이터셋이 잘 구성되어 있어서 그런지 확신이 서지 않았습니다.

4. 내 생각:
이번 과제를 통해 pre-trained 모델을 사용한 것과 사용하지 않은 것의 성능 차이를 크게 느끼지 못했습니다.

실제로 두 모델의 성능 차이가 0.03% 이내로 매우 미미해서, 모델이 잘 학습된 것인지 아니면 데이터셋이 잘 준비되어 있어서 그런지 확신이 서지 않았습니다.
비슷한 성능이 나오는 이유는 아마도 데이터셋이 잘 구성되어 있어서일 수도 있고, 오버피팅이 일어난 것일 수도 있습니다. 데이터를 잘 준다면 사실 fine-tuning 없이도 꽤 좋은 성능을 낼 수 있지 않나 하는 생각이 듭니다.

5. Loss/Accuracy 그래프:
훈련 및 테스트 동안의 손실과 정확도 곡선을 비교한 그래프는 3주차 심화과제.ipynb 맨 아래에 있습니다.
그렇지만, 차이를 크게 느끼지 못한 점을 감안하여 확인해 주세요.

결론
Pre-trained 모델 (DistilBERT)과 Non-pretrained Transformer를 비교했을 때, 두 모델 간의 성능 차이가 그렇게 크게 나지 않았습니다. 훈련 정확도와 테스트 정확도가 거의 동일했고, 손실 또한 비슷한 수준이었습니다. 이로 인해 pre-trained 모델의 효과를 체감하지 못했습니다.

원인 분석:

데이터셋이 잘 구성되어 있어서 그런 것인지,
데이터 불균형인지... 아니면 fine-tuning 모델을 잘 못 사용한것인지..
오버피팅이 일어난 것인지 확신이 서지 않았습니다.
또한 MacBook M1 Pro의 한계로 인해 GPU 연산 시간이 길어졌고, 이로 인해 빠른 실험을 하지 못한 점도 성능 차이를 크게 느끼지 못한 원인일 수 있습니다.
결국, 작은 데이터셋에서는 pre-trained 모델과 non-pretrained 모델의 차이가 크지 않을 수도 있겠다는 생각이 들었습니다. 하지만, 큰 데이터셋이나 다양한 작업에서는 pre-trained 모델이 더 큰 차이를 보일 수 있을 것입니다.
처음하다보니 이렇게 오래 걸리는 경우 이게 맞는지 틀린지에 대해 예측이 잘 안되어 아쉽습니다.
이럴 경우 어떤 장치들을 넣어서 확인하는게 좋을까요? 배치크기, max_len 등은 조절해봤습니다.
일단 가장 아쉬운건 1 에포크당 4시간??? 이렇게 나오다보니 결과 확인이 너무 어려워서 많은 시도는 못해봤네요 ㅠㅠ
감사합니다.
