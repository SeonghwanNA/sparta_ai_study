# LLaMA 모델 한국 여행 데이터 기반 파인튜닝 및 성능 비교(나만의 LLM 서비스에 경량화 기법들 적용해보기)

## 테스트 결과 (제한 사항)

LLaMA 모델의 성능을 비교하고 싶었으나 AWS 비용 문제(20만원 넘게 초과되어 해보지 못함...)와 장비(로컬 머신)의 한계로 인해 충분한 테스트를 수행하지 못했습니다.  
flash_attention도 적용해봤으나 gpu가 필요하다하여... cpu로 돌릴 수 있는 lora와 sfttrain만 적용해서 해봤습니다.
추후 환경이 된다면 이것저것 해보고 싶네요.
보다 정확하고 신뢰성 있는 결과를 얻으려면 더 큰 규모의 컴퓨팅 자원(예: 더 많은 GPU 메모리, 더 긴 훈련 시간)이 필요합니다.


https://wandb.ai/ainabakgood-personal/instruction-tuning-korea-travel-lora?nw=nwuserainabakgood&panelDisplayName=eval%2Floss&panelSectionName=eval

이 프로젝트는 LLaMA 기반 언어 모델을 한국 여행 관련 데이터(`corpus.json`)로 파인튜닝하고, 훈련 전후 모델의 성능을 비교하는 Streamlit 웹 애플리케이션을 제공합니다. LoRA (Low-Rank Adaptation) 기법을 사용하여 효율적인 파인튜닝을 수행합니다.

## 프로젝트 구성

이 프로젝트는 크게 두 가지 부분으로 구성됩니다:

1.  **`8week_advanced.py` (파인튜닝 스크립트):**
    *   LLaMA 모델을 한국 여행 데이터로 파인튜닝합니다.
    *   LoRA를 사용하여 모델 파라미터의 일부만 업데이트하여 효율성을 높입니다.
    *   `transformers`, `peft`, `datasets` 라이브러리를 사용합니다.
    *   Wandb (Weights & Biases)를 사용하여 실험 추적 및 로깅을 수행합니다.
    *   파인튜닝된 모델을 `./korea-travel-llama` 디렉토리에 저장합니다.

2.  **`test_llama_lora.py` (Streamlit 애플리케이션):**
    *   파인튜닝 전/후 모델의 성능을 비교하는 웹 인터페이스를 제공합니다.
    *   정량적 평가 (BLEU, ROUGE, METEOR)와 정성적 평가 (사용자 질문에 대한 답변 생성)를 수행합니다.
    *   `streamlit`, `transformers`, `peft`, `nltk`, `rouge_score`, `evaluate`, `pandas` 라이브러리를 사용합니다.

## `8week_advanced.py` 상세 설명

### 기능

*   **데이터 로드 및 전처리:**
    *   `corpus.json` 파일에서 한국 여행 관련 질의응답 데이터를 로드합니다.
    *   `datasets` 라이브러리를 사용하여 데이터를 훈련/검증 세트로 분할합니다.
    *   `LlamaTokenizer`를 사용하여 텍스트 데이터를 모델 입력 형식(토큰 ID)으로 변환합니다.
    *   패딩(padding)을 적용하여 모든 입력 시퀀스의 길이를 동일하게 만듭니다.
*   **모델 로드 및 LoRA 적용:**
    *   Hugging Face Hub에서 `openlm-research/open_llama_3b` 모델을 로드합니다.
    *   `peft` 라이브러리를 사용하여 LoRA 설정을 적용하고, 훈련 가능한 파라미터 수를 줄입니다.
*   **모델 훈련:**
    *   `transformers` 라이브러리의 `Trainer` 클래스를 사용하여 모델을 파인튜닝합니다.
    *   훈련 중 손실(loss)을 Wandb에 기록합니다.
    *   `TrainingArguments`를 통해 훈련 하이퍼파라미터 (배치 크기, 학습률, 에폭 수 등)를 설정합니다.
    *   `WandbCallback`을 사용하여 훈련 중 Wandb에 로그를 기록합니다.
* **모델 평가**
    * `Trainer`클래스를 통해, 모델을 평가하고, evaluation loss를 wandb에 기록합니다.
*   **모델 저장:**
    *   파인튜닝된 모델을 `./korea-travel-llama` 디렉토리에 저장합니다. (adapter_config.json, adapter_model.safetensors)

### 사용 방법 (8week_advanced.py)

1.  **사전 준비:**
    *   Python 3.8 이상, 필요한 라이브러리 설치 
    *   `corpus.json` 파일 준비
    *   (선택 사항) Wandb 계정 생성 및 로그인 (`wandb login`).

2.  **실행:**
    *   터미널에서 `8week_advanced.py` 파일이 있는 디렉토리로 이동합니다.
    *   다음 명령을 실행합니다:

        ```bash
        python 8week_advanced.py
        ```

3.  **결과 확인:**
    *   Wandb 웹사이트에서 훈련 진행 상황 및 결과를 확인할 수 있습니다.
    *   파인튜닝된 모델은 `./korea-travel-llama` 디렉토리에 저장됩니다.

## `test_llama_lora.py` 상세 설명

### 기능

*   **모델 로드:**  (기존 README 내용 +) `legacy=True`를 사용하여 SentencePiece 토크나이저를 사용.
*   **텍스트 생성:** (기존 README 내용)
*   **정량적 평가:** (기존 README 내용)
*   **정성적 평가:** (기존 README 내용)
*   **Streamlit UI:** (기존 README 내용)

### 사용 방법 (test_llama_lora.py)

(이전 README 내용과 동일)

1.  **사전 준비:** (기존 README 내용 +) `8week_advanced.py`를 실행하여 파인튜닝된 모델(`korea-travel-llama`)을 생성해야 합니다.
2.  **실행:** (기존 README 내용)
3.  **사용:** (기존 README 내용)

## 테스트 결과 (제한 사항)

이 코드는 LLaMA 모델의 파인튜닝 및 성능 비교를 위해 작성되었으나, AWS 비용 문제와 장비(로컬 머신)의 한계로 인해 충분한 테스트를 수행하지 못했습니다.  보다 정확하고 신뢰성 있는 결과를 얻으려면 더 큰 규모의 컴퓨팅 자원(예: 더 많은 GPU 메모리, 더 긴 훈련 시간)이 필요합니다.  이 코드는 기본적인 기능과 테스트 프레임워크를 제공하지만, 실제 프로덕션 환경에서 사용하기 전에 추가적인 테스트와 개선이 필요합니다.

## 개선할 점 (To-Do)

*   **더 큰 데이터셋:** 더 크고 다양한 한국 여행 관련 데이터셋을 사용하여 파인튜닝하면 모델 성능을 향상시킬 수 있습니다.
*   **하이퍼파라미터 튜닝:**  `TrainingArguments`의 하이퍼파라미터 (학습률, 배치 크기, 에폭 수 등)를 조정하여 최적의 설정을 찾을 수 있습니다.
*   **다른 LoRA 설정:** LoRA의 `r`, `alpha`, `target_modules` 등의 파라미터를 변경하여 성능 변화를 관찰할 수 있습니다.
*   **더 큰 LLaMA 모델:** `open_llama_3b` 대신 더 큰 LLaMA 모델 (7b, 13b 등)을 사용하면 성능이 향상될 수 있습니다. (더 많은 컴퓨팅 자원이 필요합니다.)
*   **정량적 평가 자동화:** `test_llama_lora.py`에서 테스트 데이터셋을 자동으로 로드하고, 정량적 평가를 수행하는 기능을 추가할 수 있습니다. (현재는 Streamlit UI에서 수동으로 실행해야 합니다.)
* **정성적 평가 개선:** 사용자가 여러 질문을 한 번에 입력하고, 각 질문에 대한 답변을 비교할 수 있도록 개선할 수 있습니다.
* **fast tokenizer 사용 시도:** transformers와 tiktoken의 업데이트에 따라, `legacy = False`설정으로도 fast tokenizer를 사용할 수 있는지 다시 시도해볼 수 있습니다.
* **Error Handling 보강:** 더 많은 부분에서 발생할 수 있는 예외를 처리하고, 사용자에게 친절한 오류 메시지를 제공할 수 있습니다.
* **배치 추론(Batch Inference):** 정량적 평가 시 여러 예제를 배치로 처리하여 평가 속도를 높일 수 있습니다.
* **조기 종료 (Early Stopping):** 훈련 중 검증 성능이 더 이상 향상되지 않으면 훈련을 조기에 중단하여 시간과 자원을 절약할 수 있습니다.
