import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
import nltk
import json
import pandas as pd
import os
import torch

# Set MPS high watermark ratio (for Mac M1/M2 GPUs, if you're using one)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Download NLTK data (only needed once, but safe to keep)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# 모델 및 토크나이저 로드 함수 (with error handling and device handling)
@st.cache_resource
def load_model_and_tokenizer(model_path, peft_model_path=None):
    """Loads the model and tokenizer, handling potential errors."""
    try:
        # Force the use of the *slow* tokenizer (SentencePiece) for LLaMA.
        # This is the key change to avoid the Tiktoken conversion error.
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)  # legacy=True is crucial

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and config.torch_dtype == "bfloat16" else torch.float16,
            device_map="auto",
        )
        model.eval()

        if peft_model_path:
            peft_config = PeftConfig.from_pretrained(peft_model_path)
            model = PeftModel.from_pretrained(model, peft_model_path)
            model.eval()
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None



def generate_text(model, tokenizer, prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True):
    """Generates text using the model and tokenizer."""
    try:
        encoded_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        # Move to the correct device (CPU or GPU)
        if model.device.type != 'cpu':
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():  # Disable gradient calculation during inference
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
                # eos_token_id=tokenizer.eos_token_id,  # Consider adding EOS token handling if needed.
            )

        generated_texts = []
        for output in outputs:
            decoded_text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(decoded_text)

        return generated_texts

    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return [""]  # Return an empty string in case of error



def evaluate_model(model, tokenizer, test_dataset):
    """Evaluates the model using BLEU, ROUGE, and METEOR metrics."""
    bleu_scores = []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    meteor_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    meteor = evaluate.load("meteor")
    smoothie = SmoothingFunction().method1

    for example in test_dataset:
        prompt = f"Instruction: {example['instruction']} Response: "
        generated_texts = generate_text(model, tokenizer, prompt, max_new_tokens=150)
        if not generated_texts: # if the generated text is empty continue.
            continue

        generated_text = generated_texts[0]
        reference = example["response"]

        tokenized_gen = nltk.word_tokenize(generated_text.lower())
        tokenized_ref = [nltk.word_tokenize(reference.lower())]  # List of lists for sentence_bleu

        # Handle empty tokenized output (avoids errors with BLEU)
        if not tokenized_gen:
            bleu = 0
        else:
            bleu = sentence_bleu(tokenized_ref, tokenized_gen, smoothing_function=smoothie)
        bleu_scores.append(bleu)


        scores = scorer.score(reference, generated_text)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

        meteor_score = meteor.compute(predictions=[generated_text], references=[reference])
        meteor_scores.append(meteor_score['meteor'])

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {key: sum(value) / len(value) if value else 0 for key, value in rouge_scores.items()}
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    return {"BLEU": avg_bleu, "ROUGE": avg_rouge, "METEOR": avg_meteor}



def main():
    """Main function for the Streamlit app."""
    st.title("모델 성능 비교 (훈련 전 vs. 훈련 후)")

    # Sidebar for model and dataset paths
    base_model_path = st.sidebar.text_input("기본 모델 경로", "openlm-research/open_llama_3b")
    peft_model_path = st.sidebar.text_input("PEFT 모델 경로", "./korea-travel-llama")
    test_data_path = st.sidebar.text_input("테스트 데이터셋 경로", "corpus.json")

    # Load models and tokenizers (handle potential errors)
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_path)
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(base_model_path, peft_model_path)

    if base_model is None or base_tokenizer is None or finetuned_model is None or finetuned_tokenizer is None:
        st.error("모델 로딩에 실패했습니다. 경로를 확인하고 다시 시도해주세요.")
        return  # Exit if model loading failed

    # Load the test dataset (with error handling)
    try:
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_dataset = json.load(f)
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {test_data_path}")
        test_dataset = []
    except json.JSONDecodeError:
        st.error(f"JSON 디코딩 오류: {test_data_path} 파일의 형식이 올바르지 않습니다.")
        test_dataset = []
    except Exception as e:
        st.error(f"테스트 데이터셋 로드 실패: {e}")
        test_dataset = []


    # Quantitative Evaluation Section
    if st.button("정량적 평가 실행"):
        if test_dataset:
            with st.spinner("평가 중..."):
                base_model_results = evaluate_model(base_model, base_tokenizer, test_dataset)
                finetuned_model_results = evaluate_model(finetuned_model, finetuned_tokenizer, test_dataset)

                st.subheader("훈련 전 모델 평가 결과")
                st.write(base_model_results)

                st.subheader("훈련 후 모델 평가 결과")
                st.write(finetuned_model_results)

                st.subheader("결과 비교")
                # Create a DataFrame for comparison
                df = pd.DataFrame({
                    "Metric": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"],
                    "훈련 전": [base_model_results['BLEU'], base_model_results['ROUGE']['rouge1'],
                              base_model_results['ROUGE']['rouge2'], base_model_results['ROUGE']['rougeL'],
                              base_model_results['METEOR']],
                    "훈련 후": [finetuned_model_results['BLEU'], finetuned_model_results['ROUGE']['rouge1'],
                              finetuned_model_results['ROUGE']['rouge2'], finetuned_model_results['ROUGE']['rougeL'],
                              finetuned_model_results['METEOR']],
                })
                st.table(df)  # Use st.table for a cleaner display
        else:
            st.warning("테스트 데이터셋을 로드해주세요.")



    # Qualitative Evaluation Section
    st.subheader("정성적 평가")
    user_input = st.text_area("질문을 입력하세요:", height=100)

    # Sidebar options for qualitative evaluation
    max_new_tokens_qualitative = st.sidebar.number_input("최대 생성 토큰 수 (max_new_tokens)", 50, 512, 150)
    do_sample_qualitative = st.sidebar.checkbox("샘플링 사용 (do_sample=True)", value=True)


    if st.button("답변 생성 및 비교"):
        if user_input:
            prompt = f"Instruction: {user_input} Response: "

            with st.spinner("훈련 전 모델 답변 생성 중..."):
                base_model_response = generate_text(
                    base_model, base_tokenizer, prompt, max_new_tokens=max_new_tokens_qualitative, do_sample=do_sample_qualitative
                )[0]

            with st.spinner("훈련 후 모델 답변 생성 중..."):
                finetuned_model_response = generate_text(
                    finetuned_model, finetuned_tokenizer, prompt, max_new_tokens=max_new_tokens_qualitative, do_sample=do_sample_qualitative
                )[0]


            st.subheader("훈련 전 모델 답변:")
            st.write(base_model_response)

            st.subheader("훈련 후 모델 답변:")
            st.write(finetuned_model_response)

        else:
            st.warning("질문을 입력해주세요.")

if __name__ == "__main__":
    main()