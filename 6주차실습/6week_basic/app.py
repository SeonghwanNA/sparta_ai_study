import base64
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경 변수에서 OPENAI_API_KEY 가져오기
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Fashion Recommendation Bot")
model = ChatOpenAI(model="gpt-4o-mini")

# 여러 이미지를 입력받도록 수정
images = st.file_uploader("사진을 여러 장 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if images:
    # 업로드된 이미지들을 표시
    for image in images:
        st.image(image)

    # 이미지들을 base64로 인코딩
    encoded_images = [base64.b64encode(image.read()).decode("utf-8") for image in images]

    # 사용자 질문을 입력받을 텍스트 박스
    user_question = st.text_input("질문을 입력하세요:")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            # 모든 이미지 정보를 담은 messages 생성
            messages = [
                HumanMessage(
                    content=[
                                {"type": "text", "text": user_question},  # 사용자 질문 추가
                            ] + [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                                } for encoded_image in encoded_images  # 모든 이미지 정보 추가
                            ]
                )
            ]
            result = model.invoke(messages)
            response = result.content
            st.markdown(response)