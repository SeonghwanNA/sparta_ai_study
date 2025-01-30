import openai
import re
import random
import os
from PIL import Image
import base64
import asyncio
import streamlit as st
import logging
import calendar
from datetime import datetime, timedelta
import pyperclip

import city_secrets

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchResults
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from fuzzywuzzy import fuzz


# 환경 변수에서 OPENAI_API_KEY 가져오기
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

travel_plan_text = ""
city_info_text = ""  # 전역 변수 선언
flag_start_plan = False  # 전역 변수 선언

# DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchResults()

# LangChain ChatOpenAI 모델 초기화
chat = ChatOpenAI(model_name="gpt-4o-mini")

# 로깅 설정
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def convert_korean_number_to_int(korean_number):
    """
    한글 숫자를 정수로 변환하는 함수
    """
    korean_numbers = {
        "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
        "육": 6, "칠": 7, "팔": 8, "구": 9, "십": 10,
        "백": 100, "천": 1000, "만": 10000, "억": 100000000, "조": 1000000000000
    }
    result = 0
    temp = 0
    for digit in korean_number:
        if digit in korean_numbers:
            number = korean_numbers[digit]
            if number >= 10:
                if temp == 0:
                    temp = 1
                result += temp * number
                temp = 0
            else:
                temp = temp * 10 + number
    result += temp
    return result


async def generate_travel_plan_prompt(user_input, city_name, start_day, days, city_start_day, city_start_date, city_order, num_cities,
                                      next_city=None):
    """
    여행 계획 생성 프롬프트 (도착/출발, 도시 이름, 시작일, 기간, 도시 시작일, 도시 순서, 다음 도시 정보 포함)
    """

    # user_input 대신 현재 도시 정보만 사용
    city_input = f"{city_name}에서 {start_day}일차부터 {start_day + days - 1}일차까지 {days}일 동안의 여행 계획을 짜줘."

    # 랜덤 도착 시간 생성 (08:00 ~ 23:00)
    arrival_hour = random.randint(7, 20)
    arrival_time = f"{arrival_hour:02d}:00"
    checkin_hour = (arrival_hour + 3) % 24  # 도착 시간 3시간 후 체크인
    checkin_time = f"{checkin_hour:02d}:00"

    # 랜덤 출발 시간 생성 (17:00 ~ 20:00)
    departure_hour = random.randint(17, 20)
    departure_time = f"{departure_hour:02d}:00"

    # Google Search API를 사용하여 여행 정보 검색
    retrieved_information = search_tool.run(f"{city_name} 여행 정보")

    # 조건부 출력 형식
    if city_order == 1:
        template = f"""
                \n\n
                다음 정보를 참고하여 {city_name}에서 {start_day}일차부터 {start_day + days - 1}일차까지 {days}일 동안의 여행 계획을 짜줘:
                {retrieved_information}

                {city_name}에 도착하는 날짜를 {city_start_day}일차로 계산해줘.
                {city_name}에 도착하는 날짜는 {city_start_date.strftime('%Y년 %m월 %d일')}입니다. 


                구글지도에 장소명을 넣을 때 영어 장소명을 넣어줘.
                장소명, 숙소이름, 간식 파는 곳 이름, 숙소명 모두 영어로 표시되게 해줘. 
                나머지 설명은 한글로 해줘        
                모든건 구글지도 기반으로 작성되어야해
                next_city로 이동을 했는데 00:00을 지나면 다음 일차로 생성을 해주고 그 날 일차 일정을 보여줘
                숙소는 호텔, 콘도, 에어비엔비 등 모두 가능해
                숙소는 구글 지도에서 리뷰수 1000건 이상 and 별점 4.2 이상을 보여줘
                {city_name} 공항/기차역에서 숙소까지 이동하는 방법을 알려줘야해
                숙소 체크인 시간을 먼저 알려주고 짐을 풀고 난 후 시간부터 여행 일정을 만들어줘                
                일차별로 최대 23:00까지 일정을 짜줘
                하지만 23:00까지는 안짜도 되는거야 유기적으로 알아서 짜줘            
                만약 사용자가 출발일을 입력하면 그 일자 그 계절에 맞게 추천해줘    
                """
        if num_cities > 1:  # 마지막 도시가 아닌 경우
            template += f"""마지막날에 {next_city}(으)로 떠나는 일정 + 그 도시에 도착해서 숙소 잡는 일정도 보여줘"""
        else:
            template += f"""마지막날 마지막 시간에 집으로 떠나는 일정을 보여줘"""

        template += f"""
                가격은 그 나라가 사용하는 화폐로 보여주고 환율 계산해서 (₩ : )로도 보여줘

                ## {city_name} 여행 일정

                **1일차**

                * {arrival_time} {city_name} 도착 ({city_name} 공항/기차역에서 숙소까지 이동 방법, 소요시간)
                * {checkin_time} 숙소 체크인 
                    • 추천 숙소 이름
                      (구글 별점 : , 리뷰 건수 : )
                    - 가장 저렴한 방 가격 ~ :
                    - 추천 이유 :
                    - https://www.google.com/maps/search/{city_name}추천 숙소 이름
                * {checkin_time} 이후 {city_name} 여행 시작

                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 저녁 식사 
                     * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                     * 추천 메뉴 :
                     * 가격 :
                     * https://www.google.com/maps/search/{city_name}레스토랑 이름
                * **시간:** 선택 관광 (추가 활동 제안)

                **2일차**

                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 점심 식사 
                     * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                     * 추천 메뉴 :
                     * 가격 :
                     * https://www.google.com/maps/search/{city_name}레스토랑 이름
                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명     
                * **시간:** 저녁 식사 
                     * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                     * 추천 메뉴 :
                     * 가격 :
                     * https://www.google.com/maps/search/{city_name}레스토랑 이름
                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명      
                * **시간:** 선택 관광 (추가 활동 제안)

                ...

                **{days}일차**

                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 장소명
                     * 활동 내용:
                     * 이동 방법, 소요시간:
                     * 예상 비용:
                     * https://www.google.com/maps/search/{city_name}장소명
                * **시간:** 점심 식사 
                     * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                     * 추천 메뉴 :
                     * 가격 :
                     * https://www.google.com/maps/search/{city_name}레스토랑 이름
                * **시간:** {city_name}에서 출발 준비 (다음 도시 이동 정보 또는 귀국 정보)


                **추가 정보:**

                * 각 장소의 입장료, 교통비 등 예상 비용 정보를 포함
                * 여행 팁 (예: 교통 패스, 환전, 유용한 앱 등)
                * 맛집 정보 (현지 음식, 추천 레스토랑 등)
                * 쇼핑 정보 (기념품, 특산품 등)
                * 안전 정보 (주의 사항, 비상 연락처 등)
                * 추가 관광지 정보 (시간이 남을 경우 방문할 만한 곳)
                * 날씨 정보 (여행 기간 동안의 예상 날씨)
                """
    else:  # 첫 번째 도시가 아닌 경우
        template = f"""
                \n\n
                다음 정보를 참고하여 {city_name}에서 {start_day}일차부터 {start_day + days - 1}일차까지 {days}일 동안의 여행 계획을 짜줘:
                {retrieved_information}

                {city_name}에 도착하는 날짜를 {city_start_day}일차로 계산해줘.

                구글지도에 장소명을 넣을 때 영어 장소명을 넣어줘.
                장소명, 숙소이름, 간식 파는 곳 이름, 숙소명 모두 영어로 표시되게 해줘.
                나머지 설명은 한글로 해줘                
                모든건 구글지도 기반으로 작성되어야해
                시간은 09:00 이렇게 꼭 표시해줘
                next_city로 이동을 했는데 00:00을 지나면 다음 일차로 생성을 해주고 그 날 일차 일정을 보여줘
                숙소는 호텔, 콘도, 에어비엔비 등 모두 가능해
                숙소는 구글 지도에서 리뷰수 1000건 이상 and 별점 4.2 이상을 보여줘     
                만약 사용자가 출발일을 입력하면 그 일자 그 계절에 맞게 추천해줘                           
                """
        if city_order != num_cities:  # 마지막 도시가 아닌 경우
            template += f"""마지막날에 {next_city}(으)로 떠나는 일정 + 그 도시에 도착해서 숙소 잡는 일정도 보여줘"""
        else:
            template += f"""마지막날 마지막 시간에 집으로 떠나는 일정을 보여줘"""

        template += f"""
                일차별로 최대 23:00까지 일정을 짜줘
                하지만 23:00까지는 안짜도 되는거야 유기적으로 알아서 짜줘                                
                가격은 그 나라가 사용하는 화폐로 보여주고 환율 계산해서 (₩ : )로도 보여줘
                
                ## {city_name} 여행 일정

            **1일차 ({city_start_day}일차)**

            **시간별 일정:**

            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 저녁 식사 
                 * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                 * 추천 메뉴 :
                 * 가격 :
                 * https://www.google.com/maps/search/{city_name}레스토랑 이름
            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 선택 관광 (추가 활동 제안)

            **2일차 ({city_start_day + 1}일차)**

            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 점심 식사 
                 * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                 * 추천 메뉴 :
                 * 가격 :
                 * https://www.google.com/maps/search/{city_name}레스토랑 이름
            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 저녁 식사 
                 * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                 * 추천 메뉴 :
                 * 가격 :
                 * https://www.google.com/maps/search/{city_name}레스토랑 이름
            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 선택 관광 (추가 활동 제안)
            

            ...

            **{days}일차 ({city_start_day + days - 1}일차)**

            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 장소명
                 * 활동 내용:
                 * 이동 방법, 소요시간:
                 * 예상 비용:
                 * https://www.google.com/maps/search/{city_name}장소명
            * **시간:** 점심 식사 
                 * 레스토랑 이름 (구글 별점 : , 리뷰 건수 : )
                 * 추천 메뉴 :
                 * 가격 :
                 * https://www.google.com/maps/search/{city_name}레스토랑 이름
            * **시간:** {city_name}에서 출발 준비 (다음 도시 이동 정보 또는 귀국 정보)


            **추가 정보:**

            * 각 장소의 입장료, 교통비 등 예상 비용 정보를 포함
            * 여행 팁 (예: 교통 패스, 환전, 유용한 앱 등)
            * 맛집 정보 (현지 음식, 추천 레스토랑 등)
            * 쇼핑 정보 (기념품, 특산품 등)
            * 안전 정보 (주의 사항, 비상 연락처 등)
            * 추가 관광지 정보 (시간이 남을 경우 방문할 만한 곳)
            * 날씨 정보 (여행 기간 동안의 예상 날씨)
            """

    return template, departure_time, arrival_time, start_day  # template 대신 prompt 반환


async def generate_city_info_prompt(city_name):
    """
    특정 도시의 정보를 요청하는 프롬프트
    """
    return f"""
    {city_name}의 여행 관련 정보를 생성해 주세요.

    오직 구글 지도상 {city_name}에 있는 곳에서 추천 식당 5곳, 미슐랭 3곳, 간식 5곳, 숙소 5곳을 알려줘
    간식은 길거리 음식, 베이커리, 카페 위주로 해줘
    숙소는 호텔, 콘도, 에어비엔비 등 모두 가능해
    그 도시나 명소에 미슐랭 식당이 없으면 출력하지마
    한국(대한민국)은 미슐랭 식당을 보여주려면 도시로 서울이 나왔을 때만 보여줘 나머진 없어
    추천 식당, 간식, 숙소는 구글 지도에서 리뷰수 1000건 이상 and 별점 4.2 이상인 것들만 보여줘
    블로그나 인터넷에 많이 언급된 곳 위주로만 알려줘
    구글지도에 장소명을 넣을 때 영어 장소명을 넣어줘
    만약 대한민군 여행지면 한국 장소명을 넣어줘
    가격은 그 나라가 사용하는 화폐로 보여주고 환율 계산해서 (₩ : )로도 보여줘
    Tip은 그 나라 그 도시에 맞는 팁들을 알려주기 
    사전 예약 같은 경우 꼭 해야하는 것들 명확한 명칭을 대면서 알려줘 예를들어 피렌체면 피렌체 카드를 구매하면이라던가
    유명한 식당 같은 경우 명칭을 알려주고 예약을 해야 먹을 수 있는지라던가 
    만약 사용자가 출발일을 입력하면 그 일자 그 계절에 맞게 추천해줘

    {city_name}(글씨 크기 15px 글씨 타입 bold)
    추천 식당(글씨 크기 13px 글씨 타입 bold)
    미슐랭 식당(글씨 크기 13px 글씨 타입 bold)
    간식(글씨 크기 13px 글씨 타입 bold)
    숙소(글씨 크기 13px 글씨 타입 bold)
    Tip(글씨 크기 13px 글씨 타입 bold) 
    

    \n\n
    
    {city_name}

    • 추천 식당(글씨 크기 13px 글씨 타입 bold)

    • 식당 이름 
      (구글 별점 :, 리뷰 건수 :, 종류:)
    - 추천메뉴 : 
    - 추천 메뉴의 가격 :
    - https://www.google.com/maps/search/{city_name}식당 이름                 

    ...

    • 미슐랭 식당

    • 식당 이름 
      (미슐랭 별점 :, 종류 :)
    - 추천메뉴 : 
    - 추천 메뉴의 가격 :
    - https://www.google.com/maps/search/{city_name}식당 이름 

    ...

    • 간식

    • 간식 파는 곳 이름 
      (구글 별점 :, 리뷰 건수:, 종류:)
    - 추천메뉴 : 
    - 추천 메뉴의 가격 :
    - https://www.google.com/maps/search/{city_name}간식 파는 곳 이름                 

    ...

    • 숙소

    • 숙소 이름 
      (구글 별점 :, 리뷰 건수:)
    - 가장 저렴한 방 가격 ~ :
    - 설명 :
    - https://www.google.com/maps/search/{city_name}숙소 이름                 

    ...

    • Tip

    • 날씨 : 
    • 교통패스 : 
    • 박물관 입장권 : 
    • 사전 예약 필수로 해야 하는 것들 :
    • 옷차림 : 
    • 가서 필수로 해봐야 하는 것 : 
    • 유명한 것 : 
    • 그 외에 알려주고 싶은 꿀팁들 :        

    ...

    """


async def get_openai_response(prompt):
    """
    OpenAI API를 이용하여 응답 생성 (비동기)
    """
    try:
        response = openai.chat.completions.create(  # await 키워드 제거
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful travel planner."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return ""


async def extract_city_name(user_input):
    """
    Langchain parser를 사용하여 사용자 입력에서 도시 이름을 추출하는 함수
    """
    logging.debug("extract_city_name")
    template = f"""
    다음 문장에서 여행할 나라이름 말고 도시 이름을 모두 추출해줘. 만약 도시 이름이 없다면 "없음"이라고 답해줘.
    여러 도시를 입력했을 경우 쉼표로 구분해줘.

    문장: {user_input}
    """

    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    # invoke 대신 __call__ 사용
    city_name = chat.invoke(prompt.format(user_input=user_input))
    logging.debug("city_name ddd")
    logging.debug(city_name)
    logging.debug(city_name.content.strip())
    logging.debug(city_name.content)
    return city_name.content.strip()


async def get_cities_from_country(user_input):
    """
    Langchain parser를 사용하여 나라 이름에서 도시 이름을 추출하는 함수
    """
    template = f"""
    다음 문장에서 여행할 나라를 찾아서, 그 나라의 유명한 도시들을 최대 3개까지 알려줘. 
    도시들은 그 나라의 구글 지도를 참고하여 이동하기 편한 순서대로 알려줘.
    딱 도시이름 3개만 출력해줘
    도시 이름, 도시 이름, 도시 이름
    만약 나라 이름을 찾을 수 없다면 "없음"이라고 답해줘.

    문장: {user_input}
    """

    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    # invoke 대신 __call__ 사용
    cities = chat.invoke(prompt.format(user_input=user_input))
    return cities.content.strip()


async def generate_accommodation_prompt(city_name):
    prompt = f"""
    {city_name}에서 숙소 10곳을 추천해줘. 
    호텔, 콘도, 에어비엔비 등 다양한 종류를 포함하고, 
    구글 지도에서 리뷰 수 1000개 이상이고 평점 4.2 이상인 곳으로만 추천해줘. 
    블로그나 인터넷에 많이 언급된 곳 위주로 알려주고, 
    구글 지도에 장소명을 넣을 때 영어 장소명을 넣어줘. 
    만약 대한민국 여행지면 한국 장소명을 넣어줘. 
    가격은 그 나라가 사용하는 화폐로 보여주고 환율 계산해서 (₩ : )로도 보여줘.
    숙소 추천 글씨 크기 15px 글씨 타입 bold


    ### 숙소 정보 출력 형식 ###
    {city_name} 숙소 추천

    • 숙소 이름
      (구글 별점 : , 리뷰 건수 : )
    - 가장 저렴한 방 가격 ~ :
    - 설명 :
    - https://www.google.com/maps/search/{city_name} 숙소 이름
    """
    return prompt


async def generate_restaurant_prompt(city_name):
    # RAG를 사용하여 도시 정보 가져오기
    # city_info1 = qa.run(f"{city_name}에 대한 여행 정보를 알려줘.")
    # 한국어 위키피디아에서 정보를 가져옵니다.
    # city_info2 = ko_qa.run(f"{city_name}에 대한 여행 정보를 알려줘.")

    # RAG를 사용하여 도시 정보 가져오기
    # city_info3 = qa_chain.run(f"{city_name}에 대한 여행 정보를 알려줘.")

    city_info = ""  # city_info1 + city_info2 #+ city_info3

    prompt = f"""
    {city_name}에서 현지 사람들이 자주 가는 식당 10곳을 추천해줘. 
    구글 지도에서 리뷰 수 1000개 이상이고 평점 4.2 이상인 곳으로만 추천해줘. 
    블로그나 인터넷에 많이 언급된 곳 위주로 알려주고, 
    구글 지도에 장소명을 넣을 때 영어 장소명을 넣어줘. 
    만약 대한민국 여행지면 한국 장소명을 넣어줘. 
    가격은 그 나라가 사용하는 화폐로 보여주고 환율 계산해서 (₩ : )로도 보여줘.
    식당 추천 글씨 크기 15px 글씨 타입 bold

    ### 식당 정보 출력 형식 ###
    {city_name} 식당 추천


    • 식당 이름
      (구글 별점 : , 리뷰 건수 : , 종류 : )
    - 추천 메뉴 :
    - 추천 메뉴의 가격 :
    - https://www.google.com/maps/search/{city_name} 식당 이름

    ## 추가 정보 ##
    {city_info}
    """
    return prompt


def generate_michelin_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}에 있는 미슐랭 식당을 최대 10곳 알려줘. 
    미슐랭 가이드에 등재된 식당만 고려하고, 
    각 식당에 대한 간단한 설명과 함께 미슐랭 별점, 종류, 구글 지도 링크를 포함해줘. 
    만약 {city_name}에 미슐랭 식당이 없으면 "없음"이라고 답해줘.
    한국(대한민국)은 도시로 서울이 나왔을 때만 미슐랭 식당을 보여주고 나머지는 "없음"이라고 답해줘.
    미슐랭 식당 글씨 크기 15px 글씨 타입 bold

    """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""
    ### 미슐랭 식당 정보 출력 형식 ###
    {city_name} 미슐랭 식당

    • 식당 이름
      (미슐랭 별점 : , 종류 : )
    - 설명 :
    - https://www.google.com/maps/search/{city_name} 식당 이름
    """
    return prompt


def generate_must_do_prompt(city_name):
    prompt = f"""
    {city_name}에서 꼭 해야 하는 것들을 10개 추천해줘. 
    블로그나 인터넷에 많이 언급된 곳 위주로 알려주고, 
    구글 지도에 장소명을 넣을 때 영어 장소명을 넣어줘. 
    만약 대한민국 여행지면 한국 장소명을 넣어줘.
    

    ### 꼭 해야 하는 것 출력 형식 ###
    {city_name} 꼭 해야 하는 것 (글씨 크기 15px 글씨 타입 bold)

    • 활동 이름
    - 설명 :
    - https://www.google.com/maps/search/{city_name} 활동 이름 (활동과 관련된 장소가 있는 경우)
    """
    return prompt


def generate_preparation_prompt(city_name):
    prompt = f"""
    {city_name} 여행 전에 미리 준비해야 하는 것들을 알려줘. 
    비자, 항공권 예약, 환전, 여행자 보험, 짐 싸기, 필수 앱 등을 포함해서 알려줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    여행 준비 사항 글씨 크기 15px 글씨 타입 bold

    ### 준비 사항 출력 형식 ###
    {city_name} 여행 준비 사항

    • 준비 사항
    - 설명 :
    """
    return prompt


def generate_transportation_prompt(city_name):
    prompt = f"""
    {city_name}에서 이용 가능한 교통편에 대해 알려줘. 
    비행기, 기차, 버스, 지하철, 택시, 자전거 등 다양한 옵션을 포함하고, 
    각 교통편의 장단점, 가격, 예약 방법 등을 자세하게 알려줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    교통 정보 글씨 크기 15px 글씨 타입 bold

    ### 교통 정보 출력 형식 ###
    {city_name} 교통 정보 

    • 교통편
    - 설명 :
    - 가격 :
    - 예약 방법 :
    - 장점 :
    - 단점 :
    """
    return prompt


def generate_seasonal_info_prompt(city_name, month):
    prompt = f"""
    {month}월에 {city_name}을 여행할 때 어떤 옷차림이 적절한지, 
    날씨는 어떤지, 
    계절별로 즐길 수 있는 특별한 활동이나 축제가 있는지 알려줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    월 여행 정보 글씨 크기 15px 글씨 타입 bold

    ### 계절별 정보 출력 형식 ###
    {city_name} {month}월 여행 정보

    • 날씨 :
    • 옷차림 :
    • 특별한 활동 :
    • 축제 :
    """
    return prompt


def generate_attractions_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}의 유명한 관광 명소를 10곳 추천해줘. 
    역사적인 장소, 박물관, 미술관, 공원, 건축물 등 다양한 종류를 포함하고, 
    각 명소에 대한 간단한 설명과 함께 구글 지도 링크를 넣어줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    관광 명소 추천 글씨 크기 15px 글씨 타입 bold

    """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""
    ### 관광 명소 정보 출력 형식 ###
    {city_name} 관광 명소 추천

    • 명소 이름
    - 설명 :
    - https://www.google.com/maps/search/{city_name} 명소 이름
    """
    return prompt


def generate_shopping_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}에서 쇼핑하기 좋은 곳을 10곳 추천해줘. 
    백화점, 쇼핑몰, 시장, 특산품 판매점 등 다양한 종류를 포함하고, 
    각 장소에 대한 간단한 설명과 함께 구글 지도 링크를 넣어줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    쇼핑 추천 글씨 크기 15px 글씨 타입 bold

    """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""  
    ### 쇼핑 정보 출력 형식 ###
    {city_name} 쇼핑 추천

    • 장소 이름
    - 설명 :
    - https://www.google.com/maps/search/{city_name} 장소 이름
    """
    return prompt


def generate_entertainment_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}에서 즐길 수 있는 엔터테인먼트를 10개 추천해줘. 
    공연, 전시, 축제, 놀이공원, 클럽 등 다양한 종류를 포함하고, 
    각 엔터테인먼트에 대한 간단한 설명과 함께 구글 지도 링크를 넣어줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    엔터테인먼트 추천 글씨 크기 15px 글씨 타입 bold

    """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""      
    ### 엔터테인먼트 정보 출력 형식 ###
    {city_name} 엔터테인먼트 추천

    • 엔터테인먼트 이름
    - 설명 :
    - https://www.google.com/maps/search/{city_name} 엔터테인먼트 이름 (관련 장소가 있는 경우)
    """
    return prompt


def generate_weather_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}의 현재 날씨와 앞으로 5일간의 날씨 예보를 알려줘. 
    최고 기온, 최저 기온, 강수 확률 등을 포함해줘.
    날씨 정보 글씨 크기 15px 글씨 타입 bold

    """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""      
    ### 날씨 정보 출력 형식 ###
    {city_name} 날씨 정보

    • 현재 날씨 :
    • 5일간의 예보 :
    """
    return prompt


def generate_budget_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}을 여행하는 데 필요한 예상 경비를 알려줘. 
    숙박비, 식비, 교통비, 관광비, 쇼핑비 등을 포함하고, 
    여행 스타일 (배낭여행, 고급 여행 등) 에 따라 예상 경비가 어떻게 달라지는지 알려줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    여행 예상 경비 글씨 크기 15px 글씨 타입 bold

    """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""      
    ### 예상 경비 정보 출력 형식 ###
    {city_name} 여행 예상 경비

    • 숙박비 :
    • 식비 :
    • 교통비 :
    • 관광비 :
    • 쇼핑비 :
    • 총 예상 경비 :
    """
    return prompt


def generate_language_prompt(city_name, user_input=None):
    prompt = f"""
    {city_name}에서 사용하는 언어와 간단한 인사말, 여행에 필요한 기본적인 표현들을 알려줘. 
    만약 대한민국 여행지면 한국어로 알려줘.
    언어 정보 글씨 크기 15px 글씨 타입 bold
        """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
        사용자는 "{user_input}"라고 요청했어.
        """

    prompt += f"""      
    ### 언어 정보 출력 형식 ###
    {city_name} 언어 정보

        • 사용 언어 :
        • 인사말 :
        • 기본 표현 :
        """
    return prompt


def generate_visa_prompt(city_name, user_input=None):
    prompt = f"""
        {city_name}를 여행하려는 한국인이 비자를 발급받아야 하는지, 
        필요하다면 어떤 종류의 비자가 필요하고, 
        비자 발급 절차는 어떻게 되는지 알려줘. 
        비자 발급에 필요한 서류, 비자 발급 비용, 비자 처리 기간 등을 자세하게 알려줘.
        비자 정보 글씨 크기 15px 글씨 타입 bold
        """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
            사용자는 "{user_input}"라고 요청했어.
            """

    prompt += f"""
        ### 비자 정보 출력 형식 ###
        {city_name} 비자 정보

        • 비자 필요 여부 :
        • 비자 종류 :
        • 비자 발급 절차 :
        • 필요 서류 :
        • 비자 발급 비용 :
        • 비자 처리 기간 :
        • 추가 정보 :
        """
    return prompt


def generate_travel_insurance_prompt(city_name, user_input=None):
    prompt = f"""
        {city_name} 여행을 위한 여행자 보험에 대한 정보를 제공해줘. 
        여행자 보험의 필요성, 보험 가입 시 고려해야 할 사항, 
        추천하는 보험 상품, 보험료 비교 방법 등을 알려줘.
        여행자 보험 정보 글씨 크기 15px 글씨 타입 bold
        """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
            사용자는 "{user_input}"라고 요청했어.
            """

    prompt += f"""
        ### 여행자 보험 정보 출력 형식 ###
        {city_name} 여행자 보험 정보

        • 여행자 보험 필요성 :
        • 보험 가입 시 고려 사항 :
        • 추천 보험 상품 :
        • 보험료 비교 방법 :
        • 추가 정보 :
        """
    return prompt


def generate_currency_exchange_prompt(city_name, user_input=None):
    prompt = f"""
        {city_name}에서 사용하는 화폐와 환율 정보를 알려줘. 
        환전 방법, 환전 수수료를 절약하는 방법, 
        {city_name}에서 사용 가능한 신용카드 종류, 
        ATM 사용 방법 등을 자세하게 알려줘.
        환전 정보 글씨 크기 15px 글씨 타입 bold
        """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
            사용자는 "{user_input}"라고 요청했어.
            """

    prompt += f"""
        ### 환전 정보 출력 형식 ###
        {city_name} 환전 정보

        • 사용 화폐 :
        • 환율 정보 :
        • 환전 방법 :
        • 환전 수수료 절약 방법 :
        • 사용 가능 신용카드 :
        • ATM 사용 방법 :
        • 추가 정보 :
        """
    return prompt


def generate_communication_prompt(city_name, user_input=None):
    prompt = f"""
        {city_name}에서 사용 가능한 통신 방법에 대한 정보를 제공해줘. 
        로밍, 현지 유심, 포켓 와이파이 등의 옵션을 비교 분석하고, 
        각 옵션의 장단점, 가격, 이용 방법 등을 자세하게 알려줘.
        통신 정보 글씨 크기 15px 글씨 타입 bold
        """
    if user_input:  # 사용자 입력이 있으면 프롬프트에 추가
        prompt += f"""
            사용자는 "{user_input}"라고 요청했어.
            """

    prompt += f"""
        ### 통신 정보 출력 형식 ###
        {city_name} 통신 정보

        • 로밍 :
          - 장점 :
          - 단점 :
          - 가격 :
          - 이용 방법 :
        • 현지 유심 :
          - 장점 :
          - 단점 :
          - 가격 :
          - 이용 방법 :
        • 포켓 와이파이 :
          - 장점 :
          - 단점 :
          - 가격 :
          - 이용 방법 :
        • 추가 정보 :
        """
    return prompt


async def get_random_cities():
    """
    OpenAI API를 사용하여 나라 이름에서 도시 이름을 추출하는 함수 (비동기)
    """
    city_list = [
        # 바다 관련
        "하와이", "발리", "푸껫", "몰디브", "세이셸", "피지", "보라보라", "코사무이", "랑카위", "산토리니", "미코노스",
        "카프리", "니스", "모나코", "바르셀로나", "마르세유", "나폴리", "두브로브니크", "칸쿤", "제주도", "오키나와",
        "몰타", "사이판", "괌", "부산 태종대", "울릉도", "태국 끄라비", "베트남 하롱베이", "이탈리아 아말피 해안",
        "스페인 코스타 브라바", "호주 골드코스트", "미국 캘리포니아", "포르투갈 나자레", "브라질 리우데자네이루",
        "부산 해운대", "속초", "강릉 경포대", "제주도 협재 해수욕장", "강원도 망상 해수욕장", "태안 만리포 해수욕장",
        "포항 월포 해수욕장",
        # 산 관련
        "스위스", "캐나다 밴쿠버", "일본 홋카이도", "칠레 파타고니아", "네팔 히말라야", "알래스카", "아이슬란드",
        "노르웨이", "스웨덴", "스위스 인터라켄", "오스트리아 잘츠부르크", "미국 옐로스톤 국립공원", "뉴질랜드 퀸스타운",
        "독일 흑림", "프랑스 알프스", "캐나다 로키 산맥", "미국 요세미티 국립공원", "그랜드 캐니언 국립공원",
        "캐나다 밴프 국립공원", "호주 블루마운틴", "중국 장가계", "일본 후지산", "인도네시아 발리",
        "필리핀 마욘 화산", "이탈리아 에트나 화산", "설악산", "지리산", "내장산", "오대산", "북한산", "한라산",
        "금강산", "덕유산", "소백산", "태백산", "일본 교토", "캐나다 몬트리올",
        # 도시 관련
        "뉴욕", "홍콩", "두바이", "상하이", "도쿄", "싱가포르", "시카고", "런던", "프랑크푸르트", "파리", "로마",
        "프라하", "피렌체", "베니스", "암스테르담", "빈", "교토", "서울 북촌", "전주 한옥마을", "경주", "베이징",
        "이스탄불", "서울", "로스앤젤레스", "마드리드", "뮌헨", "도르트문트", "맨체스터", "리버풀",
        # 사막 및 특수 환경
        "아부다비", "카이로", "모로코 마라케시", "요르단 페트라", "미국 라스베이거스", "호주 울룰루", "칠레 아타카마 사막",
        "아마존", "코스타리카", "보르네오", "베트남 다낭", "태국 치앙마이", "말레이시아 쿠알라룸푸르",
        # 온천 및 유적
        "일본 유후인", "하코네", "벳푸", "쿠사츠", "노보리베츠", "대만 타이베이", "이탈리아 로마", "캄보디아 앙코르와트",
        "인도 아그라", "멕시코 치첸이트사",
        # 영화 관련
        "에든버러", "옥스포드", "미국 올랜도", "캘리포니아 애너하임", "플로리다 올랜도", "프랑스 파리", "영국",
        "뉴질랜드", "캐나다 캘거리", "모로코 탕헤르", "콘월"
    ]

    template = f"""
        {city_list} 안에 있는 도시를 랜덤으로 하나만 뽑아주고 그 이유에 대해서 설명해줘

        ✈️ YATA가 추천하는 여행지!
        \n\n
        추천 도시 :
        \n\n
        추천 해주는 이유 :
        """

    prompt = PromptTemplate(template=template, input_variables=["cityList"])
    # invoke 대신 __call__ 사용
    cities = chat.invoke(prompt.format(user_input=city_list))
    st.write(cities.content)

    # "추천 도시:"에 이어지는 텍스트를 정규식으로 추출
    match = re.search(r"추천 도시\s*:\s*(.+)", cities.content)
    if match:
        recommended_city = match.group(1).strip()
        return recommended_city
    return st.write("추천 도시가 없습니다.\n 가고 싶은 도시를 입력하세요.")


def extract_travel_time(response):
    """
    OpenAI API 응답에서 이동 시간을 추출하는 함수
    """
    try:
        # 1. "약 3시간" 형식의 답변 처리
        match = re.search(r"약 (\d+)시간", response)
        if match:
            return int(match.group(1))

        # 2. "3시간 30분" 형식의 답변 처리
        match = re.search(r"(\d+)시간 (\d+)분", response)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            return hours + minutes / 60

        # 3. "5시간 정도" 형식의 답변 처리
        match = re.search(r"(\d+)시간 정도", response)
        if match:
            return int(match.group(1))

        # 4. 숫자만 있는 답변 처리 (예: "3")
        match = re.search(r"(\d+)", response)
        if match:
            return int(match.group(1))

    except:
        pass  # 예외 발생 시 무시

    return 2  # 이동 시간 추출 실패 시 기본값 2시간 반환


def extract_country_from_user_input(user_input):
    """
    사용자 입력에서 나라 이름을 추출하는 함수
    """
    match = re.search(r"([\w\s]+)\s+\d+일", user_input)
    if match:
        return match.group(1).strip()
    return None


async def extract_total_days(user_input):
    """
    Langchain parser를 사용하여 사용자 입력에서 총 여행 일수와 도시별 여행 일수를 추출하는 함수
    """
    template = f"""
    다음 문장에서 여행할 도시 이름과 각 도시별 여행 일정을 추출해줘. 
    만약 도시 이름이 없고 나라 이름만 있다면 여행할 나라를 찾아서, 그 나라의 유명한 도시들을 최대 3개까지 알려줘. 
    도시들은 그 나라의 구글 지도를 참고하여 이동하기 편한 순서대로 알려줘.
    각 도시는 도시 이름, 숙박 일수, 여행 일수 순서로 출력하고, 
    각 도시는 쉼표로 구분해줘.
    일정은 3박 4일, 10일, 4박, 11월 이런식으로 다양하게 입력해 한글로 삼일 열흘 이렇게 입력 할 수도 있어
    이를 잘 구분해줘
    도시별로 숙박 일수를 입력 안했다면 너가 임의로 여행 일정을 지정해줘
    단순하게 여행 총 일자만 입력했어도 너가 임의로 여행 일정을 지정해줘
    단순하게 월만 입력했어도 입력했어도 너가 임의로 여행 일정을 지정해줘
    정교하게 11월 25일부터 12월 5일까지 갈거야라고 입력하면 그 기간을 너가 x박 y일로 변경해서 넣어줘
    즉, 정확하게 도시별 일정을 지정하지 않았다면 너가 마음대로 일정을 지정해주면 되는거야
    하지만 최대 14일을 넘기지 말아줘     
    만약 도시 이름이 없다면 "없음" 이라고 답해줘
    만약 여행 일정 정보가 없다면 "없음"이라고 답해줘.

    예시: 
        오사카 2박 3일, 교토 1박 2일, 도쿄 3박 4일
        서울 2박 3일 
        리스본 11월 여행
        파리 10일 여행
        파리 11월 25일부터 12월 5일까지 여행    
        10일 여행   
        없음

    문장: {user_input}
    """
    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    # invoke 대신 __call__ 사용
    response = chat.invoke(prompt.format(user_input=user_input))
    response = response.content.strip()

    if response == "없음":
        return None, None

    try:
        print(f"extract_total_days 입력: {user_input}")  # 입력값 출력
        # 도시별 여행 일정 추출
        city_days = []
        total_days = 0
        for city_info in response.split(','):
            city_info = city_info.strip()
            # 정규식 매칭 시도
            match = re.match(r"(.+)\s+(\d+)박\s+(\d+)일", city_info)
            if match:  # "x박 y일" 형식 매칭 성공 시
                city, nights, days = match.groups()
                nights = int(nights)
                days = int(days)
                city_days.append({"city": city, "nights": nights, "days": days})
                total_days += days
            else:  # "YYYY년 MM월 DD일" 형식 매칭 시도
                match = re.match(r"(\d{4}년 \d{1,2}월 \d{1,2}일)", city_info)
                if match:  # "YYYY년 MM월 DD일" 형식 매칭 성공 시
                    date_str = match.group(1)
                    date_obj = datetime.strptime(date_str, "%Y년 %m월 %d일").date()
                    # 현재 날짜와 비교하여 total_days 계산
                    today = datetime.now().date()
                    total_days = (date_obj - today).days + 1
                    # 도시 이름은 "없음"으로 설정
                    city_days.append({"city": "없음", "nights": total_days - 1, "days": total_days})
                else:  # 도시 이름만 있는 경우
                    city = city_info
                    # 기본값으로 2박 3일 설정
                    city_days.append({"city": city, "nights": 2, "days": 3})
                    total_days += 3

        print(f"extract_total_days 출력: {total_days}, {city_days}")  # 출력값 출력
        return total_days, city_days

    except Exception as e:
        logging.error(f"Error in extract_total_days: {e}")
        st.error(f"extract_total_days 함수에서 오류 발생: {e}")  # 에러 메시지 출력

    return None, None


async def analyze_image(image_path):
    """
    이미지를 분석하여 배경에 따라 도시를 추천하는 함수 (비동기)
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # st.image(image_data)
    # 이미지 데이터를 base64 인코딩
    encoded_image_data = base64.b64encode(image_data).decode('utf-8')

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 사진의 배경은 무엇인가요?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_data}"}}
                ]
            }
        ]
    )

    background = response.choices[0].message.content  # 응답에서 배경 정보 추출
    logging.debug("background")
    logging.debug(background)
    st.markdown(background)

    # 배경에 따라 도시 리스트 추천 (더욱 상세하게!)
    if "바다" in background or "해변" in background or "해안" in background:
        if "열대" in background or "야자수" in background:
            cityList = ["하와이", "발리", "푸껫", "몰디브", "세이셸", "피지", "보라보라", "코사무이", "랑카위"]  # 열대 해변
        elif "지중해" in background:
            cityList = ["산토리니", "미코노스", "카프리", "니스", "모나코", "바르셀로나", "마르세유", "나폴리", "두브로브니크"]  # 지중해 해변
        elif "에메랄드 빛" in background or "투명한" in background:
            cityList = ["몰디브", "보라보라", "세이셸", "칸쿤", "제주도", "오키나와", "몰타", "사이판", "괌"]  # 에메랄드 빛, 투명한 바다
        elif "해안 절벽" in background or "기암괴석" in background:
            cityList = ["제주도", "부산 태종대", "울릉도", "태국 끄라비", "베트남 하롱베이", "이탈리아 아말피 해안", "스페인 코스타 브라바"]  # 해안 절벽, 기암괴석
        elif "서핑" in background or "파도" in background:
            cityList = ["하와이", "발리", "호주 골드코스트", "미국 캘리포니아", "포르투갈 나자레", "브라질 리우데자네이루"]  # 서핑 명소
        else:
            cityList = ["부산 해운대", "속초", "강릉 경포대", "제주도 협재 해수욕장", "강원도 망상 해수욕장", "태안 만리포 해수욕장", "포항 월포 해수욕장"]  # 일반 해변
    elif "산" in background:
        if "눈 덮인" in background or "빙하" in background:
            cityList = ["스위스", "캐나다 밴쿠버", "일본 홋카이도", "칠레 파타고니아", "네팔 히말라야", "알래스카", "아이슬란드", "노르웨이",
                        "스웨덴"]  # 눈 덮인 산, 빙하
        elif "숲" in background or "초원" in background:
            cityList = ["캐나다 밴쿠버", "스위스 인터라켄", "오스트리아 잘츠부르크", "미국 옐로스톤 국립공원", "뉴질랜드 퀸스타운", "독일 흑림", "프랑스 알프스",
                        "캐나다 로키 산맥"]  # 숲과 초원이 있는 산
        elif "바위" in background or "협곡" in background or "절벽" in background:
            cityList = ["미국 요세미티 국립공원", "그랜드 캐니언 국립공원", "캐나다 밴프 국립공원", "호주 블루마운틴", "중국 장가계", "베트남 하롱베이",
                        "노르웨이 피오르드"]  # 바위산, 협곡, 절벽
        elif "화산" in background:
            cityList = ["일본 후지산", "인도네시아 발리", "필리핀 마욘 화산", "이탈리아 에트나 화산", "아이슬란드", "하와이", "코스타리카"]  # 화산
        elif "단풍" in background or "가을" in background:
            cityList = ["설악산", "지리산", "내장산", "오대산", "북한산", "일본 교토", "캐나다 몬트리올"]  # 단풍 명소
        else:
            cityList = ["설악산", "지리산", "북한산", "한라산", "금강산", "덕유산", "소백산", "태백산"]  # 일반 산
    elif "도시" in background or "건물" in background:
        if "고층 건물" in background or "야경" in background:
            cityList = ["뉴욕", "홍콩", "두바이", "상하이", "도쿄", "싱가포르", "시카고", "런던", "프랑크푸르트"]  # 고층 건물과 야경
        elif "유럽" in background or "역사적인" in background:
            cityList = ["파리", "런던", "로마", "프라하", "바르셀로나", "피렌체", "베니스", "암스테르담", "빈"]  # 유럽 도시
        elif "전통" in background or "고풍스러운" in background:
            cityList = ["교토", "서울 북촌", "전주 한옥마을", "경주", "베이징", "이스탄불", "프라하", "리스본"]  # 전통적인 도시
        elif "현대적인" in background or "세련된" in background:
            cityList = ["도쿄", "싱가포르", "뉴욕", "런던", "밀라노", "파리", "홍콩", "상하이"]  # 현대적인 도시
        else:
            cityList = ["서울", "도쿄", "싱가포르", "런던", "뉴욕", "파리", "베이징", "상하이", "로스앤젤레스"]  # 일반 도시
    elif "사막" in background:
        cityList = ["두바이", "아부다비", "카이로", "모로코 마라케시", "요르단 페트라", "미국 라스베이거스", "호주 울룰루", "칠레 아타카마 사막"]  # 사막 도시
    elif "정글" in background or "열대 우림" in background:
        cityList = ["아마존", "코스타리카", "보르네오", "베트남 다낭", "태국 치앙마이", "말레이시아 쿠알라룸푸르", "인도네시아 발리", "브라질 리우데자네이루"]  # 정글/열대 우림
    elif "호수" in background or "강" in background:
        cityList = ["스위스 인터라켄", "오스트리아 잘츠부르크", "이탈리아 코모", "캐나다 밴쿠버", "헝가리 부다페스트", "체코 프라하", "베트남 하노이",
                    "중국 항저우"]  # 호수/강 도시
    elif "온천" in background or "료칸" in background:
        cityList = ["일본 유후인", "하코네", "벳푸", "쿠사츠", "노보리베츠", "대만 타이베이", "헝가리 부다페스트", "아이슬란드 블루라군"]  # 온천 도시
    elif "유적" in background or "사원" in background:
        cityList = ["이집트 카이로", "그리스 아테네", "이탈리아 로마", "페루 마추픽추", "캄보디아 앙코르와트", "인도 아그라", "태국 방콕",
                    "멕시코 치첸이트사"]  # 유적/사원 도시
    elif "축구" in background or "축구 경기장" in background or "축구 선수" in background:
        cityList = ["영국 런던", "스페인 마드리드", "바르셀로나", "독일 뮌헨", "도르트문트", "이탈리아 밀라노", "토리노", "프랑스 파리", "맨체스터", "리버풀",
                    "브라질 리우데자네이루", "아르헨티나 부에노스아이레스"]  # 축구 도시
    elif "야구" in background or "야구장" in background or "야구 선수" in background:
        cityList = ["미국 뉴욕", "로스앤젤레스", "시카고", "보스턴", "샌프란시스코", "일본 도쿄", "오사카", "한국 서울", "부산", "대구", "광주",
                    "도미니카 공화국"]  # 야구 도시
    elif "농구" in background or "농구 코트" in background or "농구 선수" in background:
        cityList = ["미국 뉴욕", "로스앤젤레스", "시카고", "보스턴", "샌프란시스코", "캐나다 토론토", "스페인 마드리드", "바르셀로나", "그리스 아테네", "터키 이스탄불",
                    "세르비아 베오그라드", "리투아니아 카우나스"]  # 농구 도시
    elif "스키" in background or "스키장" in background or "스키 선수" in background:
        cityList = ["스위스", "오스트리아", "프랑스", "캐나다", "미국", "일본", "한국", "이탈리아", "독일", "노르웨이", "스웨덴", "핀란드"]  # 스키 도시
    elif "아이언맨" in background or "마블" in background or "어벤져스" in background:
        cityList = ["뉴욕", "로스앤젤레스", "샌프란시스코", "워싱턴 D.C.", "시카고", "애틀랜타"]  # 마블 관련 도시
    elif "해리포터" in background or "호그와트" in background:
        cityList = ["런던", "에든버러", "글래스고", "옥스포드"]  # 해리포터 관련 도시
    elif "디즈니" in background or "디즈니랜드" in background:
        cityList = ["올랜도", "캘리포니아 애너하임", "플로리다 올랜도", "일본 도쿄", "프랑스 파리", "홍콩", "상하이"]  # 디즈니랜드 도시
    elif "유니버셜 스튜디오" in background or "해리포터 테마파크" in background:
        cityList = ["올랜도", "캘리포니아 할리우드", "일본 오사카", "싱가포르", "베이징"]  # 유니버셜 스튜디오 도시
    elif "스타워즈" in background:
        cityList = ["미국 캘리포니아", "플로리다", "튀니지", "아일랜드", "이탈리아", "영국"]  # 스타워즈 촬영지
    elif "반지의 제왕" in background or "호빗" in background:
        cityList = ["뉴질랜드"]  # 반지의 제왕/호빗 촬영지
    elif "타이타닉" in background:
        cityList = ["미국", "캐나다", "멕시코"]  # 타이타닉 관련 도시 (촬영지, 박물관)
    elif "라라랜드" in background:
        cityList = ["로스앤젤레스"]  # 라라랜드 촬영지
    elif "비긴 어게인" in background:
        cityList = ["뉴욕"]  # 비긴 어게인 촬영지
    elif "맘마미아" in background:
        cityList = ["그리스"]  # 맘마미아 촬영지
    elif "미션 임파서블" in background:
        cityList = ["프라하", "미국", "영국", "프랑스", "아랍에미리트"]  # 미션 임파서블 촬영지
    elif "007" in background or "제임스 본드" in background:
        cityList = ["영국 런던", "이탈리아", "프랑스", "스위스", "일본", "미국"]  # 007 촬영지
    elif "인셉션" in background:
        cityList = ["프랑스 파리", "일본 도쿄", "미국 로스앤젤레스", "캐나다 캘거리", "모로코 탕헤르"]  # 인셉션 촬영지
    elif "어바웃 타임" in background:
        cityList = ["영국 런던", "콘월"]  # 어바웃 타임 촬영지
    elif "악마는 프라다를 입는다" in background:
        cityList = ["미국 뉴욕", "파리"]  # 악마는 프라다를 입는다 촬영지
    else:
        cityList = ["런던", "파리", "뉴욕", "도쿄", "바르셀로나", "로마", "암스테르담", "베를린", "시드니"]  # 기본 도시

    return random.choice(cityList)  # 리스트에서 랜덤으로 도시 선택


async def generate_travel_plan(user_input, image_path=None):
    """
    여행 계획을 도시별로 나눠서 생성하는 함수 (비동기)
    """
    total_days, city_days = await extract_total_days(user_input)  # 도시별 여행 일수 추출
    in_flag = False

    # total_days가 None인 경우 처리
    if total_days is None:
        in_flag = True
        # 1. 도시 이름 추출
        city_names = await extract_city_name(user_input)  # 도시 이름 추출

        # 2. 도시 이름이 없으면 나라 이름에서 도시 이름 추출
        if city_names == "없음":
            city_names = await get_cities_from_country(user_input)
            if city_names == "없음":
                if image_path is None:
                    city_names = await get_random_cities()

        # 3. 도시 수 계산
        cities = [city.strip() for city in city_names.split(',')]
        num_cities = len(cities)

        # 4. 도시 수에 따라 total_days 및 city_days 설정
        if num_cities == 1:
            total_days = 5  # 도시 1개: 5일
            city_days = [{"city": cities[0], "nights": 4, "days": 5}]  # 도시 정보 추가
        elif num_cities == 2:
            total_days = 8  # 도시 2개: 8일 (3박 4일 + 3박 4일)
            city_days = [{"city": cities[0], "nights": 3, "days": 4},
                         {"city": cities[1], "nights": 3, "days": 4}]  # 도시 정보 추가
        elif num_cities == 3:
            total_days = 11  # 도시 3개: 11일
            city_days = [{"city": cities[0], "nights": 2, "days": 3},
                         {"city": cities[1], "nights": 3, "days": 4},
                         {"city": cities[2], "nights": 3, "days": 4}]  # 도시 정보 추가
        else:  # 도시 4개 이상: 도시당 2박 3일
            total_days = num_cities * 3
            city_days = [{"city": city, "nights": 2, "days": 3} for city in cities]  # 도시 정보 추가

        st.warning(f"총 여행 일수가 명시되지 않았습니다. 도시 수 ({num_cities}개)에 따라 {total_days}일을 사용합니다.")

    # city_days가 비어 있는 경우 처리
    if not city_days:
        in_flag = True
        # 도시 이름을 입력한 경우
        city_names = await extract_city_name(user_input)  # 도시 이름 추출
        if city_names != "없음":  # 도시 이름이 있는 경우
            cities = [city.strip() for city in city_names.split(',')]
            city_days = [{"city": city, "nights": total_days - 1, "days": total_days} for city in cities]
        else:  # 도시 이름이 없는 경우 (나라 이름 입력)
            country = extract_country_from_user_input(user_input)  # 나라 이름 추출
            if country:
                # OpenAI API를 사용하여 도시 정보 추출 (도시 2~3개 추천)
                num_cities = 3
                city_names = await get_cities_from_country(user_input)
                cities = [city.strip() for city in city_names.split(',')]

                # 10일을 도시별로 적절히 분배
                days_per_city = total_days // num_cities
                remaining_days = total_days % num_cities
                city_days = [{"city": city, "nights": days_per_city - 1, "days": days_per_city} for city in cities]
                for i in range(remaining_days):
                    city_days[i]["dys"] += 1
                    city_days[i]["nights"] += 1
            else:
                st.error("여행 계획을 이해할 수 없습니다. 도시 또는 나라 이름을 포함하여 입력해주세요.")
            return

    # 이미지 분석을 통해 도시 이름 추출 (image_path가 제공된 경우)
    if image_path:
        city_names = await analyze_image(image_path)
        st.markdown("추천 여행지 : " + city_names)
        logging.debug("image_path")
        logging.debug(city_names)
    else:
        if in_flag is False:
            city_names = await extract_city_name(user_input)  # 도시 이름 추출
            # 도시 이름이 없으면 나라 이름에서 도시 이름 추출
            if city_names == "없음":
                city_names = await get_cities_from_country(user_input)
                if city_names == "없음":
                    city_names = await get_random_cities()

    cities = [city.strip() for city in city_names.split(',')]  # 도시 이름을 쉼표로 분리하여 리스트 생성
    logging.debug(cities)

    # 여행 기간이 명시되지 않은 경우
    num_cities = len(cities)
    if total_days is None:
        if num_cities == 1:
            total_days = 5  # 도시 1개: 5일
            city_days = [5]
        elif num_cities == 2:
            total_days = 8  # 도시 2개: 8일 (3박 4일 + 3박 4일)
            city_days = [4, 4]
        elif num_cities == 3:
            total_days = 11  # 도시 3개: 11일
            city_days = [3, 4, 4]  # 각 도시에서 2박 3일, 3박 4일, 4박 5일
        else:  # 도시 4개 이상: 도시당 2박 3일
            total_days = num_cities * 3
            city_days = [3] * num_cities

    # 도시별 여행 일수 계산 (city_days가 None이거나 cities와 길이가 다를 경우)
    if num_cities != len(city_days):
        days_per_city = total_days // num_cities  # 도시별 여행 일수 계산
        remaining_days = total_days % num_cities  # 남은 일수 계산
        city_days = [days_per_city] * num_cities  # 기본 도시별 일수 설정
        for i in range(remaining_days):
            city_days[i] += 1  # 남은 일수를 앞 도시부터 하루씩 추가

    global travel_plan_text
    start_day = 1

    global city_info_text

    city_start_day = 1  # 도시별 시작 일차

    tasks = []  # 비동기 task를 저장할 리스트
    departure_time = None  # departure_time 초기값 설정

    # 출발일과 도착일 정보 추출
    start_end_date_str = await extract_start_end_date(user_input)

    if start_end_date_str == "없음":
        # 월 정보 추출
        month = extract_month(user_input)
        if month:
            # 여행 기간 질문
            total_days = st.number_input("여행 기간은 며칠 정도 생각하고 계신가요?", min_value=1, step=1, key="total_days")

            # 나라별 평균 여행 기간 정보 활용 (예시)

            st.write("평균적으로 7-10일 정도 여행하는 것을 추천합니다.")

            # 도시 개수에 따른 여행 기간 설정 (예시)
            cities = await extract_city_name(user_input)
            if cities != "없음":
                num_cities = len(cities.split(','))
                if num_cities == 2:
                    st.write("두 도시를 여행하는 경우, 7일 정도가 적당합니다.")
                elif num_cities >= 3:
                    st.write("세 도시 이상을 여행하는 경우, 10일 이상을 추천합니다.")

            # 기본값 설정 및 안내
            if not total_days:
                total_days = 7  # 기본 여행 기간
                st.warning(f"여행 기간이 명시되지 않았습니다. 기본값으로 {total_days}일을 사용합니다.")

            start_date, end_date = await generate_random_dates_in_month(datetime.now().year, month, total_days)
            st.warning(f"출발일과 도착일이 명시되지 않았습니다. {month}월 내의 임의의 날짜를 사용합니다.")
            logging.debug("dd1")
    else:
        try:
            logging.debug("start_end_date_str")
            logging.debug(start_end_date_str)
            logging.debug(city_names)
            # 1. 도시 이름 추출
            if city_names is None:
                city_names = await extract_city_name(user_input)  # 도시 이름 추출
                logging.debug(city_names)

            # 2. 도시 이름이 없으면 나라 이름에서 도시 이름 추출
            if city_names == "없음":
                city_names = await get_cities_from_country(user_input)
                logging.debug(city_names)
                if city_names == "없음":
                    if image_path is None:
                        city_names = await get_random_cities()

            # 3. 도시 수 계산
            cities = [city.strip() for city in city_names.split(',')]
            logging.debug("cities")
            logging.debug(cities)

            num_cities = len(cities)
            logging.debug(num_cities)

            # 출발일과 도착일 추출
            start_date_str = start_end_date_str.split(", ")[0].split(": ")[1]
            end_date_str = start_end_date_str.split(", ")[1].split(": ")[1]

            # 현재 연도 가져오기
            current_year = datetime.now().year

            # 날짜 변환 (연도 명시)
            start_date = datetime.strptime(f"{current_year}년 {start_date_str}", "%Y년 %m월 %d일").date()
            end_date = datetime.strptime(f"{current_year}년 {end_date_str}", "%Y년 %m월 %d일").date()

            # 만약 12월이 11월보다 먼저 나온다면, 연도가 다르게 설정되었을 가능성이 있음
            if end_date < start_date:
                end_date = datetime.strptime(f"{current_year + 1}년 {end_date_str}", "%Y년 %m월 %d일").date()

            # 여행 총 일수 계산
            total_days = (end_date - start_date).days + 1

            city_days = []
            days_per_city = total_days // num_cities
            remaining_days = total_days % num_cities
            city_days = [{"city": city, "nights": days_per_city - 1, "days": days_per_city} for city in cities]
            for i in range(remaining_days):
                city_days[i]["days"] += 1
                city_days[i]["nights"] += 1

                # 디버깅 출력
            print(f"출발일: {start_date}, 도착일: {end_date}")
            print(f"총 여행 기간: {total_days}일")
            logging.debug("city_days")
            logging.debug(city_days)
        except ValueError:
            # 출발일 또는 도착일 정보가 없는 경우
            month = extract_month(user_input)
            if month:
                start_date, end_date = await generate_random_dates_in_month(datetime.now().year, month, total_days)
                st.warning(f"출발일과 도착일이 명확하지 않습니다. {month}월 내의 임의의 날짜를 사용합니다.")
                logging.debug("dd2")
            else:
                # 기본값 사용
                start_date, end_date = await get_default_dates()
                st.warning("출발일과 도착일이 명확하지 않습니다. 기본 날짜를 사용합니다.")


    for i, city_name in enumerate(cities):
        try:  # try-except 블록 추가
            logging.debug(f"city_days: {city_days}, i: {i}")
            print(f"city_days: {city_days}, i: {i}")  # city_days 값과 i 값 출력
            days = city_days[i]['days']  # 수정된 부분: city_days에서 도시별 여행 일수 가져오기
        except IndexError:
            st.error(f"city_days 인덱스 오류: i={i}, city_days={city_days}")  # 에러 메시지 출력
            return

        # 도시 순서 설정
        city_order = i + 1  # 현재 도시의 순서 (1부터 시작)

        # 다음 도시 이름 설정
        next_city = cities[i + 1] if i < len(cities) - 1 else None

        # 도시별 시작일 계산 (출발일 기준)
        city_start_date = start_date + timedelta(days=start_day - 1)

        task = asyncio.create_task(generate_city_plan(
            user_input, city_name, start_day, days, city_start_day, city_start_date, city_order, num_cities, next_city
        ))
        tasks.append(task)

        # 다음 도시 시작일 계산
        city_start_day = start_day  # 다음 도시 시작 일차 업데이트

        # start_day 업데이트 (출발 시간 고려)
        if city_order < len(cities) and departure_time is not None:  # departure_time 확인
            travel_time_prompt = f"{city_name}에서 {cities[i + 1]}(으)로 이동하는 데 걸리는 시간은?"
            travel_time_response = await get_openai_response(travel_time_prompt)
            travel_time = extract_travel_time(travel_time_response)

            departure_hour = int(departure_time[:2])  # 출발 시간 추출
            next_city_arrival_hour = (departure_hour + travel_time) % 24

            # if next_city_arrival_hour < departure_hour:  # 00시 넘으면 다음 날
            #    start_day += 1

        start_day += days  # 도시에서 보낸 일수 더하기

    # 모든 task가 완료될 때까지 기다림
    results = await asyncio.gather(*tasks)

    # 결과를 하나의 문자열로 합침
    for city_plan_text, city_info_text_part, departure_time in results:  # departure_time 받아오기
        travel_plan_text += city_plan_text + "\n\n"
        city_info_text += city_info_text_part + "\n\n"

async def get_default_dates():
  """기본 출발일과 도착일을 반환하는 함수 (현재 월 기준)"""
  today = datetime.now().date()
  start_date = today + timedelta(days=7)  # 일주일 뒤를 출발일로 설정
  end_date = start_date + timedelta(days=7)  # 기본 여행 기간: 7일
  return start_date, end_date

async def display_tips():
    """5초마다 꿀팁을 표시하는 비동기 함수"""
    if "show_tips" not in st.session_state:
        st.session_state.show_tips = False

    tip_placeholder = st.empty()  # 꿀팁을 표시할 공간

    while st.session_state.show_tips:
        # 랜덤 도시와 꿀팁 선택
        random_city = random.choice(list(city_secrets.city_secrets.keys()))
        secrets = city_secrets.city_secrets[random_city]
        secret = random.choice(secrets)

        # 꿀팁 표시
        tip_placeholder.markdown(
            f"️✈️YATA의 여행지 이모저모!\n\n**도시**: {random_city}\n\n**이모저모**: {secret}"
        )

        await asyncio.sleep(5)  # 5초 대기


# 여행 계획 (사진 입력)
# image_path = "/Users/nabakgood/Desktop/sparta_python/test_image.jpg"  # 실제 이미지 경로로 변경
# 이미지 열기
# image = Image.open(image_path)
# 이미지 크기 조정 (가로 500px, 세로 비율 유지)
# image.thumbnail((500, 500))
# 이미지 표시
# display(image)

# 비동기 함수 실행
# async def main():
#    travel_plan_text = await generate_travel_plan(user_input="내가 이미지 하나 올릴테니 요런 느낌의 여행지를 추천해주고 여행 계획을 짜줘", image_path=image_path)
#
# 전체 여행 계획 출력 (선택 사항)
#    print(travel_plan_text)
#    print(city_info_text)

# asyncio.run(main())

# 꿀팁 표시 함수 (비동기)
async def generate_city_plan(user_input, city_name, start_day, days, city_start_day, city_start_date, city_order, num_cities,
                             next_city=None):
    """
    Langchain LLMChain을 사용하여 여행 계획 및 도시 정보를 생성하는 함수
    """
    global flag_start_plan
    # 도시별 여행 계획 생성 및 출력

    retrieved_information = search_tool.run(f"{city_name} 여행 정보")

    # 도시 선택
    travel_plan_prompt, departure_time, arrival_time, start_day = await generate_travel_plan_prompt(
        user_input, city_name, start_day, days, city_start_day, city_start_date, city_order, num_cities, next_city
    )  # city_order 추가

    travel_plan_schema = ResponseSchema(name="travel_plan",
                                        description=f"다음 정보를 참고하여 {city_name}에서 {start_day}일차부터 {start_day + days - 1}일차까지 {days}일 동안의 여행 계획을 짜줘:")
    travel_plan_parser = StructuredOutputParser.from_response_schemas([travel_plan_schema])

    travel_plan_chain = PromptTemplate(
        template=travel_plan_prompt,
        input_variables=[
            "city_name", "retrieved_information", "start_day", "days", "city_start_day",
            "city_order", "num_cities", "next_city", "departure_time", "arrival_time"
        ],
        partial_variables={"format_instructions": travel_plan_parser.get_format_instructions()}
    )

    city_info_prompt = await generate_city_info_prompt(city_name)
    city_info_schema = ResponseSchema(name="city_info",
                                      description=f"{city_name}의 여행 관련 정보를 생성해 주세요.")
    city_info_parser = StructuredOutputParser.from_response_schemas([city_info_schema])

    city_info_chain = PromptTemplate(
        template=city_info_prompt,
        input_variables=["city_name"],
        partial_variables={"format_instructions": city_info_parser.get_format_instructions()}
    )

    # 여행 계획 생성
    city_plan_text = chat.invoke(travel_plan_chain.format(
        city_name=city_name,
        retrieved_information=retrieved_information,
        start_day=start_day,
        days=days,
        city_start_day=city_start_day,
        city_order=city_order,
        num_cities=num_cities,
        next_city=next_city,
        departure_time=departure_time,
        arrival_time=arrival_time
    ))
    flag_start_plan = True
    logging.debug(city_plan_text)
    logging.debug(city_plan_text.content)

    # travel_plan_parsed_output = travel_plan_parser.parse(city_plan_text.content)
    logging.debug("travel_plan_parsed_output")
    # logging.debug(travel_plan_parsed_output)

    st.markdown(city_plan_text.content)

    # 도시별 정보 추출 및 출력
    city_info_text_part = chat.invoke(city_info_chain.format(city_name=city_name))
    # city_info_parsed_output = city_info_parser.parse(city_info_text_part.content)

    return city_plan_text.content, city_info_text_part.content, departure_time  # departure_time 추가

async def extract_start_end_date_prompt(user_input):
  template = f"""
  다음 문장에서 여행의 출발일과 도착일 정보를 추출해줘. 
  출발일과 도착일은 각각 "MM월 DD일" 형식으로 표현해야 해. 
  만약 출발일이나 도착일 정보가 없으면 "없음"이라고 답해줘.

  예시:
    11월 28일 출발, 12월 05일 도착

  문장: {user_input}
  """
  return template

async def extract_start_end_date(user_input):
  prompt = PromptTemplate(template=await extract_start_end_date_prompt(user_input), input_variables=["user_input"])
  response = chat.invoke(prompt.format(user_input=user_input))
  return response.content

def extract_month(user_input):
    """
    사용자 입력에서 월을 추출하는 함수
    """
    try:
        # "x월" 형식의 답변 처리
        match = re.search(r"(\d+)월", user_input)
        if match:
            month = int(match.group(1))
            return month
    except:
        pass  # 예외 발생 시 무시

    return None  # 추출 실패 시 None 반환

async def generate_random_dates_in_month(year, month, total_days):
  """해당 월의 임의의 출발일과 도착일을 생성하는 함수"""
  last_day = calendar.monthrange(year, month)[1]  # 해당 월의 마지막 날짜
  start_day = random.randint(1, last_day - total_days + 1)  # 여행 기간을 고려하여 출발일 생성
  end_day = start_day + total_days - 1

  start_date = datetime(year, month, start_day).date()
  end_date = datetime(year, month, end_day).date()

  return start_date, end_date

# 유사도 계산 함수 최적화
def calculate_similarity(str1, str2):
    """
    두 문자열의 유사도를 계산하는 함수 (FuzzyWuzzy 개선)
    """
    return fuzz.ratio(str1, str2)


def find_matching_keyword(user_input, keywords):
    """
    입력 문자열과 가장 유사한 키워드를 찾는 함수
    """
    max_similarity = 0
    matched_keyword = None
    for keyword, synonyms in keywords.items():
        for synonym in synonyms:
            similarity = calculate_similarity(user_input, synonym)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_keyword = keyword
    return matched_keyword


async def main():
    st.title("✈️Your AI Travel Agent - YATA!")
    st.write("여행 계획을 입력하세요. 예시:")
    st.write("- 2박 3일 오사카 여행 계획")
    st.write("- 파리의 맛집 추천해줘")
    st.write("- 가족여행으로 제주도 3박 4일 계획")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_area("여행 계획 입력:")
    uploaded_file = st.file_uploader("이미지 업로드 (선택 사항)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file)
    if st.button("계획 생성"):
        st.empty()
        # 이전 대화 내용 표시 (각 대화마다 st.expander 생성)
        for i, message in enumerate(st.session_state.messages):
            with st.expander(f"이전 대화 {i + 1} - {message['user']}"):
                st.write(f"사용자 : {message['user']}")
                for bot_message in message['bot']:
                    st.write(f"✈️YATA : {bot_message}")
        if user_input:
            with st.spinner("✈️YATA가 여행 계획을 생성 중입니다...\n\n최대 2분 정도 소요될 수 있습니다."):

                # 꿀팁을 표시할 빈 공간 생성
                tip_placeholder = st.empty()

                st.session_state.show_tips = True  # 꿀팁 표시 시작

                # 꿀팁을 표시하는 비동기 함수 실행
                display_task = asyncio.create_task(display_tips())
                try:
                    # 키워드 매칭 (유사 키워드, 오타 허용, 영어 키워드 포함)
                    keywords = {
                        "여행 계획": ["여행 계획", "여행 일정", "여행 코스", "여행 스케줄", "여행 계획서", "trip plan", "itinerary",
                                  "travel schedule"],
                        "숙소": ["숙소", "호텔", "모텔", "게스트하우스", "호스텔", "민박", "펜션", "리조트", "에어비앤비",
                               "accommodation", "hotel", "motel", "guesthouse", "hostel", "pension",
                               "resort", "airbnb",
                               "묵을 곳", "잠잘 곳", "머물 곳", "stay", "place to stay", "lodging"],
                        "식당": ["식당", "맛집", "레스토랑", "음식점", "카페", "바", "펍", "술집", "밥집", "브런치", "디저트",
                               "restaurant", "food", "cafe", "bar", "pub", "brunch", "dessert",
                               "먹을 곳", "맛있는 곳", "place to eat", "delicious food", "culinary"],
                        "준비": ["준비", "준비물", "챙겨야 할 것", "필요한 것", "준비 사항", "체크리스트",
                               "preparation", "things to prepare", "checklist", "what to pack",
                               "what to bring", "before travel", "pre-trip"],
                        "교통": ["교통", "이동", "이동 수단", "교통편", "대중교통", "택시", "렌터카", "자전거", "도보",
                               "transportation", "transport", "travel", "public transport", "taxi",
                               "rental car", "bike", "walk",
                               "가는 방법", "how to get to", "how to get around"],
                        "옷차림": ["옷차림", "옷", "의상", "복장", "패션",
                                "clothes", "clothing", "outfit", "fashion", "dress code",
                                "입을 옷", "what to wear"],
                        "날씨": ["날씨", "기온", "날씨 예보", "일기 예보", "기후",
                               "weather", "temperature", "forecast", "climate",
                               "현재 날씨", "current weather", "오늘 날씨", "today's weather"],
                        "관광 명소": ["관광 명소", "관광지", "명소", "유적지", "랜드마크", "볼거리", "관광 스팟",
                                  "tourist attraction", "attractions", "landmark", "sightseeing spot",
                                  "place of interest",
                                  "가볼 만한 곳", "place to visit", "must see"],
                        "쇼핑": ["쇼핑", "쇼핑몰", "쇼핑센터", "쇼핑 리스트", "기념품", "선물", "특산품",
                               "shopping", "mall", "shopping center", "shopping list", "souvenir", "gift",
                               "local specialty",
                               "살 만한 것", "things to buy"],
                        "엔터테인먼트": ["엔터테인먼트", "즐길 거리", "놀 거리", "문화 생활", "공연", "전시", "축제", "파티", "클럽",
                                   "entertainment", "fun", "cultural activities", "performance",
                                   "exhibition", "festival", "party", "club",
                                   "재미있는 것", "interesting things"],
                        "경비": ["경비", "예산", "비용", "지출", "가격",
                               "budget", "cost", "expenses", "price", "travel cost",
                               "얼마", "how much"],
                        "언어": ["언어", "말", "사투리", "표현", "인사",
                               "language", "dialect", "expression", "greeting",
                               "현지어", "local language"],
                        "비자": ["비자", "비자 발급", "비자 신청", "비자 종류",
                               "visa", "visa application", "visa types",
                               "입국", "entry"],
                        "보험": ["travel insurance", "insurance",
                               "보험 가입", "insurance coverage"],
                        "환전": ["환전", "환율", "환전소", "은행", "ATM", "카드",
                               "currency exchange", "exchange rate", "money exchange", "bank", "ATM",
                               "card",
                               "돈", "money"],
                        "통신": ["통신", "로밍", "유심", "포켓 와이파이", "데이터", "전화", "인터넷",
                               "communication", "roaming", "sim card", "pocket wifi", "data", "call",
                               "internet",
                               "와이파이", "wifi"],
                        "미슐랭": ["미슐랭", "미슐랭 가이드", "미슐랭 스타", "미슐랭 레스토랑",
                                "michelin", "michelin guide", "michelin star", "michelin restaurant",
                                "별점", "star rating"]
                    }

                    matched_keyword = find_matching_keyword(user_input, keywords)  # 변경
                    logging.debug("matched_keyword")
                    logging.debug(matched_keyword)

                    if matched_keyword == "숙소":
                        # 사용자 입력 분석 및 프롬프트 함수 선택
                        city_names = await extract_city_name(user_input)
                        logging.debug("숙소")
                        logging.debug(city_names)
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = await generate_accommodation_prompt(city_name)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "식당":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            logging.debug("여기까진 왔네???")
                            prompt = await generate_restaurant_prompt(city_name)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "미슐랭":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = await generate_restaurant_prompt(city_name)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "해야 하는 것":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_must_do_prompt(city_name)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "준비":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_preparation_prompt(city_name)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "교통":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_transportation_prompt(city_name)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "옷차림" or matched_keyword == "날씨":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            month = extract_month(user_input)  # 월 추출 함수 필요
                            if month:
                                prompt = generate_seasonal_info_prompt(city_name, month)
                                st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                                st.session_state.messages.append(
                                    {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                            else:
                                st.write("날씨를 물어 볼 때는 정확히 언제 갈지도 입력해 주세요!\n11월에 갈거야 이런식으로 입력해주시면 됩니다.")  # 오류 메시지 변경
                    elif matched_keyword == "관광 명소":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_attractions_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(city_name = city_name, user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(city_name = city_name, user_input=user_input)).content})
                    elif matched_keyword == "쇼핑":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_shopping_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "엔터테인먼트":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_entertainment_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "날씨":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_weather_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "경비" or matched_keyword == "비용":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_budget_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "언어":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_language_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "비자":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_visa_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "보험":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_travel_insurance_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "환전":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_currency_exchange_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    elif matched_keyword == "통신" or matched_keyword == "비용":
                        city_names_str = await extract_city_name(user_input)
                        if not city_names_str or city_names_str == "없음":
                            st.write("여행할 도시를 찾을 수 없습니다.\n\n 오타가 없는지 확인해주세요!\n\n여행할 도시를 꼭 입력해 주세요!")
                            return
                        city_names = city_names_str.split(',')  # 쉼표로 구분된 문자열을 리스트로 변환
                        for city_name in city_names:
                            prompt = generate_communication_prompt(city_name, user_input)
                            st.write(chat.invoke(prompt.format(user_input=user_input)).content)
                            st.session_state.messages.append(
                                {"user": user_input, "bot": chat.invoke(prompt.format(user_input=user_input)).content})
                    else:  # 여행 계획 생성
                        if uploaded_file is not None:
                            image = Image.open(uploaded_file)
                            # 이미지를 RGB 모드로 변환
                            image = image.convert("RGB")
                            image_path = "temp.jpg"
                            image.save(image_path)
                            await generate_travel_plan(user_input, image_path=image_path)
                        else:
                            await generate_travel_plan(use_input)
                        st.write(city_info_text)
                        st.session_state.messages.append({"user": user_input, "bot": [travel_plan_text, city_info_text]})


                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")

                finally:
                    st.session_state.show_tips = False  # 꿀팁 종료
                    await display_task  # 꿀팁 표시가 끝날 때까지 대기

        else:
            st.warning("여행 계획을 입력해주세요.")

# 비동기 이벤트 루프 실행
asyncio.run(main())
