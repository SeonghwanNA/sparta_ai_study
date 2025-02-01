텍스트, 이미지 기반 맞춤형 여행 계획 생성
(첨부한 영상은 오래걸리지만 프롬프트 정리와 호출 빈도를 줄여 개선했습니다. 좀더 개선하여 차주에 보여드리겠습니다.)

1. 프로젝트 개요
목표: 사용자의 텍스트 및 이미지 입력을 분석하여 개인 맞춤형 여행 계획 및 도시 정보를 제공합니다.

핵심 기능:
텍스트 분석: 여행할 도시, 기간, 스타일 등을 추출하여 여행 계획 생성에 활용합니다.
이미지 분석: 이미지 배경을 분석하여 여행지 및 테마를 추천합니다.
여행 계획 생성: OpenAI GPT-4o-mini 모델을 사용하여 상세한 여행 일정을 생성합니다.
도시 정보 제공: 추천 식당, 관광 명소, 숙소 등의 정보를 제공합니다.

2. 시스템 구성
입력:

텍스트: 여행할 나라, 도시, 기간, 스타일 등 (예: "프랑스 파리 5일 자유 여행")
이미지: 여행지 사진 (선택 사항)

처리:
텍스트 입력 처리
extract_city_name(): 도시 이름 추출
extract_total_days(): 여행 기간 추출
generate_travel_plan_prompt(): 여행 계획 생성 프롬프트 생성

이미지 입력 처리
analyze_image(): 이미지 배경 분석 및 도시 추천
generate_travel_plan_prompt(): 여행 계획 생성 프롬프트 생성

여행 계획 및 도시 정보 생성
generate_travel_plan_prompt(): OpenAI GPT-4o-mini 모델에 입력하여 여행 계획 생성
generate_city_info_prompt(): OpenAI GPT-4o-mini 모델에 입력하여 도시 정보 생성

출력:
여행 계획: 상세 일정 (텍스트 형태)
도시 정보: 추천 식당, 관광 명소, 숙소 등

3. 핵심 기술
OpenAI GPT-4o-mini: 여행 계획 및 도시 정보 생성
LangChain: 프롬프트 생성 및 관리, 외부 API 연동
RAG - DuckDuckGo Search API: 여행 정보 검색 (추후 Google Maps API, Google Search API 등으로 확장)

4. 개선 방향
속도 개선: 비동기 처리, 캐싱 전략 도입(도입했다가 뺌. 여행 계획 정보는 항상 다르게 나와야함)
프롬프트 엔지니어링: 프롬프트 템플릿 개선, Few-shot learning 적용
기능 확장: 다국어 지원, 여행 테마 추천, 챗봇 기능 등
UI/UX 개선: 지도 연동, 이미지 갤러리, 사용자 프로필 등

5. 기대 효과
사용자 맞춤형 여행 계획 제공
여행 계획 수립 시간 단축
여행 정보 접근성 향상
새로운 여행 경험 제공

6. 참고 
영상 용량이 너무커 구글 드라이브로 대체합니다.
https://drive.google.com/file/d/1UD2h6nPKuB2tc9yVA0fNQF_gQDmBk-Rz/view?usp=sharing
