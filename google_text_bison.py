from google.colab import drive
drive.mount('/content/drive')
import json
with open("pathjson", "r") as f:
   data = json.load(f)
import vertexai
from vertexai.language_models import TextGenerationModel
from google.colab import auth as google_auth
google_auth.authenticate_user()

all_courses = list(set([
    "컴파일러", "컴퓨터구조", "컴퓨팅사고와 SW코딩", "알고리즘1", "전공탐색", "컴퓨팅사고와 SW코딩",
    "데이타베이스", "SW공학및테스팅", "소프트웨어공학", "소프트웨어설계", "운영체제", "소프트웨어와 문제해결",
    "프로그래밍언어론", "기계학습개론", "컴퓨터교재연구및지도법", "컴퓨터 논리 및 논술지도", "이산수학",
    "컴퓨터학개론", "종합설계프로젝트1", "종합설계프로젝트2", "SW실용영어", "인공지능", "데이터과학기초",
    "이산수학", "선형대수", "프로그래밍기초", "인간과 컴퓨터 상호작용 설계", "고급문제해결",
    "모바일앱프로그래밍2", "모바일앱프로그래밍1", "SW와 문제해결 기초", "시스템프로그래밍", "기초프로그래밍",
    "자바프로그래밍", "자료구조응용", "자료구조", "프로그래밍기초", "프로그래밍기초", "기초프로그래밍",
    "자료구조응용", "자료구조", "디지털미디어아트", "스타트업설계", "소프트웨어융합특강1", "IT기술경영개론",
    "SW프로젝트관리기법", "SW융합커뮤니케이션", "문화 기술 개론", "소프트웨어 특강", "기초창의공학설계",
    "공학프로젝트매니지먼트", "IT지식재산권", "소프트웨어 특강", "기초수학1", "수학 II", "기초수학", "수학 I",
    "컴파일러", "자료구조", "소프트웨어융합프로젝트", "빅데이터 기초 실습", "자바프로그래밍",
    "자료구조프로그래밍", "SW 사고기법", "소프트웨어설계", "기초창의공학설계", "컴퓨터학개론", "프로그래밍기초",
    "컴퓨팅사고와 SW코딩", "알고리즘2", "모바일앱프로그래밍1", "알고리즘실습", "자바프로그래밍",
    "자료구조프로그래밍", "자료구조응용", "자료구조", "운영체제", "오픈소스프로그래밍", "창의융합설계", "SW진로설계",
    "소셜네트워크", "소셜미디어 활용전략", "기초창의공학설계", "화법교육론", "실용화법", "운영체제", "컴퓨터구조",
    "디지털설계및실험", "자료구조응용", "자료구조", "네트워크프로그래밍", "데이타통신", "종합설계프로젝트2",
    "종합설계프로젝트1", "확률및통계", "컴퓨터윤리", "시스템프로그래밍", "고급문제해결", "알고리즘1",
    "소프트웨어공학", "프로그래밍기초", "소프트웨어설계", "기초프로그래밍실습", "기초프로그래밍", "소프트웨어 특강",
    "시스템프로그래밍", "논리회로", "운영체제"]))

all_profs = ['김정근', '권영우', '남우정', '김승호', '아난드 폴',
             '정기숙', '임경식', '정원일', '정선미', '이호경', '배준현',
             '이상윤', '박소은', '김용태', '백호기', '남덕윤', '이우진', '김경훈',
             '장재석', '이시형', '김진욱', '김재수', '김필영', '김명석', '백낙훈',
             '박상효', '펑리메이', '김재일', '김명옥', '정창수', '이용주', '정설영',
             '이종택', '서영균', '이성희', '김동선', '김령환', '김동균', '김구진']

vertexai.init(project="glossy-ally-399906", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 10
}
model = TextGenerationModel.from_pretrained("text-bison")

# user_question 리스트
user_question = []
responses = []

# 각 질문에 대한 답변을 생성
for question in user_question:
    professor_info = []
    if cour in question:
        course = cour
    else:
        course = ""
    if name in question:
        prof = name
    else:
        name = ""

    for d in data:
        if name == prof and cour == course:
            if course + " " + prof in data:
                professor_info.append(data[course + " " + prof])

        elif name == "" and cour == course:
            for i in all_profs:
                if course + " " + i in data:
                    professor_info.append(data[course + " " + i])

        elif name == prof and cour == "":
            for i in all_courses:
                if i + " " + prof in data:
                    professor_info.append(data[i + " " + prof])

    prof_info_str = ""
    for info in professor_info:
        prof_info_str += f"{info['course_info']} {info['rating']} {info['assignment']} {info['team_project']} {info['grade']} {info['attendence']} {info['test_count']} {info['articles']}\n"

    input_text = f"input: {question}\nprofessor_info: {prof_info_str}"
    input_text = input_text[:10000]

    response = model.predict(
        f"""강의를 찾는 대학생들에게 강의평들을 토대로 수업이 어떤지 알려주는 서비스야, 주어진 강의평들을 요약해서 학생들에게 알려줘{input_text}
        output:
        """,
        **parameters
    )

    responses.append(f"Question : {question}\nResponse from Model: {response.text}")

# 모든 답변 출력
for resp in responses:
    print(resp)
