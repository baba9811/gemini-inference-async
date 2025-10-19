# Gemini Inference Async

Google Gemini 멀티모달 모델 비동기 추론 시스템 (체크포인트 지원)

## 주요 기능

- 비동기 배치 추론 (async/await 사용)
- 자동 체크포인트 시스템 (N개마다 저장 및 재개)
- 멀티모달 지원 (텍스트 + 이미지)
- 유연한 CLI 설정
- 상세한 에러 메시지
- 실시간 진행률 표시 (tqdm)

## 요구사항

- Python 3.12+
- uv (Python 패키지 매니저)
- Google API Key
- Hugging Face Token (private 데이터셋용)

## 설치

1. 저장소 클론:
```bash
git clone <repository-url>
cd gemini-inferene-async
```

2. uv로 의존성 설치:
```bash
uv sync
```

3. .env 파일 생성:
```bash
cp .env.example .env
```

.env 파일 수정:
```
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## 사용법

### 기본 사용

```bash
uv run python app/main.py
```

### 상세 설정

```bash
uv run python app/main.py \
  --model gemini-2.0-flash-exp \
  --split test \
  --text-column input_query \
  --image-column image_path \
  --max-concurrent 10 \
  --save-every 20 \
  --limit 100
```

### 체크포인트에서 재개

```bash
uv run python app/main.py --resume
```

## CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `gemini-2.0-flash-exp` | Gemini 모델명 |
| `--split` | `test` | 데이터셋 분할 (train/test) |
| `--text-column` | `input_query` | 텍스트 입력 컬럼명 |
| `--image-column` | `image_path` | 이미지 데이터 컬럼명 |
| `--max-concurrent` | `10` | 최대 동시 요청 수 |
| `--save-every` | `10` | N개마다 체크포인트 저장 |
| `--output-dir` | `result` | 출력 디렉토리 |
| `--limit` | `None` | 처리할 샘플 수 제한 |
| `--resume` | `False` | 체크포인트에서 재개 |

## 사용 가능한 모델

- `gemini-2.5-flash` (기본값)
- `gemini-2.5-pro`
- `gemini-2.5-flash-lite`
- `gemini-2.0-flash`
- `gemini-2.0-flash-lite`

## 출력 형식

결과는 `result/{모델명}_{split}_result.csv`에 저장:

- `id`: 샘플 ID
- `input_query`: 입력 텍스트
- `has_image`: 이미지 존재 여부 (True/False)
- `response`: 모델 응답
- `status`: 추론 상태 (success/error)
- `error_message`: 에러 상세 정보 (에러 시)

## 동작 원리

### 비동기 배치 처리

1. Hugging Face에서 데이터셋 로드
2. 배치로 분할 (크기 = save_every)
3. asyncio.gather()로 각 배치 비동기 처리
4. 각 배치 후 결과 저장

### 체크포인트 시스템

- N개 샘플마다 자동 저장
- 단일 체크포인트 파일 (매번 덮어쓰기)
- Resume으로 기존 진행 감지
- Ctrl+C로 안전 중단

### 예시 워크플로우

```bash
# 추론 시작
uv run python app/main.py --model gemini-2.0-flash-exp --save-every 20

# 중단된 경우 재개:
uv run python app/main.py --model gemini-2.0-flash-exp --resume
```

## 프로젝트 구조

```
gemini-inferene-async/
├── app/
│   ├── main.py           # 메인 추론 스크립트
│   └── dataloader.py     # 데이터셋 로더
├── result/               # 출력 디렉토리
├── .env                  # API 키
├── .env.example          # 템플릿
├── pyproject.toml        # 의존성
├── uv.lock              # 잠긴 의존성
└── README.md            # 이 파일
```

## 성능

- 비동기 속도 향상: 순차 대비 약 10배
- 동시 요청 수: --max-concurrent로 제어 (기본: 10)
- 배치 크기: --save-every로 제어 (기본: 10)

예시: 500개 샘플
- 순차 처리: ~1000초 (샘플당 2초)
- 비동기 처리 (10개 동시): ~100초

## 에러 핸들링

- API Rate Limit: semaphore로 제어
- 네트워크 이슈: error_message에 기록
- Safety Filter: prompt feedback과 함께 로깅
- 잘못된 이미지: 텍스트 전용으로 폴백

## 문제 해결

### API 키 문제
```
ValueError: GOOGLE_API_KEY is not set in .env file
```
.env 파일에 유효한 GOOGLE_API_KEY가 있는지 확인

### 데이터셋 접근 문제
```
DatasetNotFoundError: Dataset 'xxx' doesn't exist
```
private 데이터셋은 .env의 HUGGINGFACE_TOKEN 확인

### 메모리 부족
--max-concurrent 또는 --save-every 값 감소
