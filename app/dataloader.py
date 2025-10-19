from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def initialize_huggingface():
    """Hugging Face 로그인 초기화"""
    if HUGGINGFACE_TOKEN:
        login(token=HUGGINGFACE_TOKEN)
        print("Hugging Face 로그인 성공!")
    else:
        print("경고: HUGGINGFACE_TOKEN이 .env 파일에 설정되지 않았습니다.")
        print("Private 데이터셋에 접근하려면 .env 파일에 토큰을 추가하세요.")

def load_fence_dataset(split: str = "train") -> pd.DataFrame:
    """
    FENCE 데이터셋을 로드합니다.

    Args:
        split: 'train' 또는 'test'

    Returns:
        pandas DataFrame
    """
    initialize_huggingface()

    # 데이터셋 로드 (private 데이터셋을 위해 token 전달)
    dataset = load_dataset('miraekiim/FENCE', token=HUGGINGFACE_TOKEN)

    # pandas DataFrame으로 변환
    df = dataset[split].to_pandas()

    print(f"{split.capitalize()} 데이터: {len(df)}개")
    print(f"컬럼: {df.columns.tolist()}")

    return df

if __name__ == "__main__":
    # 테스트용 코드
    train_df = load_fence_dataset("train")
    test_df = load_fence_dataset("test")
    print("\n샘플 데이터:")
    print(train_df.head())