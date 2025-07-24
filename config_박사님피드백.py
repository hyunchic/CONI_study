import argparse
from pathlib import Path


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=r'C:\Users\rhs69\Desktop\CONI_hsikr\h_sikr\pythonnew')

    config = parser.parse_args()  # 기본 인자로 설정 객체 생성
    return config

