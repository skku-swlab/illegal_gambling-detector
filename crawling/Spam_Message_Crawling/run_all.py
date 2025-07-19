import subprocess
import sys
import os

# Hugging Face Hub 패키지 설치
print("\nHugging Face Hub 패키지 설치 중...")
install_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"],
    capture_output=True,
    text=True
)

# Hugging Face 토큰을 아래에 직접 입력하세요
HF_TOKEN = "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"

try:
    from huggingface_hub import login
    print("\nHugging Face에 로그인 중...")
    login(token=HF_TOKEN)
    print("✅ Hugging Face 로그인 성공.")
except Exception as e:
    print(f"❌ Hugging Face 로그인 실패: {str(e)}")
    exit(1)

# 단계별 실행 파일 리스트
steps = [
    "1_CSV_Preprocess.py",
    "2_Link_Classifier.py",
    "3_Link_Extract.py",
    "4_Link_validate.py",
    "5_DB_Insert.py"
]

# 작업 디렉토리 설정
base_dir = os.path.dirname(os.path.abspath(__file__))

for step in steps:
    print(f"\n===== {step} 실행 중 =====")
    result = subprocess.run(
        [sys.executable, os.path.join(base_dir, step)],
        capture_output=True,
        text=True,
        cwd=base_dir  # 작업 디렉토리를 스크립트 위치로 설정
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print(f"❌ {step} 실행 중 오류 발생. 중단합니다.")
        break
else:
    print("\n✅ 전체 파이프라인이 정상적으로 완료되었습니다.")
