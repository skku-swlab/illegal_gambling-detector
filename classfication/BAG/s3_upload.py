import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from typing import Optional

# aws.env 파일에서 환경 변수 로드
load_dotenv('aws.env')

def upload_to_s3(image_path: str, file_name: Optional[str] = None) -> Optional[str]:
    """
    이미지 파일을 S3에 업로드하고 객체 키를 반환합니다.
    
    Args:
        image_path (str): 업로드할 이미지 파일의 로컬 경로
        file_name (str, optional): S3에 저장될 파일 이름. 기본값은 None이며, 
                                 None인 경우 원본 파일 이름을 사용합니다.
    
    Returns:
        str: 업로드된 파일의 S3 객체 키. 업로드 실패 시 None을 반환합니다.
    """
    try:
        # 환경 변수 확인
        access_key = os.getenv('AWS_ACCESS_KEY_ID', '').strip('"')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '').strip("'").strip('"')
        region = os.getenv('AWS_REGION', 'ap-northeast-2').strip('"')
        
        if not access_key or not secret_key:
            raise ValueError("AWS 자격 증명이 설정되지 않았습니다.")
        
        # S3 클라이언트 생성
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # 파일 존재 여부 확인
        if not os.path.exists(image_path):
            print(f"에러: 파일이 존재하지 않습니다. 경로: {image_path}")
            return None
            
        # 파일 이름 설정
        if file_name is None:
            file_name = os.path.basename(image_path)
        
        # S3 설정
        bucket_name = 'gambling-image-bucket'
        prefix = 'gambling-screenshot/'
        
        # S3에 저장될 파일 경로
        s3_key = f"{prefix}{file_name}"
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        
        print(f"업로드 시작: {image_path} -> {s3_uri}")
        
        # 파일 업로드
        s3_client.upload_file(
            image_path,
            bucket_name,
            s3_key
        )
        
        print("파일 업로드 성공!")
        
        # 파일이 실제로 존재하는지 확인
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            print(f"파일이 S3에 존재합니다: {s3_uri}")
            return s3_key  # 객체 키 반환
        except ClientError as e:
            print(f"파일이 S3에 존재하지 않습니다: {e}")
            return None
        
    except ClientError as e:
        print(f"AWS 에러 발생: {e}")
        return None
    except Exception as e:
        print(f"예상치 못한 에러 발생: {e}")
        print(f"에러 타입: {type(e)}")
        import traceback
        print(f"상세 에러: {traceback.format_exc()}")
        return None

def test_s3_upload():
    """테스트 함수"""
    test_image_path = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/inference/screenshot/bk-486_com_20250528_155303.png'
    key = upload_to_s3(test_image_path)
    if key:
        print(f"업로드된 파일 객체 키: {key}")

if __name__ == "__main__":
    test_s3_upload()