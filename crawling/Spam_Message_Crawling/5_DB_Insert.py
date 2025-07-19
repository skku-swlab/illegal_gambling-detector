from config import GAMBLING_VALID_PATH
import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": "Your DB Host",
    "user": "Your DB User",
    "port": Port Number,  # e.g., 5432
    "database": "Your DB Name",
    "password": "Your DB Password",
    "sslmode": "require"
}

def get_connection():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["database"],
        password=DB_CONFIG["password"],
        sslmode=DB_CONFIG["sslmode"]
    )

def insert_url_if_not_exists(url):
    conn = get_connection()
    inserted = False
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO crawled_sites (url, reachable, response_time_ms, ip_address, account_no, platform)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    (url, None, None, None, None, "sms")
                )
                if cur.rowcount == 0:
                    print(f"이미 존재: {url}")
                else:
                    print(f"삽입 완료: {url}")
                    inserted = True
    finally:
        conn.close()
    return inserted

if __name__ == "__main__":
    csv_path = GAMBLING_VALID_PATH
    df = pd.read_csv(csv_path)
    urls = df['restored_link'].dropna().unique()

    print(f"총 {len(urls)}개의 URL을 DB에 삽입 시도합니다.")
    inserted_count = 0
    for url in urls:
        if insert_url_if_not_exists(url):
            inserted_count += 1
    print(f"\n최종적으로 삽입된(중복 제거된) 링크 수: {inserted_count}")
