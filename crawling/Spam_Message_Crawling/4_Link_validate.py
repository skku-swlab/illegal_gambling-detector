import pandas as pd
import requests
from time import sleep
import idna

# ---------- 설정 ----------
file_path = "원본데이터/20250402_SKKU_gambling.csv"
output_path = "원본데이터/20250402_SKKU_gambling_valid_only.csv"
error_output_path = "원본데이터/20250402_SKKU_gambling_exceptions.csv"
batch_size = 100

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

exclude_keywords = [
    'kakao', 'docs.google.com/forms', 't.me', 'telegram',
    'naver.', 'youtube.', 'youtu.be', 'blog.naver.', 'forms.gle'
]

html_404_keywords = ['404', 'page not found', 'notfound', 'not found', '존재하지 않음', '없는 페이지']

# ---------- 데이터 불러오기 및 중복 제거 ----------
df = pd.read_csv(file_path)
df = df[df['link_included'] == 1].copy()
df = df.drop_duplicates(subset='restored_link')
df = df[df['restored_link'].notnull()]
urls = df['restored_link'].tolist()
total = len(urls)

valid_links = []
error_links = []

print(f"🔍 총 {total}개의 고유 링크를 {batch_size}개씩 배치 처리합니다.\n")

# ---------- 배치 처리 ----------
for i in range(0, total, batch_size):
    batch = urls[i:i + batch_size]
    print(f"📦 배치 {i // batch_size + 1} / {((total - 1) // batch_size) + 1} 처리 중...")

    for j, url in enumerate(batch):
        index = i + j

        try:
            response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
            final_url = response.url

            # 한글 도메인 punycode 처리
            try:
                domain = final_url.split("/")[2]
                ascii_domain = idna.encode(domain).decode()
                final_url = final_url.replace(domain, ascii_domain)
            except:
                pass

            if not response.ok:
                print(f"  ❌ [{index}] {url} → HTTP 오류 ({response.status_code})")
                continue

            if any(excl in final_url.lower() for excl in exclude_keywords):
                print(f"  ❌ [{index}] {url} → 차단 도메인 포함 ({final_url})")
                continue

            content = response.text.lower()
            if any(keyword in content for keyword in html_404_keywords):
                print(f"  ❌ [{index}] {url} → 본문 내 '404' 또는 'not found' 포함됨")
                continue

            # ✅ 유효한 링크
            valid_links.append({'restored_link': url})
            print(f"  ✅ [{index}] {url} → OK → {final_url}")

        except Exception as e:
            print(f"  ⚠️ [{index}] {url} → 접속 실패 (유지): {e}")
            valid_links.append({'restored_link': url})
            error_links.append({'restored_link': url})
            continue

    sleep(1)

# ---------- 결과 저장 (링크만 저장) ----------
pd.DataFrame(valid_links).to_csv(output_path, index=False)
pd.DataFrame(error_links).to_csv(error_output_path, index=False)

print(f"\n✅ 완료! 유효한 링크 {len(valid_links)}개 저장됨 → {output_path}")
print(f"⚠️ 예외 발생 링크 {len(error_links)}개 따로 저장됨 → {error_output_path}")
