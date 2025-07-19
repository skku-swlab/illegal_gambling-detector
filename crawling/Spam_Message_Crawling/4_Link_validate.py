from config import GAMBLING_CLASSIFIED_PATH, GAMBLING_VALID_PATH, GAMBLING_EXCEPTION_PATH
import pandas as pd
import requests
import idna
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 설정 ----------
file_path = GAMBLING_CLASSIFIED_PATH
output_path = GAMBLING_VALID_PATH
error_output_path = GAMBLING_EXCEPTION_PATH
max_workers = 20  # 병렬 처리 수

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

exclude_keywords = [
    'kakao', 'docs.google.com/forms', 't.me', 'telegram',
    'naver.', 'youtube.', 'youtu.be', 'blog.naver.', 'forms.gle'
]

html_404_keywords = [
    '404', '존재하지 않음', '없는 페이지', '페이지를 찾을 수 없습니다', '삭제되었습니다',
    '주소가 잘못되었습니다', '찾을 수 없습니다', '페이지 오류', '유효하지 않은 링크',
    '잘못된 접근입니다', '경로 오류', '접속할 수 없습니다',
    '요청하신 페이지는 존재하지 않습니다', '페이지가 존재하지 않습니다', '오류가 발생했습니다',
    '페이지 로딩 실패',
    'page not found', 'notfound', 'not found', 'resource not found',
    'this page does not exist', 'invalid url', 'invalid address',
    'site not reachable', 'page does not exist', 'missing page',
    'cannot be found', 'http error 404', 'error 404',
    'file not found', 'oops', 'broken link'
]

def is_domain_resolvable(url):
    try:
        domain = url.split("/")[2]
        socket.gethostbyname(domain)
        return True
    except:
        return False

def validate_url(index, url, seen):
    if not is_domain_resolvable(url):
        return ("error", index, url, "DNS 해석 실패 (NXDOMAIN)")

    try:
        response = requests.get(url, headers=headers, timeout=3, allow_redirects=True)
        final_url = response.url

        try:
            domain = final_url.split("/")[2]
            ascii_domain = idna.encode(domain).decode()
            final_url = final_url.replace(domain, ascii_domain)
        except:
            pass

        if final_url in seen:
            return ("skip", index, url, f"중복 URL: {final_url}")

        if not response.ok:
            return ("error", index, url, f"HTTP 오류: {response.status_code}")

        if any(excl in final_url.lower() for excl in exclude_keywords):
            return ("error", index, url, f"차단 도메인 포함: {final_url}")

        content = response.text.lower()
        if any(keyword in content for keyword in html_404_keywords):
            return ("error", index, url, f"404 키워드 포함됨")

        return ("valid", index, url, final_url)

    except Exception as e:
        return ("error", index, url, f"예외 발생: {e}")

# ---------- 데이터 불러오기 ----------
df = pd.read_csv(file_path)
df = df[df['link_included'] == 1].copy()
df = df.drop_duplicates(subset='restored_link')
df = df[df['restored_link'].notnull()]
urls = df['restored_link'].tolist()
total = len(urls)

valid_links = []
error_links = []
seen_urls = set()

print(f"🔍 총 {total}개의 고유 링크를 병렬 처리합니다 (max_workers = {max_workers})\n")

# ---------- 병렬 처리 ----------
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(validate_url, i, url, seen_urls): url for i, url in enumerate(urls)}

    for future in as_completed(futures):
        result_type, index, original_url, msg = future.result()

        if result_type == "valid":
            if msg not in seen_urls:
                seen_urls.add(msg)
                valid_links.append({'restored_link': msg})
                print(f"  ✅ [{index}] {original_url} → OK → {msg}")
        elif result_type == "error":
            error_links.append({'restored_link': original_url})
            print(f"  ❌ [{index}] {original_url} → {msg}")
        elif result_type == "skip":
            print(f"  🔁 [{index}] {original_url} → {msg}")

# ---------- 결과 저장 ----------
pd.DataFrame(valid_links).to_csv(output_path, index=False)
pd.DataFrame(error_links).to_csv(error_output_path, index=False)

print(f"\n✔ 완료! 유효한 링크 {len(valid_links)}개 저장됨 → {output_path}")
print(f"⚠️ 예외 발생 링크 {len(error_links)}개 따로 저장됨 → {error_output_path}")
