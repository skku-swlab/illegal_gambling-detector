import asyncio
from playwright.async_api import async_playwright
import random
from datetime import datetime
import urllib.parse
import re
from bs4 import BeautifulSoup
import csv
import os
import psycopg2

# ------------------ DB 연결 설정 ------------------
DB_CONFIG = {
    "host": "gambling-crawling.cnig8owewqhg.ap-northeast-2.rds.amazonaws.com",
    "port": 5432,
    "database": "gambling_db",
    "user": "postgres",
    "password": "codeep12345!",
    "sslmode": "require"
}

def insert_if_new_url(url: str, platform: str, keyword: str, timestamp: datetime):
    if not url:
        return False
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        insert_query = """
            INSERT INTO crawled_sites (url, crawled_at, platform, keyword)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        """
        cur.execute(insert_query, (url, timestamp, platform, keyword))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[✓] DB 저장됨: {url}")
        return True
    except Exception as e:
        print(f"[X] DB 저장 실패: {url} → {e}")
        return False
# --------------------------------------------------

# ------------------ URL 필터링 설정 ------------------
EXCLUDE_KEYWORDS = [
    "youtube.com", "youtu.be", "instagram.com", "naver.", "daum.net", "google.",
    "t.co", "news.", "news1.", "kbs.co.kr", "sbs.co.kr", "mbc.co.kr",
    "yna.co.kr", "tvchosun.", "jtbc.", "chosun.com"
]

def is_filtered_url(url: str) -> bool:
    lower_url = url.lower()
    return any(domain in lower_url for domain in EXCLUDE_KEYWORDS)
# ----------------------------------------------------

# 도박 키워드 불러오기
with open("wordlist.txt", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]
random.shuffle(keywords)

# CSV 초기화
CSV_FILE = "fb_detected_posts.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["timestamp", "keyword", "content", "links", "handles", "codes"])

seen_posts = set()

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = await context.new_page()
        await page.bring_to_front()

        while True:
            keyword = random.choice(keywords)
            print(f"[🔍] 검색 중: {keyword}")
            url = f"https://www.facebook.com/search/top?q={urllib.parse.quote(keyword)}"  # ✅ '모두' 탭으로 이동
            await page.goto(url)
            await page.wait_for_timeout(5000)

            for i in range(8):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                print(f"  ↳ [{i+1}] 스크롤 내림")
                await page.wait_for_timeout(4000)

            await page.evaluate("""() => {
                document.querySelectorAll('div[role="button"]').forEach(el => {
                    if (el.innerText.includes('더 보기')) {
                        el.click();
                    }
                });
            }""")
            await page.wait_for_timeout(2000)

            posts = await page.query_selector_all("div[data-ad-preview='message']")
            print(f"  ↳ 감지된 게시물 수: {len(posts)}")

            for post in posts:
                try:
                    html = await post.inner_html()
                    text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)

                    urls_http = re.findall(r'https?://[^\s]+', text)
                    urls_obf = re.findall(r'[\w가-힣\.\-]+\s*\.\s*(?:com|net|org|co|kr)', text, re.IGNORECASE)
                    links = urls_http + urls_obf

                    handles = re.findall(r'@\w{3,}', text)
                    codes = re.findall(r'가입\s*코드[:：]?\s*[\w\-]+', text, re.IGNORECASE)

                    if not any(keyword in text for keyword in keywords):
                        continue
                    if not (links or handles or codes):
                        continue
                    if text in seen_posts:
                        continue
                    seen_posts.add(text)

                    print(f"[💬] {text}")
                    print(f"🔗 링크: {', '.join(links)}")
                    print(f"📱 아이디: {', '.join(handles)}")
                    print(f"🔐 가입코드: {', '.join(codes)}\n{'-'*60}")

                    # ✅ DB 저장 (필터링 포함)
                    for url in links:
                        url = url.strip()
                        if not url or is_filtered_url(url):
                            continue
                        insert_if_new_url(
                            url=url,
                            platform="SNS(facebook)",
                            keyword=keyword,
                            timestamp=datetime.now()
                        )

                    # ✅ CSV 저장 (기존 로직 유지)
                    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                        writer.writerow([
                            datetime.now().isoformat(),
                            keyword,
                            text,
                            "; ".join(links),
                            "; ".join(handles),
                            "; ".join(codes)
                        ])

                except Exception as e:
                    print(f"[⚠️] 게시글 처리 오류: {e}")

            await asyncio.sleep(15)

asyncio.run(run())
