import asyncio
from playwright.async_api import async_playwright
import random
from datetime import datetime
import urllib.parse
import re
import os
import csv
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

CSV_FILE = "x_detected_posts.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["timestamp", "keyword", "content", "links", "handles", "codes"])

with open("wordlist.txt", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]
random.shuffle(keywords)

seen_posts = set()

async def detect_error(page):
    html = await page.content()
    if await page.query_selector('div[role="alert"]'):
        return True
    if "문제가 발생" in html or "Something went wrong" in html or "Try reloading" in html:
        return True
    tweets = await page.query_selector_all("article")
    if len(tweets) == 0:
        return True
    return False

async def search_keyword(context, keyword):
    page = await context.new_page()
    await page.set_extra_http_headers({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36"
    })
    await page.bring_to_front()

    print(f"[🔍] 검색 중: {keyword}")
    encoded_keyword = urllib.parse.quote(keyword)
    search_url = f"https://twitter.com/search?q={encoded_keyword}&src=typed_query&f=live"
    await page.goto(search_url)
    await page.wait_for_timeout(7000)

    if await detect_error(page):
        await page.close()
        return None

    for i in range(8):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        print(f"  ↳ [{i+1}] 스크롤 내림")
        await page.wait_for_timeout(random.randint(5000, 8000))

    tweets = await page.query_selector_all("article")

    for tweet in tweets:
        try:
            content = await tweet.inner_text()

            urls_http = re.findall(r'https?://[^\s]+', content)
            urls_obf = re.findall(r'[\w가-힣\.\-]+\s*\.\s*(?:com|net|org|co|kr)', content, re.IGNORECASE)
            links = urls_http + urls_obf

            handles = re.findall(r'@\w{3,}', content)
            codes = re.findall(r'가입\s*코드[:：]?\s*[\w\-]+', content, re.IGNORECASE)

            if not any(k in content for k in keywords):
                continue
            if not (links or handles or codes):
                continue
            if content in seen_posts:
                continue
            seen_posts.add(content)

            print(f"[💬] {content}")
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
                    platform="SNS(X,트위터)",
                    keyword=keyword,
                    timestamp=datetime.now()
                )

            # ✅ CSV 저장 (기존 로직 그대로)
            with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    datetime.now().isoformat(),
                    keyword,
                    content,
                    "; ".join(links),
                    "; ".join(handles),
                    "; ".join(codes)
                ])

        except Exception as e:
            print(f"[⚠️] 게시글 처리 오류: {e}")

    await page.close()
    return True

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        while True:
            keyword = random.choice(keywords)
            success = await search_keyword(context, keyword)

            if success is None:
                print("[⛔] 오류 감지됨 → 다른 키워드로 새 탭 재시도")
                next_keyword = random.choice(keywords)
                retry = await search_keyword(context, next_keyword)

                if retry is None:
                    print("[🕒] 재시도도 실패 → 3분 대기 후 복귀...")
                    await asyncio.sleep(180)
                    continue

            await asyncio.sleep(random.randint(60, 90))

asyncio.run(run())
