import asyncio
from playwright.async_api import async_playwright
import random
from datetime import datetime

# 사용방법
# 크롬 디버깅 모드 실행 (CMD에서 다음 처럼 입력 단, 크롬 디렉토리 위치 확인필요)
# "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\ChromeDebug"
# REM 새로 뜬 창에서 트위터에 로그인 하기
# 새 터미널에서 크롤러 실행
# py x_crawler_cdp.py


# 키워드 로딩
with open("wordlist.txt", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]
random.shuffle(keywords)

seen_posts = set()

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://localhost:9222")
        page = browser.contexts[0].pages[0]
        await page.bring_to_front()

        while True:
            keyword = random.choice(keywords)
            print(f"[🔍] 검색 중: {keyword}")

            search_url = f"https://twitter.com/search?q={keyword}&src=typed_query&f=live"
            await page.goto(search_url)
            await page.wait_for_timeout(3000)

            # ✅ 스크롤 여러 번 내리기
            for i in range(8):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                print(f"  ↳ [{i+1}] 스크롤 내림")
                await page.wait_for_timeout(2500)

            # ✅ 게시글 수집 (스크롤 끝난 뒤에!)
            tweets = await page.query_selector_all("article")

            for tweet in tweets:
                content = await tweet.inner_text()
                if content in seen_posts:
                    continue
                seen_posts.add(content)

                if any(k in content for k in keywords):
                    print(f"[💬] 감지된 게시글:\n{content}\n{'-'*60}")
                    with open("detected_posts.txt", "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now()}] 키워드: {keyword}\n{content}\n{'-'*80}\n\n")

            await asyncio.sleep(10)

asyncio.run(run())
