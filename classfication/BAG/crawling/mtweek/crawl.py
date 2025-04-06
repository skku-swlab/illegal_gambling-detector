import requests
from bs4 import BeautifulSoup
import time
import random
import csv
import os

def get_article_links_from_page(page_url, headers):
    try:
        # 페이지 요청
        response = requests.get(page_url, headers=headers)
        
        # 404 에러 체크
        if response.status_code == 404:
            print(f"페이지가 존재하지 않습니다: {page_url}")
            return [], None, False
            
        response.raise_for_status()
        
        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 게시글 링크를 담을 리스트
        article_links = []
        
        # div id="tdi_70" 찾기
        target_div = soup.find('div', {'id': 'tdi_70'})
        
        if target_div:
            # 모든 a 태그 찾기
            links = target_div.find_all('a', href=True)
            
            # 링크 추출
            for link in links:
                article_links.append(link['href'])
        
        # 다음 페이지 링크 찾기
        next_page = None
        pagination = soup.find('div', class_='page-nav td-pb-padding-side')
        if pagination:
            next_link = pagination.find('a', class_='next page-numbers')
            if next_link:
                next_page = next_link['href']
        
        # 중복 제거된 링크와 다음 페이지 URL 반환
        return list(set(article_links)), next_page, True
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return [], None, False

def get_article_content(url, headers):
    try:
        # 랜덤 딜레이 추가 (1~3초)
        time.sleep(random.uniform(1, 3))
        
        # 페이지 요청
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # BeautifulSoup 객체 생성
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 텍스트 수집을 위한 리스트
        texts = []
        
        # 제목 가져오기
        title = soup.find('h1', {'class': 'tdb-title-text'})
        title_text = title.text.strip() if title else "제목 없음"
        
        # 본문 div 찾기
        content_divs = soup.find_all('div', class_=lambda x: x and all(c in x for c in ['tdb-block-inner', 'td-fix-index']))
        
        for content_div in content_divs:
            # p 태그 텍스트 수집
            p_tags = content_div.find_all('p', recursive=True)
            for p in p_tags:
                text = p.text.strip()
                if text:
                    texts.append(text)
            
            # li 태그 텍스트 수집
            li_tags = content_div.find_all('li', recursive=True)
            for li in li_tags:
                text = li.text.strip()
                if text:
                    texts.append(text)
            
            # strong 태그 텍스트 수집
            strong_tags = content_div.find_all('strong', recursive=True)
            for strong in strong_tags:
                text = strong.text.strip()
                if text:
                    texts.append(text)
            
            # h1~h6 태그 텍스트 수집
            for i in range(1, 7):
                h_tags = content_div.find_all(f'h{i}', recursive=True)
                for h in h_tags:
                    text = h.text.strip()
                    if text:
                        texts.append(text)
        
        return {
            'title': title_text,
            'content': '\n'.join(texts)
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching {url}: {e}")
        return None

def split_into_sentences(text):
    # 문장 구분자로 사용할 패턴들
    delimiters = ['. ', '? ', '! ', '.\n', '?\n', '!\n', '\n\n']
    
    # 문장 분리
    sentences = [text]
    for delimiter in delimiters:
        new_sentences = []
        for sentence in sentences:
            split = sentence.split(delimiter)
            for i, s in enumerate(split):
                if s.strip():  # 빈 문장 제외
                    if i < len(split) - 1:
                        new_sentences.append(s.strip() + delimiter.strip())
                    else:
                        new_sentences.append(s.strip())
        sentences = new_sentences
    
    # 빈 문장 및 중복 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    return list(dict.fromkeys(sentences))

def save_to_csv(articles, filename='mtweek_articles.csv'):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 헤더 작성
            writer.writerow(['id', 'label', 'title', 'content'])
            
            # 전체 문장 id를 위한 카운터
            sentence_id = 1
            
            # 각 게시글에 대해
            for article in articles:
                title = article['title']
                content = article['content']
                
                # content를 문장 단위로 분리
                sentences = split_into_sentences(content)
                
                # 각 문장을 개별 행으로 저장
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # 너무 짧은 문장 제외
                        writer.writerow([
                            sentence_id,  # 단순 증가하는 id
                            1,  # label (항상 1)
                            title,
                            sentence.strip()
                        ])
                        sentence_id += 1
        
        print(f"데이터가 {filename}에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

def main():
    # User-Agent 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 기본 URL 구조
    base_url = "https://mtweek.com/%eb%a8%b9%ed%8a%80%ec%86%8c%ec%8b%9d/page/{page}/"
    
    # 수집된 게시글을 저장할 리스트
    articles = []
    page_num = 1
    max_pages = 63  # 최대 페이지 수 설정
    
    while page_num <= max_pages:
        current_page = base_url.format(page=page_num) if page_num > 1 else "https://mtweek.com/%eb%a8%b9%ed%8a%80%ec%86%8c%ec%8b%9d/"
        print(f"\n=== {page_num}페이지 처리 중 ({current_page}) ===")
        
        # 현재 페이지의 게시글 링크 수집
        links, _, page_exists = get_article_links_from_page(current_page, headers)
        
        if not page_exists:
            print(f"마지막 페이지에 도달했습니다. (페이지 {page_num})")
            break
            
        if not links:
            print(f"{page_num}페이지에서 게시글을 찾을 수 없습니다.")
            page_num += 1
            continue
            
        print(f"발견된 게시글 수: {len(links)}")
        
        # 각 게시글 처리
        for i, link in enumerate(links, 1):
            print(f"[{i}/{len(links)}] 게시글 수집 중: {link}")
            article_data = get_article_content(link, headers)
            
            if article_data:
                articles.append(article_data)
                print(f"성공: {article_data['title']}")
            else:
                print(f"실패: {link}")
        
        # 다음 페이지로 이동
        page_num += 1
        
        # 페이지 간 딜레이 추가 (3~5초)
        if page_num <= max_pages:
            delay = random.uniform(3, 5)
            print(f"다음 페이지로 이동 대기 중... ({delay:.1f}초)")
            time.sleep(delay)
    
    # CSV 파일로 저장
    if articles:
        save_to_csv(articles)
        print(f"총 {len(articles)}개의 게시글이 수집되었습니다.")
    else:
        print("저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
