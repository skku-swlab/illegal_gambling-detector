from config import GAMBLING_CLASSIFIED_PATH
import pandas as pd
import re

# 파일 경로
file_path = GAMBLING_CLASSIFIED_PATH

# ------------------------
# 링크 복원 로직 정의
# ------------------------

# 동그라미 소문자: ⓐⓑⓒ...ⓩ
circle_map = {
    'ⓐ':'a','ⓑ':'b','ⓒ':'c','ⓓ':'d','ⓔ':'e','ⓕ':'f','ⓖ':'g','ⓗ':'h','ⓘ':'i','ⓙ':'j',
    'ⓚ':'k','ⓛ':'l','ⓜ':'m','ⓝ':'n','ⓞ':'o','ⓟ':'p','ⓠ':'q','ⓡ':'r','ⓢ':'s','ⓣ':'t',
    'ⓤ':'u','ⓥ':'v','ⓦ':'w','ⓧ':'x','ⓨ':'y','ⓩ':'z'
}

# 동그라미 숫자: ①~⑨, ⓪
number_circle_map = {
    '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
    '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⓪': '0'
}

def circle_to_alpha(m):
    return circle_map.get(m.group(0), m.group(0))

def number_circle_to_digit(m):
    return number_circle_map.get(m.group(0), m.group(0))

def robust_link_restorer(msg: str) -> str:
    text = msg

    # 1) 동그라미 문자 → 알파벳
    text = re.sub(r'[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ]', circle_to_alpha, text)

    # 2) 동그라미 숫자 → 숫자
    text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⓪]', number_circle_to_digit, text)

    # 3) 유사 문자 치환
    replacements = {
        'Ø': 'o', 'Ｏ': 'o', 'Ｓ': 's', '’': "'",
        'Ｋ': 'k', 'Ｒ': 'r', 'Ｗ': 'w', 'ｍ': 'm',
        'ｃ': 'c', 'ｏ': 'o', 'ｋ': 'k'
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # 4) .com 숨김 복원
    text = re.sub(r'쩜\s?꺼\s?엄', '.com', text)
    text = re.sub(r'쩜\s?컴', '.com', text)

    # 5) 공백 포함 도메인 정리
    text = re.sub(r'(http[s]?:\/\/)?([a-zA-Z0-9\-_]+\s+\.\s*[a-zA-Z]+)',
                  lambda m: m.group(0).replace(' ', ''), text)

    # 6) open.kakao.com
    kakao = re.search(r'(open\.kakao\.com/[^\s"\'<>가-힣]+)', text)
    if kakao:
        return 'https://' + kakao.group(1)

    # 7) 짧은 링크
    short_link = re.search(
        r'(https?:\/\/)?(vo\.la|goo\.su|t\.ly|bit\.ly|zxcv\.be|iii\.im|miniurl\.com|q6c\.us|2cm\.es|buly\.kr|pf\.kakao\.com|cm9\.site|qaa\.kr|enn\.kr|iri\.my|t\.me|buly\.kr)/[^\s"\'<>가-힣]+',
        text
    )
    if short_link:
        link = short_link.group(0)
        return link if link.startswith('http') else f"http://{link}"

    # 8) 일반 도메인 + path
    domain_path = re.search(
        r'(https?:\/\/)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(\/[^\s"\'<>가-힣]*)?)',
        text
    )
    if domain_path:
        return domain_path.group(0) if domain_path.group(1) else f"http://{domain_path.group(2)}"

    # 9) 기호 포함 .com 후보
    dotcom = re.search(r'([a-zA-Z0-9\-<>\[\]\(\)\'\":]{1,30})\.com', text)
    if dotcom:
        raw = dotcom.group(1)
        raw = re.sub(r'[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ]', circle_to_alpha, raw)
        raw = re.sub(r'[①②③④⑤⑥⑦⑧⑨⓪]', number_circle_to_digit, raw)
        raw = re.sub(r'[^a-zA-Z0-9\-]', '', raw)
        raw = re.sub(r'^-+', '', raw)
        if raw:
            return f"http://{raw}.com"

    # 10) 키워드 기반 추정 도메인
    hidden = re.findall(r'\b([a-zA-Z0-9\-]{3,})\b', text)
    for cand in hidden:
        if any(kw in cand.lower() for kw in ['bet','casino','slot','tok','lotto','kakao']):
            return f"http://{cand}.com"

    return ''

# ------------------------
# CSV 불러오기 및 저장
# ------------------------

df = pd.read_csv(file_path)
filtered_df = df[df['link_included'] == 1].copy()

# 링크 복원 수행
filtered_df['restored_link'] = filtered_df['message'].apply(robust_link_restorer)

# 저장
output_path = GAMBLING_CLASSIFIED_PATH
filtered_df.to_csv(output_path, index=False)
