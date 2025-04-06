import re

def remove_site_domain(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # site: 뒤의 도메인을 제거하는 정규식 패턴
    pattern = r'site:[^\s]+\s'
    
    # 각 줄에서 site: 도메인 부분만 제거
    new_lines = [re.sub(pattern, '', line) for line in lines]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    input_file = "search_wordlist.txt"
    output_file = "search_wordlist_cleaned.txt"
    remove_site_domain(input_file, output_file)
    print("도메인 제거가 완료되었습니다.") 