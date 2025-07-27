import json
import os
import glob

# 새로운 저장 경로 설정
save_dir = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset/ablation_study/illegal'

# 저장 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# json 파일들을 재귀적으로 찾기
files = glob.glob('/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset/illegal/*.json', recursive=True)

count = 0
modified_files = 0

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"파일 읽기 오류: {file}")
            continue
        
        changed = False
        
        # nodes 리스트에서 gambling_score가 1.0인 항목을 0.0으로 변경
        for node in data.get('nodes', []):
            if node.get('gambling_score') == 1.0:
                node['gambling_score'] = 0.0
                changed = True
                count += 1
        
        # 변경사항이 있으면 파일에 저장
        if changed:
            modified_files += 1
            # 파일명만 추출
            filename = os.path.basename(file)
            # 새 저장 경로에 파일 저장
            save_path = os.path.join(save_dir, filename)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

print(f'수정된 파일 수: {modified_files}')
print(f'수정된 노드 수: {count}') 