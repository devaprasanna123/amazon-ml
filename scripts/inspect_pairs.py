import pandas as pd

pg = r'D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv'
p1 = r'D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv'
p2 = r'D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv'
p3 = r'D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv'

gt = pd.read_csv(pg, sep='\t', dtype=str, keep_default_na=False, nrows=5000)
gt_with = gt[gt['matched_entity_ids'] != ''].head(300)

s1_needed = set(gt_with['source1_entity_id'].tolist())
s2_needed, s3_needed = set(), set()
for _, row in gt_with.iterrows():
    for mid in row['matched_entity_ids'].split(','):
        mid = mid.strip()
        if mid.startswith('S2-'):
            s2_needed.add(mid)
        elif mid.startswith('S3-'):
            s3_needed.add(mid)

def load_filtered(path, needed_ids, chunksize=20000):
    result = {}
    for chunk in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=chunksize):
        sub = chunk[chunk['entity_id'].isin(needed_ids)]
        for _, r in sub.iterrows():
            result[r['entity_id']] = r.to_dict()
        if len(result) >= len(needed_ids):
            break
    return result

s1d = load_filtered(p1, s1_needed)
s2d = load_filtered(p2, s2_needed)
s3d = load_filtered(p3, s3_needed)

shown = 0
for _, row in gt_with.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1d:
        continue
    r1 = s1d[s1_id]
    for mid in row['matched_entity_ids'].split(',')[:1]:
        mid = mid.strip()
        r2 = s2d.get(mid) or s3d.get(mid)
        if r2 is None:
            continue
        src = mid[:2]
        n1 = r1['business_name'][:70]
        n2 = r2['business_name'][:70]
        a1 = r1['business_address'][:90]
        a2 = r2['business_address'][:90]
        ctr = r1['country']
        print(f'--- {shown+1} ({src}) ---')
        print(f'S1 name : {n1!r}')
        print(f'Sx name : {n2!r}')
        print(f'S1 addr : {a1!r}')
        print(f'Sx addr : {a2!r}')
        print(f'country : {ctr}')
        print()
        shown += 1
    if shown >= 25:
        break
