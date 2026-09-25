import duckdb
import time

t0 = time.time()
res = duckdb.query("""
    SELECT country, count(*) as cnt 
    FROM read_csv('D:/AMAZON DATASET/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv', delim='\t', header=true)
    GROUP BY country
""").df()
el = time.time() - t0
print(f"DuckDB read 2.2M rows in {el:.2f}s!")
print(res)
