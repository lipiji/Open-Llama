import json
from glob import glob
from tqdm import tqdm
import zstandard as zstd

paths = glob("/data/pjli/data/chatgpt/piji/baike*")
write_path = "./pretrain_data/part-pbaike-{}.jsonl.zst"
total_num = 0
file_num = 0
wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
for path in tqdm(paths, total=len(paths)):
    with open(path, encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                line = json.loads(line, strict=False)
            except:
                print(line)
                continue
            if total_num % 65536 == 0 and total_num > 0:
                file_num += 1
                wfp.close()
                wfp = zstd.open(write_path.format(file_num), "wb", encoding="utf-8")
            wfp.write(json.dumps(line).encode("utf-8"))
            wfp.write("\n".encode("utf-8"))
            total_num += 1
wfp.close()
print("total line: {}\ntotal files: {}".format(total_num, file_num))
