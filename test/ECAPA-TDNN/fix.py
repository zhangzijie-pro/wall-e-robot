import os

IN_LIST  = r"processed/cn_celeb2/val_fbank_list.txt"
OUT_LIST = r"processed/cn_celeb2/val_fbank_list.fixed.txt"

IN_LIST = os.path.abspath(IN_LIST)
OUT_LIST = os.path.abspath(OUT_LIST)

cnt_ok = 0
cnt_fail = 0

with open(IN_LIST, "r", encoding="utf-8") as fin, \
     open(OUT_LIST, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        lab, p = line.split(maxsplit=1)

        # 统一路径格式
        p = p.strip().strip('"').strip("'")
        p = p.replace("\\", "/")

        # 🔥 修复 processed/processed
        p = p.replace("/processed/processed/", "/processed/")

        # 再转成规范绝对路径
        p_abs = os.path.normpath(p)

        if os.path.exists(p_abs):
            p_write = p_abs.replace("\\", "/")
            fout.write(f"{lab} {p_write}\n")
            cnt_ok += 1
        else:
            cnt_fail += 1

print("fixed:", cnt_ok)
print("still missing:", cnt_fail)
print("output:", OUT_LIST)
