from tqdm import tqdm
new_lines = []
for line in tqdm(open('psgs_w100.tsv').readlines()[1:]):
    id, text, title = line.strip().split("\t")
    new_lines.append(title+'\t'+text+'\n')

with open('data.csv','w') as f:
    f.writelines(new_lines)