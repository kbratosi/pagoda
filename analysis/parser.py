import csv
from pprint import pprint

def parse_csv(file_path):
    results = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        for id in range(0, len(lines), 6):
            step = int(id / 6 * 2000) 
            inception = float(lines[id].split(':')[1][:-1])
            fid = float(lines[id+1].split(':')[1].split(",")[0][2:])
            precision = float(lines[id+3].split(':')[1][1:-1])
            recall = float(lines[id+4].split(':')[1][1:-1])
            results.append({
                'step': step,
                'inception': inception,
                'fid': fid,
                'precision': precision,
                'recall': recall
            })
    pprint(results)
    with open("output.csv", mode='w', encoding='utf-8') as file:
        output = csv.DictWriter(file, fieldnames=['step', 'inception', 'fid', 'precision', 'recall'])
        output.writeheader()
        output.writerows(results)

parse_csv('stage2_raw.log')