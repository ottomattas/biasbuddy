import csv
import os
import json_lines
from tqdm import tqdm

# filepath = "D:\\Downloads\\politics.corpus\\utterances.jsonl"
dirs = "Dataset"


def dict2csv(dictlist, csvfile):
    keys = dictlist[0].keys()
    with open(csvfile, 'w', newline='', encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        try:
            dict_writer.writerows(dictlist)
        except UnicodeEncodeError:
            print("Couldn't do this.")


small_dic_list = []
print("Started.")
for idx,filepath in enumerate(os.listdir(dirs)):
    with open(filepath, 'rb') as f:
        for item in tqdm(json_lines.reader(f)):
            if item["text"] == '' or item["text"] == "[deleted]" or item["text"] == "[removed]":
                continue
            else:
                small_dic_list.append(
                    {"id": item["id"], "created": item["timestamp"], "body": item["text"], "vote": item["meta"]["score"]})
        print(f"Done processing {len(small_dic_list)} comments.")
        dict2csv(small_dic_list, f"Dataset/PoliticsBig_{idx}.csv")
        print(f"Finished {idx}.")
