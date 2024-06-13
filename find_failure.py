import os
import json

results = 'results'
target = 'depth'
wrong = None
output_name = f'{target}_wrong_gpt.json'

for folder in os.scandir(results):
    if folder.name == '.DS_Store' or target not in folder.name:
        continue
    print(folder.name)
    for result_file in list(os.scandir(folder)):
        if 'decompose' not in result_file.name or os.path.isdir(result_file):
            if result_file.name == '.DS_Store':
                continue
            tmp = set()
            print(result_file)
            if result_file.name.endswith('json'):
                data = json.load(open(result_file, 'r'))
            else:
                data = [json.loads(q) for q in open(os.path.expanduser(result_file), "r")]

            key = f"{result_file.name.split('.')[0].split('_')[-1]}_answer"
            for i in data:
                if 'Left' in i[key] or 'First' in i[key]:
                    i[key] = 'Left'
                elif 'Right' in i[key] or 'Second' in i[key]:
                    i[key] = 'Right'
                else:
                    print(i)
                if i[key] != i['answer']:
                    tmp.add((i['image_1'], i['image_2']))
            if 'gpt' in key:
                with open(output_name, 'w') as fout:
                    json.dump(list(tmp), fout)
            if not wrong:
                wrong = tmp
            else:
                wrong = wrong.intersection(tmp)
    print('overalp all model')
    print(len(wrong))
    break
    print('----------------------------------')