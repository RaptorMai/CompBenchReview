from collections import defaultdict
import json
import os

if os.path.isfile('gemini_tracker.json'):
    all_keys = json.load(open('gemini_tracker.json'))
    for i in open("gemini_key.txt", "r").readlines():
        if not i.startswith('#'):
            key = i.replace("\n", "")
            if key not in all_keys:
                all_keys[key] = {'last_inf_time': 0, 'count_minute': 0, 'count_day': 0}
    with open('gemini_tracker.json', 'w') as fout:
        json.dump(all_keys, fout)



else:
    all_keys = defaultdict(dict)
    for i in open("gemini_key.txt", "r").readlines():
        if not i.startswith('#'):
            all_keys[i.replace("\n", "")] = {'last_inf_time': 0, 'count_minute': 0, 'count_day': 0}

    with open('gemini_tracker.json', 'w') as fout:
        json.dump(all_keys, fout)