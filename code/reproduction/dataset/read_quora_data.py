

data_path = './data/quora_duplicate_questions.tsv'

i = 0

with open(data_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        print(line)
        i += 1
        if i > 10:
            break
