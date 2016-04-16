import pickle
from scipy.spatial import distance

def process_tsp(tsp_seq_file):
    with open(tsp_seq_file) as tspfile:
        for line in tspfile:
            tsp_seq_str = line
            break
        tsp_seq_str = tsp_seq_str.replace('\n', '').replace('{', '').replace('}', '')
        tsp_seq_split = tsp_seq_str.split(',')
        tsp_seq = [int(x) for x in tsp_seq_split]
        return tsp_seq

def process_dis(tsp_seq, people_embedding_file):
    prev_to_cur = []
    cur_to_next = []
    people_embedding = pickle.load(open(people_embedding_file, 'rb'))
    n = len(tsp_seq)
    for index, people_id in enumerate(tsp_seq):
        prev_to_cur.append(distance.euclidean(people_embedding[tsp_seq[index - 1]], people_embedding[tsp_seq[index]]))
        cur_to_next.append(distance.euclidean(people_embedding[tsp_seq[index]], people_embedding[tsp_seq[(index + 1) % n]] ))
    return prev_to_cur, cur_to_next