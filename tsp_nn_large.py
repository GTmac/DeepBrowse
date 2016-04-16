import pickle
from optparse import OptionParser

import tsp
import gen_tsp_csv
import csv

# load top N people from embedding
def load_embedding(embedding_fname, n):
    embeddings = pickle.load(open(embedding_fname, 'rb'))
    elements = embeddings.keys()
    values = embeddings.values()
    return elements[:n],values[:n]

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--embedding', action = 'store', type = 'string', dest = 'embedding_fname', help = 'Embedding File Path')
    parser.add_option('-n', '--number', action = 'store', type = 'int', dest = 'n', help = 'Number of most significant objects')
    parser.add_option('-k', action = 'store', type = 'int', dest = 'k', help = 'K value for the K-robust TSP tour')
    parser.add_option('-i', action = 'store', type = 'string', dest = 'knn_indices_file', help = 'KNN indices file')
    parser.add_option('-d', action = 'store', type = 'string', dest = 'knn_distances_file', help = 'KNN distances file')
    parser.add_option('-f', action = 'store', type = 'string', dest = 'info_file')
    parser.add_option('-c', action = 'store', type = 'string', dest = 'cluster_id_file', help = 'Cluster ID file')
    parser.add_option('-o', action = 'store', type = 'string', dest = 'option', help = 'Type of Embeddings - Movie|Words|People')

    (options, args) = parser.parse_args()
    print 'Loading embedding file...'
    elements, embedding = load_embedding(options.embedding_fname, options.n)
    print 'Loading KNN file...'
    knn_indices = pickle.load(open(options.knn_indices_file))
    knn_distances = pickle.load(open(options.knn_distances_file))
    print 'Constructing original TSP tour with nearest neighbor insertion algorithm...'
    tsp_tour = tsp.krobust_nn_tour_fast(options.n, options.k, embedding, knn_indices, knn_distances)

    file_type = options.option
    tsp_fname = 'nn.'+ file_type + '.tour.n=%d.k=%d.csv' % (options.n, options.k)
    fp = open(tsp_fname, 'w')
    final_tour = ','.join([str(ind) for ind in tsp_tour])
    fp.write(final_tour)
    fp.close()

    print 'Generating CSV file for the TSP tour...'
    csv_fname = 'nn.movies.tour.n=%d.k=%d.csv' % (options.n, options.k)
    tsp_seq = gen_tsp_csv.process_tsp(tsp_fname)
    if options.option == 'People':
        (prev_to_cur, cur_to_next) = gen_tsp_csv.process_dis(tsp_seq, options.embedding_fname)
    name_seq = [elements[ind] for ind in tsp_seq]
    rank_seq = tsp_seq
    cluster_ids = pickle.load(open(options.cluster_id_file, 'rb'))
    cluster_seq = [cluster_ids[ind] for ind in tsp_seq]
    n = len(tsp_seq)
    with open(csv_fname, 'wb') as res_file:
        csv_writer = csv.writer(res_file, delimiter = '\t')
        if options.option == 'People':
            line = [options.option, 'Rank', 'Cluster ID','Dis To Next']
            csv_writer.writerow(line)
            for i in range(n):
                line = [str(name_seq[i]), str(rank_seq[i]), str(cluster_seq[i]), str(cur_to_next[i])]
                csv_writer.writerow(line)
        else:
            line = [options.option,'Rank','Cluster ID']
            csv_writer.writerow(line)
            for i in range(n):
                line = [str(name_seq[i]), str(rank_seq[i]), str(cluster_seq[i])]
                csv_writer.writerow(line)
    print 'Done.'

    '''
    print 'Generating CSV file for the TSP tour...'
    csv_fname = 'nn.words.tour.n=%d.k=%d.csv' % (options.n, options.k)
    words_tour_fp = open(tsp_fname)
    words_in_tour = []
    for line in words_tour_fp:
        line = line.strip()
        words_in_tour = line.split(',')
        break
    tour_ranks, tour_embeddings = gen_words_csv.load_embeddings(words_in_tour, words, embedding)
    cluster_ids = gen_words_csv.get_cluster_info(words_in_tour, options.cluster_id_file)
    real_words_in_tour = [words[int(index)] for index in words_in_tour]
    gen_words_csv.gen_csv(real_words_in_tour, tour_ranks, cluster_ids, tour_embeddings, csv_fname)
    print 'Done.'
    '''