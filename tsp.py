import random
import sys
import pickle
import csv
import random
import numpy as np
import copy
import itertools
from sets import Set
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances

# get location in the TSP tour -- because the tour is circular
def get_loc(i, tsp_len):
    if i < tsp_len and i >= 0:
        return i
    elif i >= tsp_len:
        return i - tsp_len
    else:
        return i + tsp_len

def global_local_opt(tsp_tour, window_size, embedding, k):
    n = len(tsp_tour)
    for start_loc in range(n):
        tsp_tour = local_opt(tsp_tour, window_size, start_loc, embedding, k)

# locally swap people using a fairly small window size
def local_opt(tsp_tour, window_size, start_loc, embedding, k):
    tsp_len = len(tsp_tour)
    window = [get_loc(start_loc + i, tsp_len) for i in range(window_size)]
    window_perms = itertools.permutations(window)
    best_dis = 1e9
    best_tsp_tour = copy.deepcopy(tsp_tour)
    for perm in window_perms:
        new_tsp_tour = copy.deepcopy(tsp_tour)
        for i, index in enumerate(window):
            new_tsp_tour[index] = tsp_tour[ perm[i] ]
        new_dis = 0.0
        for index in window:
            new_dis = new_dis + krobust_distance(new_tsp_tour, index, k, embedding)
        if new_dis < best_dis:
            best_dis = new_dis
            best_tsp_tour = copy.deepcopy(new_tsp_tour)
    return best_tsp_tour

#K-robust distance if we put the v-th vertex in the position of the i-th vertex in current TSP tour
def point_krobust_distance(tsp_tour, v, i, k, n_vertices, dis_mat = None):
    tot_dis = 0.0
    if dis_mat is None:
        for j in range(1, k + 1):
            tot_dis = tot_dis + euclidean_distances(tsp_tour[v], tsp_tour[get_loc(i + j, n_vertices)])
    else:
        for j in range(1, k + 1):
            tot_dis = tot_dis + dis_mat[ tsp_tour[v] ][tsp_tour[get_loc(i + j, n_vertices)]]
    return tot_dis

#K-robust distance for a whole tour, given a distance matrix
def tour_krobust_distance(tsp_tour, k, dis_mat):
    tot_dis = 0.0
    n_vertices = len(tsp_tour)
    for i in range(n_vertices):
        tot_dis = tot_dis + point_krobust_distance(tsp_tour, i, i, k, n_vertices, dis_mat)
    # or tot_dis / 2?
    return tot_dis

#tour[0...i-1] + reverse(tour[i...x])+tour[x+1...n-1]
def get_2opt_tour(tour, i, x):
    n_vertices = len(tour)
    new_tour = tour[:i]
    new_tour.extend([tour[j] for j in range(x, i - 1, -1)])
    new_tour.extend(tour[(x+1):])
    return new_tour

# performing the naive two opt algorithm might be time consuming, thus we use K-d tree to limit the number of examined nodes here
def two_opt(tsp_tour, k, embedding, dis_mat):
    is_improved = True
    n_vertices = len(tsp_tour)

    # When swapping vertices (u, v), assume w is u's neighbor in the TSP tour.
    # We need to ensure that v is one of vertex w's K nearest neighbors, to ensure the local similarity in the TSP tour
    # Here, we choose K to be 5% * N (the number of vertices)
    nn_count = int(0.08 * len(tsp_tour))
    kd_tree = NearestNeighbors(n_neighbors = nn_count, algorithm = 'kd_tree').fit(embedding)
    distances, indices = kd_tree.kneighbors(embedding)
    iter_round = 0
    prev_best_distance = 1e10

    while is_improved:
        iter_round = iter_round + 1
        print 'Iteration round: ', iter_round
        is_improved = False
        best_distance = tour_krobust_distance(tsp_tour, k, dis_mat)
        print 'Best distance for now: ', best_distance
        if prev_best_distance - best_distance < 1e-4:
            break
        prev_best_distance = best_distance

        # mapping from index in embedding to index in TSP tour
        em_to_tour = {em_index: tsp_index for tsp_index, em_index in enumerate(tsp_tour)}

        # vertex #1: i
        for i in range(n_vertices - 1):
            if is_improved:
                break
            # vertex #2: any vertex which is very close to tsp_tour[i + 1]
            for index in indices[ tsp_tour[i + 1] ]:
                x = em_to_tour[index]
            #for x in range(i + 1, n_vertices):
                if x <= i or x - i >= n_vertices - 2 * k:
                   continue
                if is_improved:
                    break
                # if i and x are within distance 2k of each other, simply calculate the cost of the new tour near i and x
                if x - i <= 2 * k:
                    # cut a part (from i -> x) of the TSP tour
                    temp_tour = [tsp_tour[get_loc(temp_loc, n_vertices)] for temp_loc in range(i - k, x + k + 1)]
                    new_tour = get_2opt_tour(temp_tour, k, k + x - i)
                    old_distance = tour_krobust_distance(temp_tour, k, dis_mat)
                    new_distance = tour_krobust_distance(new_tour, k, dis_mat)
                    if new_distance < old_distance:
                        # print temp_tour, new_tour, x, i
                        #print new_distance, old_distance
                        is_improved = True
                        new_tour = get_2opt_tour(tsp_tour, i, x)
                        tsp_tour = copy.deepcopy(new_tour)
                        best_distance = tour_krobust_distance(tsp_tour, k, dis_mat)
                        # print 'Type I improvement'
                    continue
                #swap i, x only affects a limited set of points in the tour
                old_distance = 0.0
                new_distance = 0.0

                for j in range(i, i + k):
                    # loc in the swapped tour
                    loc = get_loc(j, n_vertices)
                    for l in range(1, i + k - j + 1):
                        # print 'a', loc, get_loc(x + l, n_vertices)
                        # print 'a', loc, get_loc(i - l, n_vertices)
                        new_distance = new_distance + dis_mat[tsp_tour[loc]][tsp_tour[get_loc(x + l, n_vertices)]]
                        old_distance = old_distance + dis_mat[tsp_tour[loc]][tsp_tour[get_loc(i - l, n_vertices)]]

                for j in range(x, x - k, -1):
                    loc = get_loc(j, n_vertices)
                    #print '!!'
                    #print j
                    #print range(1, j+k-x+1)
                    for l in range(1, j + k - x + 1):
                        # print 'b', loc, get_loc(x + l, n_vertices)
                        # print 'b', loc, get_loc(i - l, n_vertices)
                        new_distance = new_distance + dis_mat[tsp_tour[loc]][tsp_tour[get_loc(i - l, n_vertices)]]
                        old_distance = old_distance + dis_mat[tsp_tour[loc]][tsp_tour[get_loc(x + l, n_vertices)]]

                if new_distance < old_distance:
                    # print i, x, old_distance, new_distance
                    # print 'Type II improvement'
                    is_improved = True
                    tsp_tour = get_2opt_tour(tsp_tour, i, x)
                    best_distance = tour_krobust_distance(tsp_tour, k, dis_mat)
    return tsp_tour

def random_perm(n):
    a = [i for i in range(n)]
    random.shuffle(a)
    return a

def read_data(kneighbors_file):
    n = 0
    with open(kneighbors_file) as kneighbors:
        for line_num, line in enumerate(kneighbors):
            n = n + 1
            line_split = line.strip().split(' ')
            for i, num in enumerate(line_split):
                if i == 0:
                    u = int(num)
                #link to itself
                elif i <= 2:
                    pass
                elif i % 2 == 1:
                    neighbors_id[u][i / 2 - 1] = int(num)
                    #neighbors_dis[line_num][i / 2 - 1] = float(line_split[i + 1])
    return n

#get K-robust distance, if we insert people[cur_ln] before people[index]
def get_krobust_distance(k, embedding, tsp_tour, index, cur_ln):
    ret = 0.0
    tsp_len = len(tsp_tour)

    for i in range(index - k, index + k):
        i = ((i % tsp_len) + tsp_len) % tsp_len
        ret += get_dis_by_ln(embedding, tsp_tour[i], cur_ln)
    return ret

# K-robust distance "near" insert_loc
def krobust_distance(tsp_tour, insert_loc, k, dis_mat = None):
    krobust_dis = 0.0
    tsp_len = len(tsp_tour)
    cur_index = tsp_tour[insert_loc]
    for i in range(-k, k + 1):
        neighbor_index = tsp_tour[get_loc(insert_loc + i, tsp_len)]
        krobust_dis = krobust_dis + dis_mat[cur_index][neighbor_index]
    return krobust_dis

# sum up the distance in a window size of 2k + 1
def local_krobust_distance(tsp_tour, insert_loc, k, dis_mat = None):
    tot_krobust_dis = 0.0
    for i in range(insert_loc - k, insert_loc + k + 1):
        tot_krobust_dis = tot_krobust_dis + krobust_distance(tsp_tour, get_loc(i, len(tsp_tour)), k, dis_mat)
    return tot_krobust_dis

# insert a new node at the start of the TSP tour
# this calculation is actually for a one-way trip, not a circular tour
def krobust_distance_new_node(tsp_tour, k, new_node_index, dis_mat = None, embeddings = None):
    tot_krobust_dis = 0.0

    # use embeddings
    if dis_mat is None:
        for i in range(0, min(k, len(tsp_tour)) ):
            tot_krobust_dis = tot_krobust_dis + distance.euclidean(embeddings[new_node_index], embeddings[tsp_tour[i]])
    else:
        for i in range(0, min(k, len(tsp_tour)) ):
            tot_krobust_dis = tot_krobust_dis + dis_mat[new_node_index][tsp_tour[i]]
    return tot_krobust_dis

# construct an original tour with nearest neighbor insertion method
# an improved algorithm, which utilizes the pre-calculated K nearest neighbors for each vertex
def krobust_nn_tour_fast(n, k, embedding, knn_indices, knn_distances):
    tsp_tour = [0]
    vertices = [i for i in range(1, n)]
    used = {i: False for i in range(n)}
    used[0] = True
    cur_vertex = 0

    for i in range(n - 1):
        if i % 100 == 0:
            print i
        # insert the vertex with minimum tour cost
        min_dis = 1e9
        insert_vertex = -1
        if i >= k - 1:
            # for the first k vertices in the TSP tour: mapping from vertex index i to the distance between i and this vertex
            ind_to_dis = [{} for j in range(k)]
            for j in range(k):
                for knn_index, knn_dis in zip(knn_indices[tsp_tour[j]], knn_distances[tsp_tour[j]]):
                    ind_to_dis[j][knn_index] = knn_dis
            #hit_count = 0
            #miss_count = 0
            for ind, vertex in enumerate(knn_indices[cur_vertex]):
                if vertex >= n or used[vertex] is True:
                    continue
                new_node_ind_to_dis = {}
                for knn_index, knn_dis in zip(knn_indices[vertex], knn_distances[vertex]):
                    new_node_ind_to_dis[knn_index] = knn_dis
                new_dis = 0.0
                for l in range(k):
                    if vertex in ind_to_dis[l]:
                        new_dis = new_dis + ind_to_dis[l][vertex]
                    elif tsp_tour[l] in new_node_ind_to_dis:
                        new_dis = new_dis + new_node_ind_to_dis[tsp_tour[l]]
                    else:
                        new_dis = new_dis + distance.euclidean(embedding[vertex], embedding[tsp_tour[l]])
                if new_dis < min_dis:
                    min_dis = new_dis
                    insert_vertex = vertex
        # insert the first k vertices / vertices that are not in the K nearest neighbors
        if insert_vertex == -1:
            # print 'missed.'
            # insert the vertex with minimum tour cost
            for vertex in vertices:
                if used[vertex] is True:
                    continue
                new_dis = krobust_distance_new_node(tsp_tour, k, vertex, embeddings = embedding)
                if new_dis < min_dis:
                    min_dis = new_dis
                    insert_vertex = vertex

        used[insert_vertex] = True
        tsp_tour.insert(0, insert_vertex)
        cur_vertex = insert_vertex
    return tsp_tour

# construct an original tour with nearest neighbor insertion method
# an naive O(n^2) algorithm
def krobust_nn_tour(n, k, embedding):
    tsp_tour = [0]
    #dis_mat = euclidean_distances(embedding[:n])
    vertices = [i for i in range(1, n)]

    for i in range(n - 1):
        # insert the vertex with minimum tour cost
        min_dis = 1e9
        insert_vertex = 0
        insert_index = 0
        for index, vertex in enumerate(vertices):
            new_dis = krobust_distance_new_node(tsp_tour, k, vertex, dis_mat = None, embeddings = embedding)
            if new_dis < min_dis:
                min_dis = new_dis
                insert_index = index
                insert_vertex = vertex
        vertices.pop(insert_index)
        tsp_tour.insert(0, insert_vertex)
        if i % 100 == 0:
            print i
    return tsp_tour

# build a high quality original tour
def krobust_topk_tour(n, k, embedding, distance_measure, order, is_local_opt = 0):
    tsp_tour = []
    dis_mat = euclidean_distances(embedding[:n])

    # insert each vertex in the given order
    for i in range(n):
        min_dis = 1e9
        insert_loc = 0
        for j in range(i):
            new_tsp_tour = copy.deepcopy(tsp_tour)
            new_tsp_tour.insert(j, order[i])
            new_dis = distance_measure(new_tsp_tour, j, k, dis_mat)
            if new_dis < min_dis:
                min_dis = new_dis
                insert_loc = j
        tsp_tour.insert(insert_loc, order[i])
        if (is_local_opt == 1):
            window_size = 5
            # need to check if the tour really changes
            local_opt(tsp_tour, window_size, i, embedding, k)
    return tsp_tour

def get_dis_by_ln(embedding, ln1, ln2):
    return distance.euclidean(embedding[ln1], embedding[ln2])

class Node(object):

    def __init__(self, data = None, prev_node = None, next_node = None):
        self.data = data
        self.prev_node = prev_node
        self.next_node = next_node

    def get_data(self):
        return self.data
    def get_prev(self):
        return self.prev_node
    def get_next(self):
        return self.next_node
    def set_prev(self, new_prev):
        self.prev_node = new_prev
    def set_next(self, new_next):
        self.next_node = new_next

def insert(new_node, loc):
    loc.next_node.prev_node = new_node
    new_node.next_node = loc.next_node
    new_node.prev_node = loc
    loc.next_node = new_node

def print_tour(head, fp):
    res = []
    res.append(head.get_data())
    iter = head.next_node
    while iter.get_data() != head.get_data():
        res.append(iter.get_data())
        iter = iter.next_node
    tour = ','.join(str(id) for id in res)
    fp.write(tour)
    fp.write('\n')

#get K-robust distance in the linked list, if we insert people[cur_ln] right after cur_node
def list_get_krobust_distance(k, embedding, cur_node, cur_ln):
    ret = 0.0
    iter = cur_node

    for i in range(k):
        ret += get_dis_by_ln(embedding, iter.get_data(), cur_ln)
        iter = iter.prev_node

    iter = cur_node.next_node
    for i in range(k):
        ret += get_dis_by_ln(embedding, iter.get_data(), cur_ln)
        iter = iter.next_node
    return ret

# return vertices index in descending order of significance
def descending_order(n):
    return range(0, n)

# return vertices index in ascending order of significance
def ascending_order(n):
    return range(n - 1, -1, -1)

# return vertices index in random order of significance
def random_order(n):
    return np.random.permutation(n)

# TODO: support external nearest neighbors file
# init with a small TSP tour
def large_krobust_tour(k, init_tsp_tour_file, people_rank_file, embedding_file, debug, output_file):
    fp = open(output_file, 'w')
    init_tsp_tour = []
    with open(init_tsp_tour_file, 'rb') as tsp_tour_reader:
        for line in tsp_tour_reader:
            line = line.strip()
            init_tsp_tour.extend([int(index) for index in line.split(',')])
    #if debug:
    #    print 'init tsp: ', init_tsp_tour

    print 'Loading embedding file...'
    embedding = pickle.load(open(embedding_file, 'rb'))
    nbrs = NearestNeighbors(n_neighbors = 251, algorithm = 'ball_tree').fit(embedding)
    print 'Load complete.'
    #distances, neighbors = nbrs.kneighbors(embedding)

    rank_to_ln = {}
    tsp_tour = []
    ln_to_node = {}
    init_n = len(init_tsp_tour)
    threshold = 6000
    head = None
    tail = None
    iter = None
    tsp_len = 0

    with open(people_rank_file, 'rb') as people_rank:
        reader = csv.DictReader(people_rank, delimiter = ',')
        for cur_ln, row in enumerate(reader):
            min_dis = 1e9
            ins_loc = None

            if cur_ln < init_n:
                #first people
                if tsp_len == 0:
                    head = Node(init_tsp_tour[cur_ln], None, None)
                    head.set_prev(head)
                    head.set_next(head)
                    ln_to_node[ init_tsp_tour[cur_ln] ] = head
                    tail = head
                else:
                    new_node = Node(init_tsp_tour[cur_ln], None, None)
                    ln_to_node[ init_tsp_tour[cur_ln] ] = new_node
                    insert(new_node, tail)
                    tail = new_node
                tsp_len = tsp_len + 1
                continue

            #insert in a proper position, for the first $threshold$ count of most significant people
            elif cur_ln <= threshold:
                iter = head
                for i in range(tsp_len):
                    new_dis = list_get_krobust_distance(k, embedding, iter, cur_ln)
                    #new_dis = get_dis_by_ln(embedding, tsp_tour[i], cur_ln) + get_dis_by_ln(embedding, tsp_tour[i - 1], cur_ln)
                    if new_dis < min_dis:
                        min_dis = new_dis
                        ins_loc = iter
                    iter = iter.next_node
                tsp_len = tsp_len + 1
                print tsp_len
            #for the rest: choose from several knn to insert
            else:
                distances, neighbors = nbrs.kneighbors(embedding[cur_ln])
                for neighbor in neighbors[0]:
                    if neighbor in ln_to_node:
                        ins_loc = ln_to_node[neighbor]
                        break
                if ins_loc is not None:
                    tsp_len = tsp_len + 1

            if ins_loc is None:
                pass
            else:
                new_node = Node(cur_ln, None, None)
                ln_to_node[cur_ln] = new_node
                insert(new_node, ins_loc)
                if debug and cur_ln % 1000 == 0:
                    print_tour(head, fp)
    print_tour(head, fp)


