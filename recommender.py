import pyspark
from pyspark import SparkContext
sc = SparkContext("local", "BDA")

import json
import numpy as np
import sys

filename = sys.argv[1]

json_file = sc.textFile(filename)
records = json_file.map(lambda js: json.loads(js)).filter(lambda dic: 'reviewerID' in dic and 'asin' in dic and 'overall' in dic)

user_product_records = records.map(lambda dic: ((dic['reviewerID'], dic['asin']), dic['overall'])).groupByKey().map(lambda t: (t[0], list(t[1])))
user_product_last_rating = user_product_records.map(lambda t: (t[0], list(t[1])[-1]))
grouped_by_asin = user_product_last_rating.map(lambda t: (t[0][1], (t[0][0], t[1]))).groupByKey().map(lambda t: (t[0], list(t[1])))
min_users_filtered = grouped_by_asin.filter(lambda t: len(t[1]) > 24)
flat_records = min_users_filtered.flatMapValues(lambda t:t)
grouped_by_reviewer = flat_records.map(lambda t: (t[1][0], (t[0], t[1][1]))).groupByKey().map(lambda t: (t[0], list(t[1])))
min_items_filtered = grouped_by_reviewer.filter(lambda t: len(t[1]) > 4)

all_users = min_items_filtered.map(lambda t: t[0])
sorted_user_list = sorted(all_users.take(1000))

utility_matrix = min_items_filtered.flatMapValues(lambda t: t)
utility_matrix_by_item = utility_matrix.map(lambda t: (t[1][0], (t[0], t[1][1]))).groupByKey().map(lambda t: (t[0], list(t[1])))
all_items = utility_matrix_by_item.map(lambda t: t[0])



##Actual Recommender system
## Neigbourhood


def mean_center(user_rating_kvs):
    ratings_list = list(map(lambda t: t[1], user_rating_kvs))
    return list(map(lambda t: (t[0], (t[1] - np.mean(ratings_list))), user_rating_kvs))


def similarity_with_targets(other):
    other_item = other[0]
    other_user_rating_kvs = other[1]
    other_user_rating_kvs_mc = mean_center(other_user_rating_kvs)
    res = []
    other_dict = dict(other_user_rating_kvs_mc)
    targets = targeted_items_shared.value
    for (target, target_user_rating_kvs) in targets:
        sim_num = 0
        target_dict = dict(mean_center(target_user_rating_kvs))
        common_users = [k for k in other_dict if k in target_dict]
        if len(common_users) > 1:
            sim_num = sum(map(lambda cu: other_dict[cu] * target_dict[cu], common_users))
        ss_other = sum(map(lambda x: x * x, other_dict.values()))
        ss_target = sum(map(lambda x: x * x, target_dict.values()))
        sim_denom = np.sqrt([ss_other])[0] * np.sqrt([ss_target])[0]
        if sim_denom > 0 and sim_num > 0 and target != other_item:
            res.append((sim_num/sim_denom, target, other_user_rating_kvs))
    return res

# store targets as bc
items_to_be_predicted = eval(sys.argv[2])
#items_to_be_predicted = ['B00EZPXYP4', 'B00CTTEKJW']
targeted_items = utility_matrix_by_item.filter(lambda t: t[0] in items_to_be_predicted)
targeted_items_shared = sc.broadcast(targeted_items.take(1000))


neighbours_by_targets = utility_matrix_by_item.flatMap(lambda t: similarity_with_targets(t)).map(lambda t: (t[1], (t[0], t[2]))).groupByKey().map(lambda t: (t[0], list(t[1])))


#Predict unknown ratings for targets

def prepare_unknown_users(target_known_user_rating_kvs):
    all_distinct_users = all_users_shared.value
    target_known_user_ratings_dict = dict(target_known_user_rating_kvs)
    return list(filter(lambda user: user not in target_known_user_ratings_dict, all_distinct_users))

def redistribute(item, unknown_user_list, neighbours):
    res = []
    for user in unknown_user_list:
        res.append((item, user, neighbours))
    return res


def filter_zero_neighbours(unknown_user, neighbours):
    res = []
    for (sim, user_rating_kvs) in neighbours:
        user_rating_kvs_dict = dict(user_rating_kvs)
        if unknown_user in user_rating_kvs_dict:
            res.append((sim, user_rating_kvs_dict[unknown_user]))
    return res

def weighted_average(sim_rating_kvs):
    num = sum(map(lambda t: t[0]*t[1], sim_rating_kvs))
    den = sum(map(lambda t: t[0], sim_rating_kvs))
    return num/den

all_users_shared = sc.broadcast(all_users.take(1000))
targeted_items_with_unknown_ratings = targeted_items.map(lambda t: (t[0], prepare_unknown_users(t[1])))
targeted_items_with_unknown_ratings_neighbours = targeted_items_with_unknown_ratings.join(neighbours_by_targets)
target_predictable_user_weights = targeted_items_with_unknown_ratings_neighbours.flatMap(lambda t: redistribute(t[0], t[1][0], t[1][1])).map(lambda t: (t[0], t[1], filter_zero_neighbours(t[1], t[2]))).filter(lambda t: len(t[2]) > 1)
target_predictable_user_predicted_ratings = target_predictable_user_weights.map(lambda t: (t[0], t[1], weighted_average(t[2])))

res = target_predictable_user_predicted_ratings.take(1000)
print(res)
