import pyspark
from pyspark import SparkContext
sc = SparkContext("local", "BDA")

import json
import re
from operator import add
import sys

filename = sys.argv[1]
json_file = sc.textFile(filename)
records = json_file.map(lambda js: json.loads(js))


def reg_filter(word):
    return True if re.match(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', word.lower()) else False


# Find top 1k common words across reviews using word count algo
reviews = records.filter(lambda review: 'reviewText' in review)
review_words = reviews.map(lambda review: (review['reviewText'])).flatMap(lambda review: review.split())
filtered_review_words = review_words.filter(lambda word: reg_filter(word))
word_freq_kvs = filtered_review_words.map(lambda word: (word.lower(), 1)).reduceByKey(add)
onek_common_words = word_freq_kvs.sortBy(lambda t: t[1], False).map(lambda t: t[0]).take(1000)
onek_common_words_shared = sc.broadcast(onek_common_words)


# Find relative frequency of 1k words
# Prepare data for 1k linear regressions grouped by common word
def relative_frequencies(review_text):
    onek_common_words = onek_common_words_shared.value
    words = review_text.split()
    qualified_words = []
    for w in words:
        if reg_filter(w):
            qualified_words.append(w.lower())
    if len(qualified_words) > 0:
        onek_rel_freqs = []
        for cw in onek_common_words:
            onek_rel_freqs.append((cw, qualified_words.count(cw)/len(qualified_words)))
        return onek_rel_freqs
    else:
        return list(zip(onek_common_words, [0]*1000))

review_with_rel_freqs = reviews.map(lambda review: ((review['overall'], int(review['verified'])), relative_frequencies(review['reviewText'])))
review_with_rel_freqs_flattened = review_with_rel_freqs.flatMapValues(lambda t: t).map(lambda t: (t[1][0], (t[1][1], t[0][0], t[0][1])))
review_with_rel_freqs_grouped_by_word = review_with_rel_freqs_flattened.groupByKey().map(lambda t: (t[0], list(t[1])))

import numpy as np
from scipy import stats as ss


def find_betas_pvalues(freq_rating_verified_list):
    # Find z values
    relative_frequencies = [t[0] for t in freq_rating_verified_list]
    ratings = [t[1] for t in freq_rating_verified_list]
    rf_mean = np.mean(relative_frequencies)
    rf_std = np.std(relative_frequencies)
    rat_mean = np.mean(ratings)
    rat_std = np.std(ratings)
    relative_frequency_zvalues = [(t-rf_mean)/rf_std for t in relative_frequencies]
    ratings_zvalues = [(t-rat_mean)/rat_std for t in ratings]

    N = len(ratings)
    X = np.array(relative_frequency_zvalues)
    row_to_be_added = np.full((1, N), 1)
    X_N = np.transpose(np.vstack((X, row_to_be_added)))
    Y = np.transpose(np.array(ratings_zvalues))
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_N), X_N)), np.transpose(X_N)), Y)

    y_pred = beta[1]+beta[0]*X
    y = np.array(ratings_zvalues)
    rss = np.sum(np.square(y_pred-y))
    m = len(beta) - 1
    df = N-(m+1)
    s_square = rss/df

    deno_sum = np.sum(np.square(relative_frequency_zvalues))
    standard_error = s_square/deno_sum
    t_value = beta[0]/np.sqrt([standard_error])[0]
    t_cdf_value = ss.t.cdf(t_value, df)
    p_value = t_cdf_value if t_cdf_value < 0.5 else (1-t_cdf_value)
    return (beta[0], p_value*1000)


def find_betas_pvalues_controlled(freq_rating_verified_list):
    # Find z values
    relative_frequencies = [t[0] for t in freq_rating_verified_list]
    ratings = [t[1] for t in freq_rating_verified_list]
    verified = [t[2] for t in freq_rating_verified_list]
    rf_mean = np.mean(relative_frequencies)
    rf_std = np.std(relative_frequencies)
    rat_mean = np.mean(ratings)
    rat_std = np.std(ratings)
    verified_mean = np.mean(verified)
    verified_std = np.std(verified)
    relative_frequency_zvalues = [(t-rf_mean)/rf_std for t in relative_frequencies]
    ratings_zvalues = [(t-rat_mean)/rat_std for t in ratings]
    verified_zvalues = [(t-verified_mean)/verified_std for t in verified]

    N = len(ratings)
    X = np.array(relative_frequency_zvalues)
    X2 = np.array(verified_zvalues)
    row_to_be_added = np.full((1, N), 1)
    X_N = np.vstack(np.vstack((X, row_to_be_added)))
    X_N_N = np.transpose(np.vstack((X_N, verified_zvalues)))
    Y = np.transpose(np.array(ratings_zvalues))
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_N_N),X_N_N)),np.transpose(X_N_N)),Y)

    y_pred = beta[1]+beta[0]*X+beta[2]*X2
    y = np.array(ratings_zvalues)
    rss = np.sum(np.square(y_pred-y))
    m = len(beta) - 1
    df = N-(m+1)
    s_square = rss/df

    deno_sum = np.sum(np.square(relative_frequency_zvalues))
    standard_error = s_square/deno_sum
    t_value = beta[0]/np.sqrt([standard_error])[0]
    t_cdf_value = ss.t.cdf(t_value, df)
    p_value = t_cdf_value if t_cdf_value < 0.5 else (1-t_cdf_value)
    return (beta[0], p_value*1000)


betas_p_values = review_with_rel_freqs_grouped_by_word.map(lambda t: (t[0], find_betas_pvalues(t[1])))
t20_positive_correlated_words = betas_p_values.sortBy(lambda d: d[1][0], False).take(20)
t20_negative_correlated_words = betas_p_values.sortBy(lambda d: d[1][0], True).take(20)

betas_p_values_controlled = review_with_rel_freqs_grouped_by_word.map(lambda t: (t[0], find_betas_pvalues_controlled(t[1])))
t20_positive_correlated_words_controlled = betas_p_values_controlled.sortBy(lambda d: d[1][0], False).take(20)
t20_negative_correlated_words_controlled = betas_p_values_controlled.sortBy(lambda d: d[1][0], True).take(20)

print("The top 20 word positively correlated with rating")
print(t20_positive_correlated_words)

print("The top 20 word negatively correlated  with rating")
print(t20_negative_correlated_words)

print("The top 20 words positively related to rating, controlling for verified")
print(t20_positive_correlated_words_controlled)

print("The top 20 words negatively related to rating, controlling for verified")
print(t20_negative_correlated_words_controlled)
