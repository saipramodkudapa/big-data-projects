##########################################################################
## Simulator.py  v0.2
##
## Implements two versions of a multi-level sampler:
##
## 1) Traditional 3 step process
## 2) Streaming process using hashing
##
## Original Code written by H. Andrew Schwartz
## for SBU's Big Data Analytics Course
## Spring 2020
##
## Student Name: Sai Pramod Kudapa
## Student ID: 112686280

##Data Science Imports:
import numpy as np
import mmh3
from random import random
from random import shuffle

##IO, Process Imports:
import sys
from pprint import pprint


##########################################################################
##########################################################################
# Task 1.A Typical non-streaming multi-level sampler
import numpy as np
import mmh3
import datetime

# helper method to compute gcd to simplify fraction
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# helper methods to determine bucket size and no of buckets for a percentage
# a is bucket size and b is no of buckets
def buckets(p):
    a = float(p * 100)
    b = 100.0
    while a.is_integer() is False:
        a *= 10
        b *= 10
    g = int(gcd(a, b))
    if b > 100:
        return int(a/g), int(b/g)
    return int(a), int(b)


def typicalSampler(filename, percent=.01, sample_col=0):
    csv_file1 = open(filename, "r")
    csv_file2 = open(filename, "r")
    uuids = set()
    for row in csv_file1:
        attrs = row.split(',')
        user_id = attrs[2]
        uuids.add(user_id)

    rand_uuids_size = int(float(uuids.__len__()) * percent)
    random_no = np.random.randint(0, (uuids.__len__() - rand_uuids_size))
    random_sample_uuids = list(uuids)[random_no: random_no + rand_uuids_size]

    mean, m2, c = 0.0, 0.0, 0
    for row in csv_file2:
        attrs = row.split(',')
        user_id = attrs[sample_col]
        if user_id in random_sample_uuids:
            amount = float(attrs[3])
            c += 1
            delta = amount - mean
            mean += delta/c
            m2 += delta * (amount - mean)

    variance = m2/c
    sd = np.sqrt([variance])[0]
    return mean, sd


##########################################################################
##########################################################################
# Task 1.B Streaming multi-level sampler
def streamSampler(csv_file, percent=.01, sample_col=0):
    mean, m2, c = 0.0, 0.0, 0
    a, b = buckets(percent)
    for row in csv_file:
        attrs = row.split(',')
        user_id = attrs[sample_col]
        if mmh3.hash(user_id) % b < a:
            amount = float(attrs[3])
            c += 1
            diff = amount - mean
            mean += diff / c
            m2 += diff * (amount - mean)
    variance = m2/c
    sd = np.sqrt([variance])[0]
    return mean, sd


##########################################################################
##########################################################################
# Task 1.C Timing
files = ['transactions_small.csv', 'transactions_medium.csv', 'transactions_large.csv']
percents = [.02, .005]

if __name__ == "__main__":

    for perc in percents:
        print("\nPercentage: %.4f\n==================" % perc)
        for f in files:
            print("\nFile: ", f)
            t1 = datetime.datetime.now()
            typicalSamplerResult = typicalSampler(f, perc, 2)
            t2 = datetime.datetime.now()
            print("  Typical Sampler: ", typicalSamplerResult)
            print("  Time elapsed for Typical Sampler: ", (t2-t1).total_seconds()*1000)
            fstream = open(f, "r")
            t3 = datetime.datetime.now()
            streamSamplerResult = streamSampler(fstream, perc, 2)
            t4 = datetime.datetime.now()
            print("  Stream Sampler:  ", streamSamplerResult)
            print("  Time elapsed for Stream Sampler: ", (t4-t3).total_seconds()*1000)

