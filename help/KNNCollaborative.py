#KNNCollaborative
import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import SparseVector
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import  Rating
import random
import scipy.sparse as sps
from pyspark.mllib.linalg import Vectors

def vectorize(ratings, numMovies):
    return ratings.map(lambda x : (x[0]-1, (x[1]-1, x[2])))\
           .groupByKey()\
           .mapValues(lambda x: SparseVector(numMovies,x))

def parseRating (line):
    # userID:: movieID:: rating::timestamp
    parts = line.strip().split("::")
    return long(parts[3])%10 , (int (parts[0]),int(parts[1]),float(parts[2]) )

def loadRatings (sc, MLDir):
    return sc.textFile(join (MLDir,"ratings.dat")).map(parseRating)

def sqrdDiff(user, testuser):
    print(type(user[1]),type(testuser[1]))
    user = user[1]
    testuser = testuser[1]
    indices = user.indices
    print("*****", indices)
    error=0.0
    count=0.0
    for index in testuser.indices:
        print("*INDEX*:", type(index))
        index = int(index)
        error+=(testuser[index]-user[index])**2
        count+=1
    print(error/count)
    return error/count

if __name__=="__main__":
    if(len(sys.argv)<1):
        print("Usage: spark-submit KNNCollaborative.py ./MovieLensDir")
        sys.exit(1)

# Parse
MLDir = sys.argv[1].strip()

# Initialize Spark Context
conf = SparkConf().setAppName("KNNColloborative").set("spark.executor.memory","2g")
sc = SparkContext(conf = conf)

# Load data
ratings = loadRatings(sc, MLDir) #load all data into ratings RDD
print("Type of ratings obj", type(ratings)) # RDD - PipelinedRDD
print("Count of ratings:", ratings.count())
print("Sample ratings:", ratings.take(1))
numUsers = ratings.values().map(lambda x: x[0]).max()+1
numMovies = ratings.values().map(lambda x: x[1]).max()+1
ratings = vectorize(ratings.values(), numMovies) # vectorize (userId, SparseVector)
training, test = ratings.randomSplit([.95,0.5])

testuser = test.take(1)[0]
print(training.flatMap(lambda user: (sqrdDiff(user,testuser),user[0])).takeOrdered(10))
sc.stop()
