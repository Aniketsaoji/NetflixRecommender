from __future__ import division
import sys
import itertools
from math import sqrt
import numpy as np
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
#from pyspark.mllib.recommendation import ALS

"""
first:
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_45.jdk/Contents/Home
mvn -version

then, when in the folder you're running this in:
/Users/jamesledoux/spark-1.6.1/bin/pyspark

not sure why I need to do this first, but for now I need to re-do this for it to work
"""

"""
run command:
#/Users/jamesledoux/spark-1.6.1/bin/spark-submit python/MovieLensKNN.py /Users/jamesledoux/Downloads/tutorial/data/movielens/medium/ personalRatings.txt

make content-based recommendations using KNN clustering on the MovieLens data set
"""

#def get_ratings():
    #format: [user, movie, rating, timestamp]
    #return [line.strip().split('::') for line in open('ratings.dat')]

def parseRating(line):
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))


def parseMovie(line):
    fields = line.split("::")
    return int(fields[0]), fields[1]


def loadRatings(ratingsFile):
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


def get_distance(user1, user2):
    """
    returns MSE for common ratings between two users
    question: should I find a way to reward for a high number of common movies?
              more shared movies => a better neighbor for prediction one would think
    """
    common_movies = 0
    distances = []
    for movie in user1:
        if movie in user2:
            distances.append((user1[movie] - user2[movie])**2)
            common_movies += 1
    if common_movies == 0:
        MSE = float("inf")    #nothing in common, so max out the distance
    else:
        MSE = np.mean(distances) + 5/common_movies   #penalty for few common movies
    return MSE


def predicted_movies(knn):
    """
    for movie in knn rating history, get an adjusted mean rating and return these
    sorted greatest to least
    """
    movies = {}
    k = len(knn)
    for i in knn:
        for movie in i.values()[0][1]:
            if movie in movies:
                movies[movie].append(i.values()[0][1][movie])
            else:
                movies[movie] = [i.values()[0][1][movie]]
    #mean val + a reward for movies seen by multiple neighbors (max poss. 1 pt. boost)
    for movie in movies:
        movies[movie] = np.mean(movies[movie]) + len(movies[movie])/k
    recommendations = []
    for i in movies:
        recommendations.append({i: movies[i]})
    recommendations = sorted(movies, key=lambda x: movies[x], reverse = True)
    return recommendations

ratingsDir = "/Users/jamesledoux/Documents/Big Data/movielens/medium/ratings.dat"
moviesDir = "/Users/jamesledoux/Documents/Big Data/movielens/medium/movies.dat"
# set up environment
conf = SparkConf() \
  .setAppName("MovieLensKNN") \
  .set("spark.executor.memory", "2g")

sc = SparkContext(conf=conf)

#(person id, movie id, rating out of 5)
ratings = loadRatings(ratingsDir)
#movies = dict(sc.textFile(moviesDir).map(parseMovie).collect())

users = {}
#movie_dict = {}   #do I use this anywhere?

#create dict of users
#user: {movie: rating, movie: rating}, user: {movie: rating}, user: {}}
for (user, movie, rating) in ratings:
    if user not in users:
        users[user] = {}
        users[user][movie] = rating
    else:
        users[user][movie] = rating



user_id = 24   #a test user. parameterize this later.
k = 5  #also parameterize this

#[ { user_id: (distance, {movie: rating, movie: rating}), user_id: (dist, {movie: rating })}]
neighbors = []
user_ratings = users[user_id]

for user in users:
    if user != user_id:
        ratings = users[user]
        dist = {}
        dist[user] = (get_distance(ratings, user_ratings), ratings)
        neighbors.append(dist)

nearest_neighbors = sorted(neighbors, key=lambda x: x.values()[0][0])
knn = nearest_neighbors[0:k]

#get top 8 recommendations based on k-nearest neighbors
movies = dict(sc.textFile(moviesDir).map(parseMovie).collect())
recs = predicted_movies(knn)[0:8]
for i in recs:
    print movies[i]

for i in recs

"""
main:

choose k (no. of neighbors), user_id, and number of desired recommendations

predicted_movies(knn)[0:no_recs]

sc.stop()
