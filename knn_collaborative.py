from __future__ import division
import sys
import itertools
from math import sqrt
import numpy as np
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
#from pyspark.mllib.recommendation import ALS
#/Users/jamesledoux/spark-1.6.1/bin/pyspark

#make these arguments so they work for other people
ratingsDir = "/Users/jamesledoux/Documents/BigData/netflixrecommender/ratings.dat"
moviesDir = "/Users/jamesledoux/Documents/Big Data/netflixrecommender/movies.dat"

conf = SparkConf().setAppName("MovieLensKNN").set("spark.executor.memory", "7g")
sc = SparkContext(conf=conf)

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

#mean squared distance between the movie ratings two users have in common
def get_distance(user1, user2):
    common_movies = 0
    distances = []
    for movie in user1:
        if movie in user2:
            distances.append((user1[movie] - user2[movie])**2)
            common_movies += 1
    if common_movies == 0:
        MSE = float("inf")    #nothing in common, so max out the distance
    else:
        MSE = np.mean(distances) + 5/common_movies   #MSE + penalty for having fewer common movies
    return MSE

def predicted_movies(knn, n_recs):
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
        movies[movie] = np.mean(movies[movie]) + len(movies[movie])/k  #rationale: if more neighbors saw it and liked it, it is probably a better recommendation
    recommendations = []
    for i in movies:
        recommendations.append({i: movies[i]})
    recommendations = sorted(movies, key=lambda x: movies[x], reverse = True)  #movies w/ best avg score rated highest
    return recommendations[0:n_recs]



#(person id, movie id, rating out of 5)
ratings = loadRatings(ratingsDir)

"""
Create dict of users and their movie ratings
Each user is a key. Each user's value is a dictionary, whose keys are movies and values are ratings.
{user: {movie: rating, movie: rating}, user: {movie: rating}, user: {}}
"""
users = {}
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
user_ratings = users[user_id]   # a dict of movie: rating pairs

for user in users:
    if user != user_id:   #if not the user you are finding neighbors for
        ratings = users[user]
        dist = {}
        dist[user] = (get_distance(ratings, user_ratings), ratings)
        neighbors.append(dist)

nearest_neighbors = sorted(neighbors, key=lambda x: x.values()[0][0])
knn = nearest_neighbors[0:k]

#get top 8 recommendations based on k-nearest neighbors
movies = dict(sc.textFile(moviesDir).map(parseMovie).collect())
#get the n top movies from your k nearest neighbors
n = 8   #make k and n command line args
recs = predicted_movies(knn, n)#[0:8]
for i in recs:
    print movies[i]

"""
eval:

set aside 3 ratings for 10 different users
see what nearest neighbors predict they will score these as. Get squared difference.
"""

"""
main:
choose k (no. of neighbors), user_id, and number of desired recommendations

predicted_movies(knn)[0:no_recs]

sc.stop()
