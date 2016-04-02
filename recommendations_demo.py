from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkConf, SparkContext
from os.path import join
import sys


"""
kmeans(data, k, ...)

ratings
uid, movid, rating, timestamp

make a matrix:    cols movies, rows users


"""
if(len(sys.argv) != 2):
    print "usage: /sparkPath/bin/spark-submit  name.py  movieDirectory"

conf = SparkConf().setAppName("KMeans Collaborative")
#sc = SparkContext(conf)
movieLensHomeDir = sys.argv[1]   # passed as argument
#movieLensHomeDir = "/Users/jamesledoux/Documents/Big\ Data/movielens/medium/"
sc =SparkContext()

def parseRating(line):
    #uid::movieID::rating::timestamp
    parts = line.strip().split("::")
    return long(parts[3])%10, (int(parts[0])-1, int(parts[1])-1, float(parts[2]))  #parentheses probably wrong here

def loadRatings(sc, MLDir):
    return sc.textFile(join(MLDir, "ratings.dat")).map(parseRating)

def vectorize(ratings, numMovies):
    return ratings.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(lambda x: SparseVector(numMovies, x))

ratings = loadRatings(sc, movieLensHomeDir)
print "type of ratings obj: ", type(ratings)
print "count of ratings: ", ratings.count()
print "sample rating: ", ratings.take(1)

#ratings RDD:  (time stamp, (uid, mid, ratings) )
#num of users  (userid, movieid, ratings)
numUsers = ratings.values().map(lambda x: x[0]).max()+1
numMovies = ratings.values().map(lambda x: x[1]).max()+1

# ntransform into sparse vectors
"""
represent as sparse vector to save on space

three cols: id, vector size, movie:rating dictionary
size needed so u know how many zeros are in the mat?
uid, size of vector *around 3,000), (movie:rating, movie:rating, movie:rating))
"""


ratingsSV = vectorize( ratings.values(), numMovies)
print "RatingsSV Type:", type(ratingsSV)
print "RatingsSV Count:", ratingsSV.count()


#next: include a validation set. train on train, get errors on validation
#then: score == RMSE on the test set
minError = float("inf")
bestModel = None
bestK = None
test_values = [10, 20, 50, 100, 200]
error_storage = []
for i in test_values:
    model = KMeans.train(ratingsSV.values(), i, maxIterations = 20, runs = 10)
    #test this error on the validation set once we make that
    error = model.computeCost(ratingsSV.values())
    error_storage.append(error)
    if error < minError:
        bestModel = model
        minError = error
        bestK = i

"""
#save model once you have the best version
#model.save(sc, "KMeansModelCollaborative")
#model = KMeansModel.load(sc, "KMeansModelCollaborative")

#take one sample   the [0] is because this will be returned as a list -- not an index
user = ratingsSV(1)[0] #take a sample of 1 from the data set (use test data when doing this)
label = model.predict(user)   #outputs which cluster this user belongs to
# ==> a cluster id between 1 and k
clusterCenters = model.clusterCenters     #a list of centers (len == k)
clusterCenters[0] #len == total num of movies, each obs == avg rating for people in this group
movieID = 4
print "predicted value: ", clusterCenters[label][movieId]
"""




#for i in range(10, 20):
# ratingsSV = > RDD where each item is ( userID , SparseVector)

##    if error < minError:
##        bestModel = model
##        minError = error
##        bestK = i
user = ratingsSV.values().take(1)[0] #take a sample of 1 from the data set (use test data when doing this)
print "Type user:", type(user)
print "User:", user

label = model.predict(user)   #outputs which cluster this user belongs to
clusterCenters = model.clusterCenters     #a list of centers (len == k)
#clusterCenters[0] #len == total num of movies, each obs == avg rating for people in this group
print "Len Cluster Centers:", len(clusterCenters)
movieID = 4

print "predicted value: ", clusterCenters[label][movieID]

sc.stop()
