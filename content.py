from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
#imbdPY an api with better content data
"""
mid, 1: val, 2: val, 3:, val, 30: val, 45: val

sparse matrix w/ dummies for actors, genres, etc.
"""
#/Users/jamesledoux/spark-1.6.1/bin/pyspark

#step 1: create spark context
conf = SparkConf().setAppName("KMeans-Content").set("spark.executor.memory", "7g")
sc = SparkContext()


def parseRating(line):
    parts = line.strip().split("::")
    return (int(parts[0])-1, int(parts[1])-1, float(parts[2]))


#step 2: load in input file
path = "/Users/jamesledoux/Documents/BigData/netflixrecommender/movie_features_dataset.dat/"
data = MLUtils.loadLibSVMFile(sc, path)

labels = data.map(lambda x: x.label)
features = data.map(lambda x: x.features)


#normalize:
#http://spark.apache.org/docs/1.5.1/api/python/pyspark.mllib.html
scaler = StandardScaler(withMean = True, withStd = True).fit(features)  #data needs to be dense (zeros included)
#scaler = StandardScaler(withMean = False, withStd = True).fit(features)  #becomes dense if using withMean. may run out of memory locally

#convert data to dense vector to be normalized
data2 = labels.zip(scaler.transform(features.map(lambda x: Vectors.dense(x.toArray()))))
#data2 = labels.zip(scaler.transform(features))   #use this line if having memory issues

#(mid  vector)
numFeatures = data2.values().take(1)[0]
#data2.values().take(2)[0]  #to see what the data looks like

#split the data
train, val, test = data2.randomSplit([.8, .1, .1])

minError = float("inf")
bestModel = None
bestK = None
test_values = [5, 6, 7, 8, 9, 10, 11, 12]
"""
errors for [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25]: best is 10
[7127750.091446353, 7127750.414798906, 7127751.405619981, 7127754.433954623, 7127750.722505061, 7127838.947968733, 7127783.328832407, 7127750.988140488, 7127747.7471736595, 7127750.230260722, 7127754.734692583]

errors for [8, 9, 10, 11, 12, 13, 14, 15]:  best is 8
[7127746.282929602, 7127821.055026695, 7127750.747225445, 7127751.5867197495, 7127751.9117591, 7127756.4383017905, 7127754.529469332, 7127756.724856978]

for [5, 6, 7, 8, 9, 10, 11, 12]: best is 12
[7127750.287024926, 7127752.929726642, 7127746.997023219, 7127745.9775424525, 7127748.620281398, 7127751.659824361, 7127753.905078466, 7118664.342015211]
"""
error_storage = []

#k = 10

for i in test_values:
    model = KMeans.train(train.values(), i, maxIterations=10, runs=10, epsilon=.00001)
    error = model.computeCost(val.values())
    error_storage.append(error)
    print "model with " + str(i) + " clusters done"
    if error < minError:
        bestModel = model
        minError = error
        bestK = i

#now score model on the test data
error = bestModel.computeCost(test.values())


modelCenters = bestModel.clusterCenters
#get rdd of clusterid, movieid tuples  (predict after training)
trainingClusterLabels = train.map(lambda x: (bestModel.predict(x[1]), x[0]))
#recs for user:
#moviesLiked <- user liked
#for m in moviesLiked
#   label = use k means to predict the cluster for m
#get most frequent clusterLabel
#recommend movies from this cluster

path2 = "/Users/jamesledoux/Documents/BigData/netflixrecommender/ratings.dat"
ratings = sc.textFile(path2).map(parseRating)  #make the parse rating function
ratingsByUser = ratings.map(lambda x: (x[0], (x[1],x[2])))
ratingsByUser = ratingsByUser.groupByKey().map(lambda x: (x[0], list(x[1]))).collect()

user = ratingsByUser[0]
userHighRatings = [movieRating for movieRating in user[1] if movieRating[1] == 5]

singleRating = userHighRatings[0]
clusterId = model.predict(data2.lookup(singleRating[0])[0])
samplesInRelevantCluster = trainingClusterLabels.lookup(clusterId)

print len(samplesInRelevantCluster)
for i in samplesInRelevantCluster:
    print i

# looking at the results, it seems like almost everything is being sent to the same cluster

"""
knn:

for every person
    for every movie they both rated
        distance = sum ( difference in ratings )^2  /  number of ratings in common
"""
sc.stop()