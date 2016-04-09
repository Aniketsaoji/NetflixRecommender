from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors

#/Users/jamesledoux/spark-1.6.1/bin/pyspark

conf = SparkConf().setAppName("KMeans-Content").set("spark.executor.memory", "7g")
sc = SparkContext()

def parseRating(line):
    parts = line.strip().split("::")
    return (int(parts[0])-1, int(parts[1])-1, float(parts[2]))


#load in input file
path = "/Users/jamesledoux/Documents/BigData/netflixrecommender/movie_features_dataset.dat/"
data = MLUtils.loadLibSVMFile(sc, path)

labels = data.map(lambda x: x.label)
features = data.map(lambda x: x.features)


#normalize:
#scaler = StandardScaler(withMean = True, withStd = True).fit(features)  #data needs to be dense (zeros included)
scaler = StandardScaler(withMean = False, withStd = True).fit(features)  #becomes dense if using withMean. may run out of memory locally

#convert data to dense vector to be normalized
#data2 = labels.zip(scaler.transform(features.map(lambda x: Vectors.dense(x.toArray()))))
data2 = labels.zip(scaler.transform(features))   #use this line if having memory issues


#hide 10% of the data for final test
data, test = data2.randomSplit([.9, .1])
#get size of chunks for 10-fold cross-validation
partitionSize = (len(data.collect())/10)

#train/validate 10 times on each k
i = 0
j = partitionSize
data = data.collect()
cv_error_storage = []

for w in range(10):
    #new train/validation split
    train = data[0:i] + data[j:]
    val = data[i:j]
    train = sc.parallelize(train)
    val = sc.parallelize(val)
    minError = float("inf")
    bestModel = None
    bestK = None
    test_values = [2,3,4,5,6,7,8,9,10]
    error_storage = []

    for x in test_values:
        model = KMeans.train(train.values(), x, maxIterations=10, runs=10, epsilon=.00001)
        error = model.computeCost(val.values())
        error_storage.append(error)
        print "******     model with " + str(x) + " clusters done in validation fold " + str(w+1) + " ***********"
        if error < minError:
            bestModel = model
            minError = error
            bestK = x
    cv_error_storage.append(error_storage)
    i = i + partitionSize
    j = j + partitionSize


CVerrors = []
for i in range(len(test_values)):
    val = np.mean(sum(val[i] for val in cv_error_storage))
    CVerrors.append(val)

minError = float('inf')
j = 0
for i in CVerrors:
    if i < minError:
        minError = i
        bestK = test_values[j]
        j = j+1
print 'best k: ' + str(bestK)
plt.plot(CVerrors)
plt.show()



#now score model on the test data
bestModel = KMeans.train(train.values(), bestK, maxIterations=10, runs=10, epsilon=.00001)
error = model.computeCost(test.values())
print "best model with k = " + str(k) " finished with error: " + str(error)

modelCenters = bestModel.clusterCenters

#get rdd of clusterid, movieid tuples  (predict after training)
trainingClusterLabels = train.map(lambda x: (bestModel.predict(x[1]), x[0]))

###################   example of this in action   ######################
#get recommendations for a user based on movies he/she liked
path2 = "/Users/jamesledoux/Documents/BigData/netflixrecommender/ratings.dat" #make this a passed-in argument
ratings = sc.textFile(path2).map(parseRating)  #make the parse rating function
ratingsByUser = ratings.map(lambda x: (x[0], (x[1],x[2])))
ratingsByUser = ratingsByUser.groupByKey().map(lambda x: (x[0], list(x[1]))).collect()

user = ratingsByUser[0]
userHighRatings = [movieRating for movieRating in user[1] if movieRating[1] == 5]

singleRating = userHighRatings[0]
clusterId = model.predict(data2.lookup(singleRating[0])[0])

#these are the recommended movies:
samplesInRelevantCluster = trainingClusterLabels.lookup(clusterId)


"""
errors for [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25]: best is 10
[7127750.091446353, 7127750.414798906, 7127751.405619981, 7127754.433954623, 7127750.722505061, 7127838.947968733, 7127783.328832407, 7127750.988140488, 7127747.7471736595, 7127750.230260722, 7127754.734692583]

errors for [8, 9, 10, 11, 12, 13, 14, 15]:  best is 8
[7127746.282929602, 7127821.055026695, 7127750.747225445, 7127751.5867197495, 7127751.9117591, 7127756.4383017905, 7127754.529469332, 7127756.724856978]

for [5, 6, 7, 8, 9, 10, 11, 12]: best is 12
[7127750.287024926, 7127752.929726642, 7127746.997023219, 7127745.9775424525, 7127748.620281398, 7127751.659824361, 7127753.905078466, 7118664.342015211]
"""

sc.stop()
