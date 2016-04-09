from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from matplotlib import pyplot as plt
import numpy as np
from os.path import join
import sys
#pyspark.tuning.ml.crossvalidator import?

if(len(sys.argv) != 2):
    print "usage: /sparkPath/bin/spark-submit  name.py  movieDirectory"

conf = SparkConf().setAppName("KMeans Collaborative").set("spark.executor.memory", "7g")
movieLensHomeDir = sys.argv[1]   # passed as argument
#movieLensHomeDir = "/Users/jamesledoux/Documents/BigData/netflixrecommender/"
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
print "count of ratings: ", len(ratings.collect())
print "sample rating: ", ratings.take(1)

#num of users  (userid, movieid, ratings)
numUsers = ratings.values().map(lambda x: x[0]).max()+1
numMovies = ratings.values().map(lambda x: x[1]).max()+1


ratingsSV = vectorize( ratings.values(), numMovies)
print "RatingsSV Type:", type(ratingsSV)
print "RatingsSV Count:", ratingsSV.count()

"""
cross validation:
1: set aside 10 percent of data for a final test
2: get size of remaining data
3: train / test model iteratively, where each iteration hides the next .10 of the
    data as a validation set, keeping the rest as training data
- keep MSE each time, and then take mean(sum(MSEs)) at each K, which will be the
    validation MSE. Min validation MSE == best k.
"""
data, test = ratingsSV.randomSplit([.9, .1])
partitionSize = (len(data.collect())/10)

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
    test_values = [80, 90, 100, 110]
    error_storage = []
    for x in test_values:
        model = KMeans.train(train.values(), x, maxIterations=10, runs=10, epsilon=.00001)
        error = model.computeCost(val.values())
        error_storage.append(error)
        print "******     model with " + str(x) + " clusters done in validation fold " + str(w+1) + " ***********"
        print "with error: " + str(error)
        if error < minError:
            bestModel = model
            minError = error
            bestK = x
    cv_error_storage.append(error_storage)
    i = i + partitionSize
    j = j + partitionSize


#get CVerrors (mean of the errors from the 10 cross validated samples)
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
lista = CVerrors
listb = [9067860.9215988759, 8889939.3706063181, 8825103.9420495424, 8762589.0398594737, 8724469.4796739593, 8713997.2527633812, 8679368.9173081759]
listc = listb + lista
plt.plot(listc)
plt.show()
#for the purposes of this exercise, the goal was to get the best k
#but if we were to predict from here, we would run the following commented out code, which
#is a final model with the optimal k, and then predict movies based off that
"""
model = KMeans.train(train.values(), bestK, maxIterations=10, runs=10, epsilon=.00001)
error = model.computeCost(test.values())
print "best model with k = " + str(k) " finished with error: " + str(error)
"""

"""
80, 90, 100, 110
[7803423.6893760711, 7791393.5365485651, 7786281.8075585756, 7774209.5088197934]
[10, 20, 30, 40, 50, 60, 70]
[9067860.9215988759, 8889939.3706063181, 8825103.9420495424, 8762589.0398594737, 8724469.4796739593, 8713997.2527633812, 8679368.9173081759]

[5, 9, 12]
[1018858.080614449, 988271.0902984664, 989009.7983816753]
[12, 24, 36]
[981547.3412806683, 967495.7386560757, 956038.1052810646]
[40, 80, 120]
[955456.0452899686, 941952.0370086507, 937651.9618938119]
[160, 200]
[958382.594830455, 957538.972414101]
[120, 140, 150]
[955591.5111267052, 960119.3043533752, 964760.2851641065]
"""



#model.save(sc, "KMeansModelCollaborative")
#model = KMeansModel.load(sc, "KMeansModelCollaborative")






#for i in range(10, 20):
# ratingsSV = > RDD where each item is ( userID , SparseVector)

##    if error < minError:
##        bestModel = model
##        minError = error
##        bestK = i
user = ratingsSV.values().take(5)[4] #take a sample of 1 from the data set (use test data when doing this)
print "Type user:", type(user)
print "User:", user

label = bestModel.predict(user)   #outputs which cluster this user belongs to
clusterCenters = model.clusterCenters     #len == total num of movies, each obs ==  rating for people in this group
#clusterCenters[0] #len == total num of movies, each obs == avg rating for people in this group
print "Len Cluster Centers:", len(clusterCenters)
movieID = 4

print "predicted value: ", clusterCenters[label][movieID]


sc.stop()
