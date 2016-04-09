from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors

def parseRating(line):
    parts = line.strip().split("::")
    # userID, movieID, rating
    return (int(parts[0]),int(parts[1]),float(parts[2]))

# Step 1 - create spark context
conf = SparkConf().setAppName("Kmeans-Content").set("spark.executor.memory","4g").set("spark.storage.memoryFraction","0")
sc = SparkContext()

# Step 2 - Load in input file
data = MLUtils.loadLibSVMFile(sc, "/Users/David/spark-1.6.0-bin-hadoop2.6/tutorial/data/movielens/medium/movie_features_dataset.dat")

#movieIDs == labels
labels = data.map(lambda x: x.label)
features = data.map(lambda x: x.features)

#for every feature in features:
#   f' = (f-mean)/std

# Step 3 - standardize the data with unit values and 0 mean
#scaler = StandardScaler(withMean = True, withStd = True).fit(features)
scaler = StandardScaler(withMean=False, withStd=True).fit(features)
data2 = labels.zip(scaler.transform(features))
#data2 = labels.zip(scaler.transform(features.map(lambda x: Vectors.dense(x.toArray()))))

#(movieID vector)
numFeatures = len(data2.values().take(1)[0])
#print("Type of data2:", type(data2)) #RDD
#print("Type of data2.values():",  type(data2.values())) #pipelinedrdd
#print("Sample:", data2.values().take(1)[0])

#splitting up the data
training,validation,test = data2.randomSplit([.85,.10,.05])
#data2.map(lambda x: (random.randInt(1,10),x))
#train = data2.filter(lambda x: if x[0] in range(1,9))
#print("Training Dataset Size:", training.count())
#print("Validation Dataset Size:", validation.count())
#print("Test Dataset Size:", test.count())

k=20
maxIterations = 10
runs = 5
epsilon=0.00001
model = KMeans.train(training.values(), k, maxIterations, runs)

clusterCenters = model.clusterCenters
trainingClusterLabels = training.map(lambda x: (model.predict(x[1]),x[0]))
# RDD where each item is (clusterID, movieID)

# given a movie find the appropriate cluster

# recommendaiton for user'
# moviesLiked <- user' liked
# for m' in moviesLiked:
#   clusterLabel = use KMeans to predict the cluster for m'
# most frequent clusterLabel

#open ratings file and select a user rating
ratings = sc.textFile("/Users/David/spark-1.6.0-bin-hadoop2.6/tutorial/data/movielens/medium/ratings.dat").map(parseRating)

ratingsByUser = ratings.map(lambda x: (x[0],(x[1],x[2])))
ratingsByUser = ratingsByUser.groupByKey().map(lambda x: (x[0],list(x[1]))).collect()
#ratingsByUser -> RDD of (key=>userId, values=>Sparse/DenseVector)

# lets do a test for 1 user
user = ratingsByUser[0] # look at 1 user
# get the ratings of movies the user liked, i.e. movies with rating = 5
userHighRatings = [movieRating for movieRating in user[1] if movieRating[1] == 5]
singleRating = userHighRatings[0]
clusterId = model.predict(data2.lookup(singleRating[0])[0])
samplesInRelevantCluster = trainingClusterLabels.lookup(clusterId)
print(samplesInRelevantCluster)

sc.stop()
