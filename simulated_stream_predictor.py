# Databricks: "Real-Time" (Simulated) Streaming Play by Play Predictions

# SparkSQL imports
import pyspark.sql.functions as f
from pyspark.sql import Row
from pyspark.sql.types import *

# SparkML imports
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, Bucketizer, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

# Establish Spark context
sc = spark.sparkContext 

# COMMAND ----------

### CREATE TRUNCATED DATASET TO WORK WITH (for a more targeted problem domain)


# Load in the full-version of the raw NFL Play-By-Play dataset
nfl_df = spark.read \
    .format("csv") \
    .option("delimiter", ",") \
    .option("header", True) \
    .option("ignoreLeadingWhiteSpace", True) \
    .option("inferSchema", True) \
    .option("mode", "dropMalformed") \
    .load("dbfs:///FileStore/tables/4334_A2/nfl_pbp.csv")
# nfl_df.printSchema()

# Truncate dataset: filter for stats (columns) & seasons (rows) of interest
nfl_df = nfl_df.filter(f.col('game_id') > 2014000000).select(
    f.col('game_id').alias('game_id'),  # keep game_id (season info) for data partitioning later
    f.col('posteam').alias('pos_team'),  # team on offense for the play
    f.col('defteam').alias('def_team'),  # team on defense for the play
    f.col('yardline_100').alias('yardline_100'),  # yards till endzone
    f.col('ydstogo').alias('yds_till_first'),  # yards till 1st down
    f.col('half_seconds_remaining').alias('half_sec_remaining'),  # time remaining in half (sec)
    f.col('score_differential').alias('score_differential'),  # pos_team score - def_team score at the time of play
    f.col('posteam_timeouts_remaining').alias('pos_team_timeouts'),  # remaining timeouts for offense
    f.col('play_type').alias('label')  # label (run, pass, field-goal, punt, etc.)
).filter( (f.col('label')=='run') | (f.col('label')=='pass') | (f.col('label')=='punt') | (f.col('label')=='field-goal') )  # simplify problem to 4-level classfication
# nfl_df.count()  # 170167
# nfl_df.show(5)

# Create truncated file to use for project
nfl_df.write \
    .format("csv") \
    .mode("overwrite") \
    .option("header", True) \
    .save("FileStore/tables/4334_A2/nfl_truncated.csv/")

# COMMAND ----------

### LOAD IN TRUNCATED DATASET (truncated version used for this project)

# Define schema
nfl_schema = StructType([
    StructField('game_id', LongType(), True),
    StructField('pos_team', StringType(), True),
    StructField('def_team', StringType(), True),
    StructField('yardline_100', LongType(), True),
    StructField('yds_till_first', LongType(), True),
    StructField('half_sec_remaining', LongType(), True),
    StructField('score_differential', LongType(), True),
    StructField('pos_team_timeouts', LongType(), True),
    StructField('play_type', StringType(), True)
])

# Load in the truncated-version of the raw NFL Play-By-Play dataset
nfl_df = spark.read \
    .format("csv") \
    .option("delimiter", ",") \
    .option("header", True) \
    .option("ignoreLeadingWhiteSpace", True) \
    .schema(nfl_schema) \
    .option("mode", "dropMalformed") \
    .load("dbfs:///FileStore/tables/4334_A2/nfl_truncated.csv") \
    .persist()  # for multiple upcoming uses

print(nfl_df.count())  # 170167
nfl_df.printSchema()
nfl_df.show(5)

# COMMAND ----------

# Clean edge cases for "historical data" (standardize team names to newest versions)
# these cleaning measures won't apply for "streamed" new data
nfl_df = nfl_df.withColumn('pos_team', f.regexp_replace('pos_team', 'JAC', 'JAX'))
nfl_df = nfl_df.withColumn('pos_team', f.regexp_replace('pos_team', 'STL', 'LA'))
nfl_df = nfl_df.withColumn('pos_team', f.regexp_replace('pos_team', 'SD', 'LAC'))
nfl_df = nfl_df.withColumn('def_team', f.regexp_replace('def_team', 'JAC', 'JAX'))
nfl_df = nfl_df.withColumn('def_team', f.regexp_replace('def_team', 'STL', 'LA'))
nfl_df = nfl_df.withColumn('def_team', f.regexp_replace('def_team', 'SD', 'LAC'))


### DATA-PARTITIONING

# Construct train & test sets
train_df = nfl_df.filter(f.col('game_id') < 2018000000)
test_df = nfl_df.filter(f.col('game_id') > 2018000000)
# print(train_df.count(), test_df.count())  # outputs: 138445, 31647

# COMMAND ----------

### SET-UP SIMULATED STREAMING (TEST) DATASET

# Find how many games encompass 2018 season (test-data) to partition accordingly
# test_df.groupBy(f.col('game_id')).agg(f.count("game_id").alias('count')).count()  # 224 games total in test data

# Repartition play-data
test_df = test_df.repartition(20)  # ideally, repartition a record for each game to each dir, to simulate ideal game-like live score conditions
test_df = test_df.sort('game_id')  # for TESTING purposes: keep order by game ID

# Write out each partition to a file
test_df.write \
    .format("csv") \
    .mode("overwrite") \
    .option("header", True) \
    .save("FileStore/tables/4334_A2/partitioned_test_data/")  # write to to partitioned-number of files

# COMMAND ----------

### Establish pipeline encompassing from pre-processing to modeling stages

# Index string columns (team info & label)
pos_team_indexer = StringIndexer(inputCol='pos_team', outputCol='pos_team_index')  # estimator object
def_team_indexer = StringIndexer(inputCol='def_team', outputCol='def_team_index')  # estimator object
label_indexer = StringIndexer(inputCol='play_type', outputCol='label')  # estimator object

# Bin the score_differential & half_sec_remaining categories 
score_splits = [-float("inf"), -14, -7, 7, 14, float("inf")]  # split into single-possession, 2-possession, 2+ possession bins
score_bucketizer = Bucketizer(splits=score_splits, inputCol="score_differential", outputCol="score_diff_bucket")  # transformer object

# Construct a vectorizer for applying machine learning algorithm
vectorizer = VectorAssembler(inputCols=['pos_team_index', 'def_team_index', 'yardline_100', 'yds_till_first', 'half_sec_remaining', 'score_diff_bucket', 'pos_team_timeouts'], outputCol='features')

# Scale the feature set (vectorized input)
scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features') # estimator object

# Instantiate estimator of choice
rf = RandomForestClassifier(
    maxDepth = 7,  # manually tuned for optimal balance between accuracy & runtime
    numTrees = 30, # manually tuned for optimal balance between accuracy & runtime
    seed = 42
)

# Assemble pipeline
pipe = Pipeline(stages=[
    pos_team_indexer,   # indexes team info string-formatted column
    def_team_indexer,   # indexes team info string-formatted column
    label_indexer,      # indexes play type string-formatted column
    score_bucketizer,   # bins scoring info
    vectorizer,         # outputs vectorized feature set
    scaler,             # normalizes feature set (since features were in different scales)
    rf                  # applies ML classification algorithm
])
pipe_model = pipe.fit(train_df)

# COMMAND ----------

### STATIC PREDICTIONS

## TRAIN SET PREDICTIONS
train_pred = pipe_model.transform(train_df).persist()
correct_count = train_pred.filter(f.col('label')==f.col('prediction')).count()
total_count = train_pred.count()
acc = correct_count / total_count
print('Accuracy for train set predictions: ', acc)

## TEST SET PREDICTIONS
test_pred = pipe_model.transform(test_df).persist()
correct_count = test_pred.filter(f.col('label')==f.col('prediction')).count()
total_count = test_pred.count()
acc = correct_count / total_count
print('Accuracy for test set predictions: ', acc)

# COMMAND ----------

# Static Query (with columns of interest for final table queries)
static_query_df = train_pred.select(
    f.col('game_id'), 
    f.col('pos_team'), 
    f.col('def_team'), 
    f.col('yardline_100'), 
    f.col('yds_till_first'), 
    f.col('half_sec_remaining'), 
    f.col('score_differential'), 
    f.col('play_type'), 
    f.col('label'), 
    f.col('probability'), 
    f.col('prediction')
)
static_query_df.createOrReplaceTempView('dynamic_tbl')

# COMMAND ----------

### STRUCTURED STREAMING

## SOURCE
source_stream = spark.readStream \
    .format("csv") \
    .option("header", True) \
    .schema(nfl_schema) \
    .option("maxFilesPerTrigger", 1) \
    .load("dbfs:///FileStore/tables/4334_A2/partitioned_test_data")

## QUERY
test_pred = pipe_model.transform(source_stream)
test_query = test_pred.select(
    f.col('game_id'), 
    f.col('pos_team'),
    f.col('def_team'), 
    f.col('yardline_100'),
    f.col('yds_till_first'), 
    f.col('half_sec_remaining'),
    f.col('score_differential'), 
    f.col('play_type'),   ### REMOVE TO FULLY SIMULATE REAL-LIVE CASE OF PREDICTING WITHOUT KNOWING FINAL OUTCOME
    f.col('label'),       ### REMOVE TO FULLY SIMULATE REAL-LIVE CASE OF PREDICTING WITHOUT KNOWING FINAL OUTCOME
    f.col('probability'), 
    f.col('prediction')
)

## SINK
sink_stream = test_query.writeStream \
    .outputMode("append") \
    .format("memory") \
    .queryName("dynamic_tbl") \
    .trigger(processingTime='10 seconds') \
    .start()

# COMMAND ----------

# SQL query must look into the most recent additions as sink outputmode has to be 'append'
spark.sql("select * from dynamic_tbl order by game_id desc limit 20").show()

# COMMAND ----------


