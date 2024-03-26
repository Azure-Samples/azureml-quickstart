import argparse
from operator import add
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession
import os 
import mltable
from importlib.metadata import version
print('MLTABLE VERSION', version('mltable'))


spark = (
    SparkSession.builder.appName("AML Dataprep")
    .getOrCreate()
)



parser = argparse.ArgumentParser()
parser.add_argument("--invoice_data")
parser.add_argument("--training_data")

args = parser.parse_args()
print(args.invoice_data)
print(args.training_data)

df = spark.read.parquet(args.invoice_data).limit(10000).cache()

training_data_path = os.path.join(args.training_data, "data")

print (f"Writing to {training_data_path}")

df.repartition("debtor_id").write.partitionBy("debtor_id")\
        .mode('overwrite')\
        .parquet(training_data_path)

#Create an mltable from the parquet files
paths = [
    {
        "pattern": os.path.join(training_data_path,"debtor_id=*/*")
    }
]

tbl = mltable.from_parquet_files(
    paths=paths,
    
)

tbl2 = tbl.extract_columns_from_partition_format("/debtor_id={debtor_id}")
tbl2.save(path=args.training_data, colocated=False, show_progress=True, overwrite=True)