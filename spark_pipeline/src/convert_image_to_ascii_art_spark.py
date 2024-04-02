import argparse
from operator import add
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession
import os 
from importlib.metadata import version
import sys
import pandas as pd

from os import listdir
from os.path import isfile, join

import PIL.Image
import base64
import cv2 
import numpy as np

from convert_image_to_ascii_art import convert_image_to_ascii


spark = (
    SparkSession.builder.appName("AML Dataprep")
    .getOrCreate()
)


parser = argparse.ArgumentParser()
parser.add_argument("--frames_dir")
parser.add_argument("--frames_ascii_dir")

args = parser.parse_args()
print("args.frames_dir: ", args.frames_dir)

file_df = spark.read.option("header", True).csv(f"{args.frames_dir}/all_frames64.csv")
print("file_df:", file_df)

file_df.show()

def process_image(frame_num, im_b64):

    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # Convert to Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    img = PIL.Image.fromarray(img)
    ascii_art = convert_image_to_ascii(img)
    return (frame_num, ascii_art)

df_result = file_df.rdd.map(lambda x : process_image(x[1], x[2]))

print(f"Got df_result", df_result)
df_result = df_result.toDF()

df_result = df_result.withColumnRenamed(df_result.columns[0],"frame_num")
df_result = df_result.withColumnRenamed(df_result.columns[1],"ascii_art")

df_result.show()

df_result.write.mode('overwrite')\
    .parquet(f"{args.frames_ascii_dir}/all_frames_ascii.parquet")