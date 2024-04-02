import argparse
import pyspark.pandas as pd
from pyspark.sql import SparkSession
import pandas as pd

import PIL.Image
import base64
import cv2 
import numpy as np



spark = (
    SparkSession.builder.appName("AML Dataprep")
    .getOrCreate()
)


parser = argparse.ArgumentParser()
parser.add_argument("--frames_ascii_dir")
parser.add_argument("--captured_data")
parser.add_argument("--captured_data_training")

args = parser.parse_args()
print("args.frames_ascii_dir: ", args.frames_ascii_dir)
print("args.captured_data: ", args.captured_data)
print("args.captured_data_training: ", args.captured_data_training)

all_frames_ascii_uri = f"{args.frames_ascii_dir}/all_frames_ascii.parquet"
keyboard_events_uri = f"{args.captured_data}/keyboard_events.csv"


all_frames_ascii_df = spark.read.parquet(all_frames_ascii_uri)
keyboard_events_df = spark.read.option("header", True).csv(keyboard_events_uri)

keyboard_events_df = keyboard_events_df.rdd.map(lambda x: (int(float(x["time"])), x["time"],x["event_type"],x["scan_code"],x["name"],x["is_keypad"])  ).toDF()
keyboard_events_df = keyboard_events_df.withColumnRenamed(keyboard_events_df.columns[0],"time_hash")
keyboard_events_df = keyboard_events_df.withColumnRenamed(keyboard_events_df.columns[1],"time")
keyboard_events_df = keyboard_events_df.withColumnRenamed(keyboard_events_df.columns[2],"event_type")
keyboard_events_df = keyboard_events_df.withColumnRenamed(keyboard_events_df.columns[3],"scan_code")
keyboard_events_df = keyboard_events_df.withColumnRenamed(keyboard_events_df.columns[4],"name")
keyboard_events_df = keyboard_events_df.withColumnRenamed(keyboard_events_df.columns[5],"is_keypad")

#x["ascii_art"]
all_frames_ascii_df = all_frames_ascii_df.rdd.map(lambda x: (int(float(x["frame_num"])), x["frame_num"], x["ascii_art"])   ).toDF()
all_frames_ascii_df = all_frames_ascii_df.withColumnRenamed(all_frames_ascii_df.columns[0],"time_hash")
all_frames_ascii_df = all_frames_ascii_df.withColumnRenamed(all_frames_ascii_df.columns[1],"frame_time")
all_frames_ascii_df = all_frames_ascii_df.withColumnRenamed(all_frames_ascii_df.columns[2],"ascii_art")

df_result = keyboard_events_df.join(all_frames_ascii_df, on=["time_hash"], how="inner")

print(f"Got df_result", df_result)
#df_result = df_result.toDF()

df_result.show()

df_result.write.mode('overwrite').option("delimiter", ",").option("header", True)\
    .csv(f"{args.captured_data_training}/merged_data.csv")

df_result.write.mode('overwrite').format('json').save(args.captured_data_training)

