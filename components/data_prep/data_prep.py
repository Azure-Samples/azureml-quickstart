import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data_csv", type=str, help="name of train data")
    parser.add_argument("--test_data_csv", type=str, help="name of test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    credit_df = pd.read_csv(args.data, header=1, index_col=0)

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    credit_train_df, credit_test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )

    os.makedirs(args.train_data_csv, exist_ok=True)
    os.makedirs(args.test_data_csv, exist_ok=True)

    print("???!!!", args.train_data_csv)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    # train_data_path = os.path.join(args.train_data, "data.csv")
    # print("train_data_path", os.path.abspath(train_data_path))


    credit_train_df.to_csv(os.path.join(os.getcwd(), args.train_data_csv, "data.csv"), index=False)

    # test_data_path = os.path.join(args.test_data, "data.csv")
    # print("test_data_path", os.path.abspath(test_data_path))
    credit_test_df.to_csv(os.path.join(os.getcwd(), args.test_data_csv, "data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
