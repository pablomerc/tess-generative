import pandas as pd

file_path = '/data/vision/billf/scratch/pablomer/data/dataset_28x28_filtered_elev_above_zero.pkl'

try:
    df = pd.read_pickle(file_path)
    # You can now work with the DataFrame 'df'
    print(df.head())
except Exception as e:
    print(f"An error occurred while loading the pickle file with pandas: {e}")

print(df.keys())
