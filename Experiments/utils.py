import pandas as pd
import numpy as np

def add_split_column(df):
    # Generate a random assignment for each row
    np.random.seed(42)
    random_values = np.random.choice(
        ['train', 'test', 'calibration'],
        size=len(df),
        p=[0.6, 0.2, 0.2]  # Probabilities for train, test, and calibration
    )
    # Add the new 'split' column to the DataFrame
    df['split'] = random_values
    return df