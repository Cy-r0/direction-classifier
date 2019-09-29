import os
import shutil

import numpy as np
import pandas as pd


class DatasetGenerator(object):
    """
    Read .csv files of raw time series data, and:
        1. choose features and generate new ones (e.g. moving averages);
        2. extract data samples from time series.
    Raw data needs to contain at least the following features:
        Year, Month, Day, Hour, Open, High, Low, Close.
    Args:
        symbol (string): determines both the raw data file to read, and how to
            name the dataset.
        past_timesteps (int): number of past timesteps to keep in each sample.
        target_perc (float): percentage of starting price that needs to be
            reached in order for the target to be valid.
    """

    def __init__(self, symbol, past_timesteps, target_perc):
        datafile = os.path.join("raw_data", symbol + ".csv")
        assert os.path.exists(datafile), "raw data file not found"

        root = os.path.join("prepared_data", symbol)
        # Delete dataset root if it already exists
        if os.path.exists(root):
            shutil.rmtree(root)

        # Make directories for samples and targets
        self.samples_dir = os.path.join(root, "samples")      
        os.makedirs(self.samples_dir)
        self.targets_dir = os.path.join(root, "targets")
        os.makedirs(self.targets_dir)
        
        self.data = pd.read_csv(datafile)
        self.data_len = len(self.data)

        self.past_timesteps = past_timesteps
        assert past_timesteps > 0, "number of past timesteps is not > 0"
        self.target_perc = target_perc

        self.extract_samples()

    def extract_samples(self):
        """
        Create data samples, find targets, and write both to csv.
        """
        for s in range(self.past_timesteps, self.data_len):
            sample = self.data[s - self.past_timesteps:s]
            target = self._find_target(s)

            # Put together file name
            year = self.data.at[s, "Year"]
            month = self.data.at[s, "Month"]
            day = self.data.at[s, "Day"]
            hour = self.data.at[s, "Hour"]
            sample_name = str(year) \
                + self._2digits(month) \
                + self._2digits(day) \
                + self._2digits(hour)

            sample.to_csv(os.path.join(self.samples_dir, sample_name + ".csv"), 
                index=False)
            target.to_csv(os.path.join(self.targets_dir, sample_name + ".csv"), 
                index=False)
    
    def _find_target(self, start_idx):
        """
        Iterate through dataset row by row, looking at each candle to see if it
        reaches the target up or down.
        Args:
            start_idx (int): row to start looking from.
        """
        start_price = self.data.at[start_idx - 1, "Close"]
        up_target = start_price + start_price * self.target_perc
        down_target = start_price - start_price * self.target_perc

        reached_up = False
        reached_down = False
        idx = start_idx
        target_delay = -1    # Dirty fix to start from 0 in the while loop 

        while not reached_up and not reached_down and idx < self.data_len:
            timestep = self.data.iloc[idx]
            if timestep["High"] >= up_target:
                reached_up = True
            if timestep["Low"] <= down_target:
                reached_down = True

            idx += 1
            target_delay += 1
        
        if reached_up and not reached_down:
            target = 1
        elif reached_down and not reached_up:
            target = 0
        elif reached_up and reached_down:
            target = 9999
        elif not reached_up and not reached_down:
            target = None

        # Variables need to be made into lists otherwise pandas wants an index
        return pd.DataFrame({"Target": [target], "TargetDelay": [target_delay]})

    def _2digits(self, number):
        """
        Convert integer to two-digit format, prepending a zero if its smaller 
        than 10, and then convert it to string.
        """
        if number < 10:
            return "0" + str(number)
        else:
            return str(number)

    def select_features(self,
        features_to_keep=["Hour", "High", "Low", "Close"]):
        """
        Choose which features to keep from the original data.
        Args:
            features_to_keep (list): list of features to keep.
        """
        self.data = self.data[features_to_keep]

    def make_new_features(self):
        """
        (not implemented).
        """
        pass

            
        






if __name__ == "__main__":
    _ = DatasetGenerator(symbol="USDFAK", past_timesteps=2, target_perc=0.1)
    
    