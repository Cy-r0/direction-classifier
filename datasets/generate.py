import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm


class DCDatasetGenerator(object):
    """
    Read .csv files of raw time series data, generate new features if needed
    (e.g. moving averages), and then extract data samples.
    Raw data needs to contain at least the following features:
        Year, Month, MonthDay, WeekDay, Hour, Open, High, Low, Close.
    Args:
        symbol (string): determines both the raw data file to read, and how to
            name the dataset.
        past_timesteps (int): number of past timesteps to keep in each sample.
        target_perc (float): percentage of starting price that needs to be
            reached in order for the target to be valid.
    NOTE: only works when executed from "datasets" directory.
    """

    def __init__(self, symbol, past_timesteps, target_perc):
        datafile = os.path.join("raw_data", symbol + ".csv")
        assert os.path.exists(datafile), "raw data file not found"

        self.root = os.path.join("prepared_data", symbol)
        # Delete dataset root if it already exists
        if os.path.exists(self.root):
            shutil.rmtree(self.root)

        # Make directories for samples and targets
        self.samples_dir = os.path.join(self.root, "samples")      
        os.makedirs(self.samples_dir)
        self.targets_dir = os.path.join(self.root, "targets")
        os.makedirs(self.targets_dir)
        
        self.data = pd.read_csv(datafile)
        self.data_len = len(self.data)
        self.prepared_data_len = len(self.data) - past_timesteps

        self.past_timesteps = past_timesteps
        assert past_timesteps > 0, "number of past timesteps is not > 0"
        self.target_delta = target_perc / 100

        # Initialise all stats to keep track of
        self.targetcount_high = 0
        self.targetcount_low = 0
        self.targetcount_both = 0
        self.targetcount_none = 0
        self.mean_targetdelay_high = 0
        self.max_targetdelay_high = 0
        self.min_targetdelay_high = 999999
        self.mean_targetdelay_low = 0
        self.max_targetdelay_low = 0
        self.min_targetdelay_low = 999999

        self.extract_samples()

        self.log_stats()


    def extract_samples(self):
        """
        Create data samples, find targets, and write both to csv.
        """
        for s in tqdm(range(self.past_timesteps, self.data_len)):
            sample = self.data[s - self.past_timesteps:s]
            target = self._find_target(s)
            
            # Accumulate stats
            target_value = target.at[0, "Target"]
            target_delay = target.at[0, "TargetDelay"]

            if target_value == 1:
                self.targetcount_high += 1
                self.mean_targetdelay_high += target_delay
                # Update min and max target delay
                if target_delay > self.max_targetdelay_high:
                    self.max_targetdelay_high = target_delay
                if target_delay < self.min_targetdelay_high:
                    self.min_targetdelay_high = target_delay

            elif target_value == 0:
                self.targetcount_low += 1
                self.mean_targetdelay_low += target_delay
                # Update min and max target delay
                if target_delay > self.max_targetdelay_low:
                    self.max_targetdelay_low = target_delay
                if target_delay < self.min_targetdelay_low:
                    self.min_targetdelay_low = target_delay
            
            elif target_value == 2:
                self.targetcount_both += 1
            
            elif target_value == None:
                self.targetcount_none += 1

            # Dont write to csv if target is invalid
            if target_value != 0 and target_value != 1:
                continue
            else:
                # Put together file name
                year = self.data.at[s, "Year"]
                month = self.data.at[s, "Month"]
                day = self.data.at[s, "MonthDay"]
                hour = self.data.at[s, "Hour"]
                sample_name = str(year) \
                    + self._2digits(month) \
                    + self._2digits(day) \
                    + self._2digits(hour)

                sample.to_csv(os.path.join(self.samples_dir, sample_name + ".csv"), 
                    index=False)
                target.to_csv(os.path.join(self.targets_dir, sample_name + ".csv"), 
                    index=False)
        
        # Normalise mean target delays
        self.mean_targetdelay_high /= self.targetcount_high
        self.mean_targetdelay_low /= self.targetcount_low
    
    def _find_target(self, start_idx):
        """
        Iterate through dataset row by row, looking at each candle to see if it
        reaches the target up or down.
        Args:
            start_idx (int): row to start looking from.
        """
        start_price = self.data.at[start_idx - 1, "Close"]
        up_target = start_price + start_price * self.target_delta
        down_target = start_price - start_price * self.target_delta

        reached_up = False
        reached_down = False
        idx = start_idx
        target_delay = -1    # Dirty fix to start from 0 in the while loop 

        # Iterate thru dataset
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
            target = 2
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
        features_to_keep=["Weekday", "Hour", "High", "Low", "Close"]):
        """
        Choose which features to keep from the original data.
        Args:
            features_to_keep (list): list of features to keep.
        """
        if features_to_keep == "All":
            pass
        else:
            self.data = self.data[features_to_keep]

    def make_new_features(self):
        """
        (not implemented).
        """
        pass

    def log_stats(self):
        """
        Log dataset statistics.
        NOTE: requires dataset samples to be already extracted.
        """
        log_path = os.path.join(self.root, "stats.txt")
        with open(log_path, "w") as f:
            print("Past timesteps: %d" %(self.past_timesteps), file=f)
            print("Target delta (fraction of starting price): %f" 
                %(self.target_delta), file=f)
            print("%d/%d (%f %%) datapoints discarded because both top and " \
                "bottom targets are reached in the same hour."
                %(
                    self.targetcount_both,
                    self.prepared_data_len,
                    self.targetcount_both / self.prepared_data_len * 100),
                file=f)
            print("%d/%d (%f %%) datapoints discarded because neither target " \
                "is reached by the end of the dataset."
                %(
                    self.targetcount_none, 
                    self.prepared_data_len, 
                    self.targetcount_none / self.prepared_data_len * 100),
                file=f)
            total_valid_points = self.targetcount_high + self.targetcount_low
            print("%d/%d (%f %%) datapoints with a high target, reached after " \
                "%.2f hours on average (%d max, %d min)."
                %(
                    self.targetcount_high,
                    total_valid_points,
                    self.targetcount_high / total_valid_points * 100,
                    self.mean_targetdelay_high,
                    self.max_targetdelay_high,
                    self.min_targetdelay_high),
                file=f)
            print("%d/%d (%f %%) datapoints with a low target, reached after " \
                "%.2f hours on average (%d max, %d min)."
                %(
                    self.targetcount_low, 
                    total_valid_points, 
                    self.targetcount_low / total_valid_points * 100,
                    self.mean_targetdelay_low,
                    self.max_targetdelay_low,
                    self.min_targetdelay_low),
                file=f)



if __name__ == "__main__":
    _ = DCDatasetGenerator(symbol="EURUSD", past_timesteps=240, target_perc=0.1)
    
    