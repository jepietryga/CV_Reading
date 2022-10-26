# Class and functions for handling ECV experiments on Megalibraries

import read_sdc
import glob
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ECV_Experiment():

    def __init__(self,directory_list=[],scan_per_experiment=3):
        '''
        Arg:
            directory_list (List[String]): List of paths to folders holding experiment data given in order of experiment
            scan_per_experiment (int)    : Number of scans per point in the megalibrary
            scan_values    (List[int])   : Values associated with the scan rates (MUST HAVE EQUAL SIZE AS SCAN_PER_EXPERIMENT)
                                            (mV/s)
        '''
        #assert(scan_per_experiment == len(scan_rate_values))

        self.directory_list = directory_list
        self.scan_per_experiment = scan_per_experiment
        self.experiment_arr = []
        self.Experiments = []
        self._df = None

        # Initiate Paths
        self._get_experiment_arr()
        self._create_experiments()

    def _get_experiment_arr(self):
        experiment_arr = []
        for directory in self.directory_list:
            # File Read-in
            file_list = glob.glob(f'{directory}/* - *.ucv')
            file_list = sorted(file_list, key=lambda x:ucv_number_extractor(x))

            # Experiment Chunking (Group file_paths based on which NP being scanned)
            # KEY ASSUMPTION: All scans are continguous
            for ii in np.arange(0,len(file_list),self.scan_per_experiment):
                try:
                    experiment_ID = file_list[ii]
                    
                    experiment = tuple(
                                    [file_list[ii+j] for j in np.arange(self.scan_per_experiment)]
                                )
                    experiment_arr.append(experiment)
                except:
                    print(f'WARNING: Unmatched experiment for {ii}\nLikely scan reset')
        self.experiment_arr = experiment_arr

    def _create_experiments(self):
        '''
        Using all grouped scans, create Experiments
        '''
        self.Experiments = []
        for experiment in self.experiment_arr:
            print(experiment)
            self.Experiments.append(Experiment(experiment))
    
    def _create_columns(self):
        columns = ['Experiments']
        for scan in self.scan_rate_values:
            name = f'{scan} mV/s'
            columns.append(f'{name} Current (A)')
            columns.append(f'{name} Data ((V,A))')
        columns.append('Slope (F)')
        columns.append('Intercept (A)')
        columns.append('R2')
        self.columns = columns
        return columns

    @property
    def df(self):
        return self._df

    @df.getter
    def df(self):
        if self._df == None:
            self._df = self.get_df()
        return self._df

    def get_df(self):
        # self._create_columns()
        # df = pd.DataFrame(columns=self.columns)
        data = []

        for experiment in self.Experiments:
    
            data.append(pd.DataFrame.from_dict(experiment.fitting_row))

            #new_row = pd.DataFrame(new_data)
        df = pd.concat(data)
        df.reset_index(inplace=True)
        return df

    def __getitem__(self,i):
        return self.Experiments[i]

class Experiment():

    def __init__(self,paths=None):
        '''
        An experiment covers an individual nanoparticle on grid with all scans associated

        Arg:
            paths List[Strings] = Paths related to scans of the nanoparticle
        '''
        self.paths = paths
        self.data = {}
        self.fitting_row = {}

        self._create_data_dict()
        self._create_fitted_row()

    def _create_data_dict(self):
        '''
        For each scan, get experimentally relevant values
        NOTE: To add more, need to look at the dict provided as 'total_data' and examine it
        '''
        x = None
        y = None
        for run in self.paths:
            try:
                self.data[run] = {}
                total_data = read_sdc.read_ulv(run)
                self.data[run]["Time"] = total_data["Data0"] # Seconds
                self.data[run]["Potential"] = total_data["Data1"] # Volts
                self.data[run]["Current"] = total_data["Data2"] # Amps
                self.data[run]["Scan Rate"] = total_data["CVESweepRateVSec"] # V
                if x == None:
                    x = total_data["X_Position"]
                    y = total_data["Y_Position"]
                self.data[run]["X"] = x
                self.data[run]["Y"] = x
            except:
                print(f'WARNING: {run} failed to read data. Maybe Bad data file?')
    
    def _create_fitted_row(self):
        '''
            Return row of a singular experiment, fitting all relevant data
        '''
        avg_current_val_arr = []
        combined_data_arr = []
        scan_rate_values = []
        # Process Data
        for run in self.paths:
            try:
                run_dict = self.data[run]
                avg_val, nn, mid_volts = get_weighed_max_current(run_dict["Potential"],run_dict["Current"])

                # For calculation
                avg_current_val_arr.append(avg_val)
                scan_rate_values.append(run_dict["Scan Rate"])

                # For later storage and external plot usage
                combined_data = list(zip(run_dict["Potential"],run_dict["Current"]))
                combined_data_arr.append(combined_data)
            
            except:
                print(f': Error reading {run}, not included in fitting')
                #avg_current_val_arr.append(-1)
                #combined_data_arr.append((-1,-1))
        
        # Fit Data
        m,x = np.polyfit(scan_rate_values,avg_current_val_arr,1)
        fitted_current_arr = [m*scan+x for scan in scan_rate_values]
        r2 = R2(avg_current_val_arr,fitted_current_arr)

        # Store Data
        new_row = {}
        new_row["Experiments"] = [self.paths]

        ii = 0
        for scan in scan_rate_values:
            name = f'{scan} V/s'
            max_current_string = f'{name} Current (A)'
            data_string = f'{name} Data ((V,A))'
            new_row[max_current_string] = [avg_current_val_arr[ii]]
            new_row[data_string] = [combined_data_arr[ii]]
            ii += 1
        new_row["Slope (F)"] = m
        new_row["Intercept (A)"] = x
        new_row["R2"]=r2

        self.fitting_row = new_row

    def plot_experiment(self):
        '''
           Plot an experiment tuple!
        '''
        row = self.fitting_row
        x = []
        y = []
        r2 = row["R2"]
        m = row["Slope (F)"]
        b = row["Intercept (A)"]
        for key, val in row.items():
            if "Current" in key:
                x.append(float(key.split(' ')[0]))
                y.append(val)
        x = np.array(x)
        y = np.array(y)
        fitted_data = x*m+b
        plt.scatter(x,y,color='red')
        plt.plot(x,fitted_data,color='black')
        

        plt.text(min(x),max(fitted_data)*.9,f'R$^2$: {r2:.4}\nSlope: {m:.4} F')
        plt.xlabel("Scan Rate (V/s)")
        plt.ylabel("Current (A)")


def ucv_number_extractor(file_path):
    '''
    From standard file name convention, get organizational numbers
    '''
    split_file = file_path \
                    .split("-")[-1] \
                    .split(".")[0]
    return int(split_file)

def get_cv_values(file_path):
    print(f'Reading {file_path}...')
    total_data = read_sdc.read_ulv(file_path)
    #print(total_data["DataTitle0"])
    #print(total_data["DataTitle1"])
    #print(total_data["DataTitle2"])
    time_arr = total_data["Data0"]
    potential_arr = total_data["Data1"]
    current_arr = total_data["Data2"]
    return time_arr,potential_arr,current_arr

# IF the graph is presumed symmetric and like a parallelogram, then simply getting the middle of the entire x-range is
# Sufficient for finding the middle value
def simple_central_value(x_arr):
    return ((max(x_arr)+min(x_arr))/2)

def get_weighed_max_current(volts_arr,amps_arr,nearest_neighbors=4):
    '''
        Nearest neighbors (int) : allows us to define a larger average range
        Returns: 
        avg_value (float) : Average current of all nn 
        nn (list[tuple]) : All points used in calculation used (for plot checking)
        middle_volts (float) : Central value (used for plot checking)
    '''
    middle_volts = simple_central_value(volts_arr)
    va_zipped = zip(volts_arr,amps_arr)
    va_zipped = sorted(va_zipped,key=lambda x: np.abs(x[0]-middle_volts)) # Points closest top AND bottom
    nn_amps = [np.abs(va_zipped[ii][1]) for ii in np.arange(nearest_neighbors)]
    return np.sum(nn_amps)/nearest_neighbors, va_zipped[:nearest_neighbors], middle_volts

def R2(actual_data,fitted_data):
    actual_data = np.array(actual_data)
    fitted_data = np.array(fitted_data)
    numerator = sum((actual_data-fitted_data)**2)
    denominator = sum((actual_data-np.mean(actual_data))**2)
    #print(actual_data-fitted_data)
    #print(np.mean(actual_data))
    #print(numerator,denominator)
    return 1-numerator/denominator