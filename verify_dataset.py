#!/usr/bin/env python
"""
    A simple script to analyze a given dataset file and perform some statistics.
    The dataset is a CSV file with at most four headers formated as follows:
    PATH,   LABEL,  SUBSET_LABEL,   DESCRIPTION
  
    It can support less than four headers and can be extended easily to support further (or less) columns.
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2019, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"


# NOTE: THIS IS MY FIRST EVER CODE I WROTE IN PYTHON, SO IT MIGHT NOT BE VERY LEAN-CODED PROGRAM.


import csv
import numpy as np
import os.path
from enum import Enum
import random
import math


class OperationType(Enum):
        REPORT_ONLY = 1
        REDUCE = 2 
        DUPLICATE = 3
        # other method can be supported in the future

class DataSample:
    """
        Represents the data sample. Objects can be constructed by either passing 
        a list "list_of_attributes" or each by passing each atrribute seperately.
    """
    def __str__(self):
        obj_string = ''
        try:
            if self.path is not None:
                obj_string += str(self.path)
            if self.label is not None:
                obj_string += ',\t' + str(self.label)
            if self.subset_label is not None:
                obj_string += ',\t' + str(self.subset_label) 
            if self.description is not None:
                obj_string +=  ',\t' + str(self.description)
        except AttributeError:
            pass
        return obj_string

    def __init__(self, list_of_attributes = None, path = None, label = None, subset_label = None, description = None):
        if list_of_attributes is not None:
            if isinstance(list_of_attributes, list):
                length = len(list_of_attributes)
                if length == 0:
                    print('WARNINIG: empty sample!')
                elif length < 2:    
                    print('WARNINIG: labels are missing!')
                elif length < 3:    
                    self.path = list_of_attributes[0]
                    self.label = list_of_attributes[1]
                    self.subset_label = 1 # only one set
                elif length < 4:
                    self.path = list_of_attributes[0]
                    self.label = list_of_attributes[1]
                    self.subset_label = list_of_attributes[2]
                elif length < 5:
                    self.path = list_of_attributes[0]
                    self.label = list_of_attributes[1]
                    self.subset_label = list_of_attributes[2]
                    self.description = list_of_attributes[3]
                else:
                    print('WARNINIG: sample format is not supported')  
        else:
            self.path = path
            self.label = label
            if subset_label is not None:
                 self.subset_label = subset_label
            else:
                self.subset_label = 1 # only one set
            if description is not None: self.description = description 
                
    def get_label(self):
        return self.label

    def get_subset_label(self):
        return self.subset_label 
    
    def get_path(self):
        return self.path


def load_file(path, ignore_header = True, show = False):
    """Load data samples from a file and return a list of DataSample objects"""
    data_samples = []
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        row_iterator = iter(reader)
        if ignore_header == True:
            next(row_iterator) # ignore header
        for row in row_iterator:
            data_samples.append(DataSample(row))
        if show == True: 
            for row in data_samples:
                print(row)
    csvfile.close()
    return data_samples

def save_to_file(data_samples, path, header):
    with open(path, mode = 'w') as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow(header)
        for sample in data_samples:
            csv_writer.writerow([sample.get_path(), sample.get_label(), sample.get_subset_label()])
            



def run(data_samples, skewness, noised = False, operation = OperationType.REPORT_ONLY):
    """ 
    Check the data samples for any skewness and print and collect some statistics on the way 

    :param DataSample data_samples: List of data sample objects, see DataSample
    :param OperationType operatio: Type of operation to perform
    :params skewness float: A maximum threshold of the skewness between the classes
    :return: True if successed, False otherwise
    :rtype: bool
    :raises None:
    """
    print('-- Report --')
    # collect details
    # not the leanest approach, duplicated storage for these lists, have to check how to use np with my DataSample class
    length = len(data_samples)
    labels = [data_sample.get_label() for data_sample in data_samples]
    subset_labels = [data_sample.get_subset_label() for data_sample in data_samples]
    
    unique_labels = np.unique(labels)
    unique_subset_lables = np.unique(subset_labels)

    # print them out
    print('Number of data samples is :' + str(length))
    print('Number of labels is : ' + str(len(unique_labels)))
    print('Number of subsets is : ' + str(len(unique_subset_lables)))

    # check files existence
    """ print('Checking files existence ...')
    for sample in data_samples:
        if(os.path.isfile(sample.get_path()) == False):
            print(sample.get_path() + '\t does not exist!')    """ 
    
    # check skewness in labels
    
    # 1 -  within the whole dataset
    labels_histogram, _ = np.histogram([int(i) for i in labels], bins = len(unique_labels))
    print('Number of samples per class:')
    for i in range(len(unique_labels)):
        print('\tClass:' + unique_labels[i] + '(' + str(labels_histogram[i]) + ')')

    print('Percentages:')    
    for i in range(len(unique_labels)):
        print('\tClass:' + unique_labels[i] + '(' + str( "{:.2f}".format(labels_histogram[i]/float(length))) + '%' +  ')')

    # 2 - within each set
    print('Sets Percentages:')  
    # collect percentages 
    #set_percentages = [[0 for x in range(len(unique_labels))] for y in range(len(unique_subset_lables))]  
    set_percentages = np.zeros((len(unique_subset_lables), len(unique_labels)))
    for i in range(len(unique_subset_lables)):
        fi = subset_labels.index(unique_subset_lables[i])
        li = len(subset_labels) - subset_labels[::-1].index(unique_subset_lables[i]) - 1
        subset_labels_histogram, _ = np.histogram([int(n) for n in labels[fi:li]], len(unique_labels))
        for j in range(len(unique_labels)):
            set_percentages[i,j] = subset_labels_histogram[j]/float(len(subset_labels[fi:li]))
            print('\tSet:' + unique_subset_lables[i] + ' Class:' + unique_labels[j] + '(' + str( "{:.2f}".format(set_percentages[i][j])) + '%' +  ')')

    if operation != OperationType.REPORT_ONLY:
        if operation == OperationType.REDUCE:
            min_label = np.min(set_percentages, axis = 1)
            for i in range(len(unique_subset_lables)):
                for j in range(len(unique_labels)):
                    diff = set_percentages[i,j] - min_label[i]
                    if diff > skewness:
                        # fix this subset
                        fi = subset_labels.index(unique_subset_lables[i])
                        li = len(subset_labels) - subset_labels[::-1].index(unique_subset_lables[i]) - 1
                        indexes = fi + np.where(np.array(labels[fi:li]) == unique_labels[j])[0] # class to reduce
                        # noising condition: random_noising < skewness
                        
                        reduction_amount  = (li - fi) * diff 
                        reduction_amount -=  reduction_amount * random.uniform(0, skewness)   # amount of samples to remove
                        rand_indxes = random.sample(indexes, int(reduction_amount)) # random smpling
                        
                        # update data strucutres
                        data_samples = [sample for sample in data_samples if sample not in [data_samples[k] for k in rand_indxes]]    
                        length = len(data_samples)
                        labels = [data_sample.get_label() for data_sample in data_samples]
                        subset_labels = [data_sample.get_subset_label() for data_sample in data_samples]
        elif operation == OperationType.REDUCE:
            pass
    return data_samples

# main routine, shall be then converted into args for the main function


data_samples = load_file('/home/ftam/Dropbox/splits_sets.txt', True, False) # load file
data_samples = run(data_samples, 0.25, operation = OperationType.REDUCE, noised= True) # process it
run(data_samples, skewness = 0.10, operation = OperationType.REPORT_ONLY) # verify the results
save_to_file(data_samples, '/home/ftam/Dropbox/splits_sets_new.txt', ["path", "class", "fold"])