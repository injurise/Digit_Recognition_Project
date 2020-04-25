#!/usr/bin/python3
#test edit
import csv

filepath = "dataset/dataset_sample.csv"

with open(filepath, mode='r') as dataset:
    csv_reader = csv.DictReader(dataset)
    line_count = 0

    for row in csv_reader:

        if line_count == 0:
            print("first row are headers")
            line_count += 1

        print("every other row is the data of the 28x28 pixel images")
        line_count += 1
    
    print("Images detected in dataset: "+ str(line_count))

print("test submission")
