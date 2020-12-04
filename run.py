import os
import csv


def train_pre():
    fptr = open('datasets/train.txt', 'r')
    csv_ptr = open('datasets/train.csv', 'w')
    csv_writer = csv.writer(csv_ptr)
    field = ['tweets', 'label']
    csv_writer.writerow(field)
    while fptr:
        data = fptr.readline()
        data = data.split()
        if len(data) == 0:
            break
        tweet = ' '.join(data[i] for i in range(2, len(data)))
        row = [tweet, data[1]]
        csv_writer.writerow(row)
    fptr.close()
    csv_ptr.close()


def test_pre():
    fptr = open('datasets/test.txt', 'r')
    csv_ptr = open('datasets/test.csv', 'w')
    csv_writer = csv.writer(csv_ptr)
    field = ['tweets', 'label']
    csv_writer.writerow(field)
    while fptr:
        data = fptr.readline()
        data = data.split()
        if len(data) == 0:
            break
        tweet = ' '.join(data[i] for i in range(2, len(data)))
        row = [tweet, data[1]]
        csv_writer.writerow(row)
    fptr.close()
    csv_ptr.close()
