import numpy as np
from math import *

# smoothing constant
# try it
k = 3

labels = []
with open("./data22/training_labels.txt") as label_file:
    labels = [int(n) for n in label_file.read().splitlines()]

high = np.zeros((5,30,13))
low = np.zeros((5,30,13))

def train_digit(filePath):
    spec_count = 0
    line_num = 0
    with open (filePath) as train_file :
        lines = train_file.read().splitlines()

        for line in lines:

            row = line_num % 33

            # if not in separate lines
            if row < 30:

                col = 0
                for c in line:
                    if c is ' ':
                        high[labels[spec_count]-1,row,col] += 1
                    else:
                        low[labels[spec_count]-1,row,col] += 1
                    col += 1

            # start with new spec
            if row is 30:
                spec_count += 1

            line_num += 1


# train  data
train_digit("./data22/training_data.txt")

high_likelihoods = (high + k) / ( 60 + k*5)
low_likelihoods = (low + k) / ( 60 + k*5)

p_class = np.zeros(5)

for i in range(5):
    p_class[i] = len([n for n in labels if n is (i+1) ])
p_class = p_class/sum(p_class)

log_p_class = np.log(p_class)



def test_digit(filePath):
    output = np.zeros(0, dtype=int)
    line_num = 0
    temp = np.zeros((30, 13))

    with open (filePath) as test_file :
        lines = test_file.read().splitlines()
        for line in lines:

            row = line_num % 33

            # if not in separate lines
            if row < 30:
                col = 0
                for c in line:
                    if c is ' ':
                        temp[row,col] = 1
                    else:
                        temp[row, col] = 0
                    col += 1


            # end of a spec, reset temp
            if row is 29:
                results = np.zeros(5)

                for i in range(5):
                    temp_i = temp * high_likelihoods[i] + (1-temp)*low_likelihoods[i]
                    results[i] = log_p_class[i] +  sum(sum( np.log(temp_i) ))

                output = np.append(output, np.argmax(results)+1 )

                temp = np.zeros((30, 13))

            line_num += 1
    return output


output = test_digit("./data22/testing_data.txt")

test_labels = []
with open("./data22/testing_labels.txt") as label_file:
    test_labels = [int(n) for n in label_file.read().splitlines()]


count = 0
for i in range(len(test_labels)):
    if test_labels[i] == output[i]:
        count+=1

print(count/len(test_labels))