import numpy as np
from math import *
import os
from seg import *

# [no, yes]
high = np.zeros((2,25,10))
low = np.zeros((2,25,10))

# [no_num, yes_num]
total = np.zeros(2)

# smoothing constant
# try it
k = 3


def train_yes_no( ):

    path = "./txt_yesno/training"
    for fileName in os.listdir(path):
        filePath = path+'/'+fileName

        # number of no
        total[0] += fileName.count('0')
        # number of yes
        total[1] += fileName.count('1')
        # list of yes and no labels
        spec_labels = [int(str) for str in fileName.split('.')[0].split('_')]

        with open (filePath) as train_file :

            lines = train_file.read().splitlines()
            max = -np.inf
            temp_max = np.zeros((8,25,10))
            temp = np.zeros((8,25,10))

            # try plausible trimming the head and interval numbers
            for trim_head in range(22,25):
                for interval in [1,2,3]:

                    line_num = 0

                    for line in lines:
                        chunks = [line[ trim_head+i*(10+interval):trim_head+i*(10+interval)+10 ] for i in range(8)]

                        spec_num = 0

                        for chunk in chunks:
                            col = 0
                            for c in chunk:
                                if c is ' ':
                                    temp[spec_num,line_num, col] = 1
                                else :
                                    temp[spec_num,line_num, col] = 0
                                col += 1
                            spec_num += 1

                        line_num += 1

                    curr_result = 0
                    for i in range(8):
                        temp_yes = temp[i] * yes_high_likelihoods + (1 - temp[i]) * yes_low_likelihoods
                        yes_prob = log_p_yes + sum(sum(np.log(temp_yes)))

                        temp_no = temp[i] * no_high_likelihoods + (1 - temp[i]) * no_low_likelihoods
                        no_prob = log_p_no + sum(sum(np.log(temp_no)))

                        if spec_labels[i] is 0:
                            curr_result += (no_prob-yes_prob)
                        else :
                            curr_result += (yes_prob-no_prob)

                    if (curr_result > max):
                        max = curr_result
                        temp_max = temp

            # by this point you should get best segment
            # do count
            for i in range(8):

                high[spec_labels[i]] = high[spec_labels[i]]  + sum(temp_max)
                low[spec_labels[i]]  = low[spec_labels[i]]  + (sum(1-temp_max))


# training
train_yes_no()

yes_total = total[1]
no_total = total[0]


yes_high_likelihoods = (high[1] + k) / (yes_total + k*2)
no_high_likelihoods = (high[0] + k) / (no_total + k*2)

yes_low_likelihoods = (low[1] + k) / (yes_total + k*2)
no_low_likelihoods = (low[0] + k) / (no_total + k*2)


# log prob yes/ no
log_p_yes = np.log(yes_total / (yes_total+no_total))
log_p_no = np.log(no_total / (yes_total+no_total))








# testing part
def test_yes_no(path):

    output = np.zeros(0, dtype=int)

    for fileName in os.listdir(path):
        filePath = path+'/'+fileName


        with open (filePath) as test_file :
            line_num = 0
            temp = np.zeros((25, 10))

            lines = test_file.read().splitlines()
            for line in lines:

                # if not in separate lines
                col = 0
                for c in line:
                    if c is ' ':
                        temp[line_num,col] = 1
                    else:
                        temp[line_num, col] = 0
                    col += 1


                temp_yes = temp * yes_high_likelihoods + (1-temp)*yes_low_likelihoods
                yes_prob = log_p_yes +  sum(sum( np.log(temp_yes) ))

                temp_no = temp * no_high_likelihoods + (1-temp)*no_low_likelihoods
                no_prob = log_p_no +  sum(sum( np.log(temp_no) ))


                output = np.append(output, 1 ) if yes_prob > no_prob else np.append(output, 0 )

                line_num += 1

    return output



output = test_yes_no("./txt_yesno/yes_test")
print(sum(output)/len(output))

output = test_yes_no("./txt_yesno/no_test")
print(1-sum(output)/len(output))