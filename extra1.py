import numpy as np
from math import *
import os

# [no, yes]
high = np.zeros((2,25,10))
low = np.zeros((2,25,10))

# [no_num, yes_num]
total = np.zeros(2)

# smoothing constant
# try it
k = 5


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

            heads = np.zeros(8,dtype=int)


            spec_num = 0
            while (spec_num<8):
                starts = np.zeros(25, dtype=int)
                i = 0

                for line in lines:
                    adding = 0

                    if spec_num ==0 :
                        index = line.find('   ')
                    else:
                        index = line[(heads[spec_num-1]+10):].find("   ")
                        adding = (heads[spec_num-1]+10)

                    if index == -1:
                        starts[i] = 200
                    else:
                        starts[i] = index+adding

                    i +=1


                heads[spec_num] = np.sort(starts)[1]
                spec_num +=1
            if heads[7] ==200:
                heads[7] = heads[6]

            # extract
            temp = np.zeros((8,25,10))

            for spec_num in range(8):
                head = heads[spec_num]
                for row in range(25):
                    line = lines[row][head:(head+10)]

                    for col in range(10):
                        if line[col] == ' ':
                            temp[spec_num,row,col] = 1
                        else:
                            temp[spec_num, row, col] = 0




            for i in range(8):
                high[spec_labels[i]] = high[spec_labels[i]] + temp[i]
                low[spec_labels[i]] = low[spec_labels[i]] + ((1 - temp[i]))




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

                line_num += 1


            temp_yes = temp * yes_high_likelihoods + (1-temp)*yes_low_likelihoods
            yes_prob = log_p_yes +  sum(sum( np.log(temp_yes) ))

            temp_no = temp * no_high_likelihoods + (1-temp)*no_low_likelihoods
            no_prob = log_p_no +  sum(sum( np.log(temp_no) ))


            output = np.append(output, 1 ) if yes_prob > no_prob else np.append(output, 0 )



    return output



output1 = test_yes_no("./txt_yesno/yes_test")
print(len(output1))
yesyes = sum(output1)/len(output1)

output2 = test_yes_no("./txt_yesno/no_test")
print(len(output2))
nono = (1-sum(output2)/len(output2))

print("Accuracy:")
print( (sum(output1)/len(output1) + (1-sum(output2)/len(output2)))/2 )

print('\n             predicted yes     predicted no')
print('actual yes   '+str(yesyes)+'             '+str(1-yesyes))
print('actual no    '+str(1-nono)+'             '+str(nono))