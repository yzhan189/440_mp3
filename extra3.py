import numpy as np
from math import *

# now there are 11 possible features
# feature[i,j] means the count of ( j-th row's average is i/10 )
yes_feature = np.zeros((11,25))
no_feature = np.zeros((11,25))


# smoothing constant
# try it
k = 3


def train_yes_no(filePath, feature):
    total = 0
    line_num = 0
    with open (filePath) as train_file :
        lines = train_file.read().splitlines()

        for line in lines:

            row = line_num % 28

            # start with new spec
            if row is 0:
                total += 1

            # if not in separate lines
            if row < 25:
                index = line.count(' ')
                feature[index,row] += 1

            line_num += 1


    return total

# train yes and no data
yes_total = train_yes_no("./yesno/yes_train.txt",yes_feature)
no_total = train_yes_no("./yesno/no_train.txt",no_feature)



# P( yes )
p_yes = yes_total / (yes_total+no_total)
# P( no )
p_no = no_total / (yes_total+no_total)

log_p_yes = np.log(p_yes)
log_p_no = np.log(p_no)


# calculate likelihoods with smoothing
# P( F | yes )
yes_likelihoods = (yes_feature + k) / (yes_total + k*2)
# P( F | no )
no_likelihoods = (no_feature + k) / (no_total + k*2)






# testing part
def test_yes_no(filePath):
    output = np.zeros(0, dtype=int)
    line_num = 0
    temp = np.zeros(25,dtype=int)

    with open (filePath) as test_file :

        lines = test_file.read().splitlines()

        for line in lines:

            row = line_num % 28

            # if not in separate lines
            if row < 25:
                # note it is the sum, not average
                # this is also the index
                temp[row] = line.count(' ')

            # end of a spec, reset temp
            if row is 24:

                # choose the right feature (index), and right row (i+1)
                temp_yes = np.array([ yes_likelihoods[index] for index in temp ]).diagonal()
                yes_prob = log_p_yes +  sum( np.log(temp_yes) )


                temp_no =  np.array([ no_likelihoods[index] for index in temp ]).diagonal()
                no_prob = log_p_no +  sum( np.log(temp_no) )

                output = np.append(output, 1 ) if yes_prob > no_prob else np.append(output, 0 )

                temp = np.zeros(25,dtype=int)


            line_num += 1
    return output





output1 = test_yes_no("./yesno/yes_test.txt")
yesyes = sum(output1)/len(output1)

output2 = test_yes_no("./yesno/no_test.txt")
nono = (1-sum(output2)/len(output2))

print("Accuracy:")
print( (sum(output1)/len(output1) + (1-sum(output2)/len(output2)))/2 )

print('\n             predicted yes     predicted no')
print('actual yes   '+str(yesyes)+'             '+str(1-yesyes))
print('actual no    '+str(1-nono)+'             '+str(nono))
