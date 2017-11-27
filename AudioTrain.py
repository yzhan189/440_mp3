import numpy as np
from math import *


yes_high = np.zeros((25,10))
yes_low = np.zeros((25,10))

no_high = np.zeros((25,10))
no_low = np.zeros((25,10))

# smoothing constant
# try it
k = 3


def train_yes_no(filePath, high,low):
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
                col = 0
                for c in line:
                    if c is ' ':
                        high[row,col] += 1
                    else:
                        low[row,col] += 1
                    col += 1

            line_num += 1

    return total

# train yes and no data
yes_total = train_yes_no("./yesno/yes_train.txt",yes_high,yes_low)
no_total = train_yes_no("./yesno/no_train.txt",no_high,no_low)


# calculate likelihoods with smoothing
# P( high | yes )
yes_high_likelihoods = (yes_high + k) / (yes_total + k*2)
# P( high | no )
no_high_likelihoods = (no_high + k) / (no_total + k*2)

yes_low_likelihoods = (yes_low + k) / (yes_total + k*2)
no_low_likelihoods = (no_low + k) / (no_total + k*2)


#print(yes_high_likelihoods)

# P( yes )
p_yes = yes_total / (yes_total+no_total)
# P( no )
p_no = no_total / (yes_total+no_total)

log_p_yes = log10(p_yes)
log_p_no = log10(p_no)



# testing part
def test_yes_no(filePath):
    output = np.zeros(0, dtype=int)
    line_num = 0
    temp = np.zeros((25, 10))

    with open (filePath) as test_file :
        lines = test_file.read().splitlines()
        for line in lines:

            row = line_num % 28

            # if not in separate lines
            if row < 25:
                col = 0
                for c in line:
                    if c is ' ':
                        temp[row,col] = 1
                    else:
                        temp[row, col] = 0
                    col += 1


            # end of a spec, reset temp
            if row is 24:
                temp_yes = temp * yes_high_likelihoods + (1-temp)*yes_low_likelihoods
                yes_prob = log_p_yes +  sum(sum( np.log(temp_yes) ))

                temp_no = temp * no_high_likelihoods + (1-temp)*no_low_likelihoods
                no_prob = log_p_no +  sum(sum( np.log(temp_no) ))


                output = np.append(output, 1 ) if yes_prob > no_prob else np.append(output, 0 )

                temp = np.zeros((25, 10))

            line_num += 1
    return output

output = test_yes_no("./yesno/yes_test.txt")
print(sum(output)/len(output))

output = test_yes_no("./yesno/no_test.txt")
print(1-sum(output)/len(output))
