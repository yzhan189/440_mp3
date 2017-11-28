from AudioTrain import *

yes_high = np.zeros((25,10))
yes_low = np.zeros((25,10))

no_high = np.zeros((25,10))
no_low = np.zeros((25,10))

yes_total = train_yes_no("./yesno/yes_train.txt",yes_high,yes_low)
no_total = train_yes_no("./yesno/no_train.txt",no_high,no_low)


YES_H = (yes_high + k) / (yes_total + k*2)
NO_H = (no_high + k) / (no_total + k*2)

YES_L= (yes_low + k) / (yes_total + k*2)
NO_L = (no_low + k) / (no_total + k*2)
