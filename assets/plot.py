import csv
import matplotlib.pyplot as plt

filename = 'mujoco_a3c_joint_4_run_1.csv'

# read the csv file
with open(filename, 'r') as f:
    reader = csv.reader(f)
    # skip the first row
    next(reader)
    # initialize empty lists for column 1 and column 3 data
    column1 = []
    column3 = []
    # loop through each row in the csv file
    for row in reader:
        # append the data to the appropriate lists
        column1.append(float(row[0]))
        column3.append(float(row[1]))

# plot the data
plt.plot(column1, column3)
plt.title('J4 tuning')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()
