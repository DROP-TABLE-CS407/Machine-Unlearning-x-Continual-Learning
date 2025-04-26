import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# make 1 BIG figure which will contain all the plots 4x5 grid, each subplot will be 1 of the different tasks accuracy as it runs
plt.figure(figsize=(20, 15))
plt.suptitle('Task test set accuracies for each task as it runs', fontsize=20)

import matplotlib.cm as cm

# Define a colormap to get distinct colors for each task
colors = cm.rainbow(np.linspace(0, 1, 20))


# READ FROM ALL_TASK_ACCURACIES_AVERAGED.csv
# Read the CSV file
df = pd.read_csv('ALL_TASK_ACCURACIES_AVERAGED.csv')
# Convert the DataFrame to a list of lists

# remove the first column (iteration number) from the DataFrame
df = df.drop(df.columns[0], axis=1)

ALL_TASK_ACCURACIES_AVERAGED =  df.values.tolist()
# Print the shape of the DataFrame
print("Shape of DataFrame:", df.shape)
# Print the first few rows of the DataFrame
print("First few rows of DataFrame:")
print(df.head())
# Print the last few rows of the DataFrame
print("Last few rows of DataFrame:")
print(df.tail())

for i in range(20):
    plt.subplot(4, 5, i + 1)
    # Extract task i accuracy at each iteration (column-wise)
    task_i_accuracies = [iteration[i] for iteration in ALL_TASK_ACCURACIES_AVERAGED]
    plt.plot(task_i_accuracies, color=colors[i])
    plt.title('Task ' + str(i + 1))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid()
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust the top to make room for the suptitle
plt.savefig('ALL_TASK_ACCURACIES.png')
plt.show()
# save the figure
plt.savefig('ALL_TASK_ACCURACIES.png', dpi=600, transparent=True)
plt.close()