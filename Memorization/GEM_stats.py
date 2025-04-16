import matplotlib.pyplot as plt
import numpy as np


def avg_95_5(sample_accuracies_baseline, sample_accuracies_most, sample_accuracies_least, title, ylabel):
    """
    Computes the average of the last 20 test accuracies for each confidence sample,
    then plots the average along with 95% confidence error bars for each experiment variant.
    """
    averages = []
    maxes = []
    mins = []
    labels = ['Baseline GEM', 'Selective GEM (Most)', 'Selective GEM (Least)']
    
    # Iterate over the three experiment conditions.
    for sample_accuracies in [sample_accuracies_baseline, sample_accuracies_most, sample_accuracies_least]:
        avg_total = 0
        average_list = []
        for j in range(CONFIDENCE_SAMPLES):
            # Assume sample_accuracies[j] is a list/array of test accuracies (e.g. 400 values reshaped as needed)
            last_20 = sample_accuracies[j][-20:]
            avg_val = np.mean(last_20)
            avg_total += avg_val
            average_list.append(avg_val)
        avg_total /= CONFIDENCE_SAMPLES
        averages.append(avg_total)
        maxes.append(np.percentile(average_list, 97.5))
        mins.append(np.percentile(average_list, 2.5))
    
    colors = ['blue', 'green', 'red']
    # Plot the average accuracies with error bars.
    plt.bar(labels, averages,
            yerr=[[averages[i] - mins[i] for i in range(3)], [maxes[i] - averages[i] for i in range(3)]],
            capsize=10, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Experiment Variant')
    plt.rcParams["figure.figsize"] = (3, 2)
    plt.grid(axis='y')
    # plt.savefig(title + '.png')
    plt.show()


def plot_ECEs(ece_baseline, ece_most, ece_least, title):
    """
    Plots the Expected Calibration Error (ECE) for each experiment variant with 95% confidence error bars.
    """
    colors = ['blue', 'green', 'red']
    labels = ['Baseline GEM', 'Selective GEM (Most)', 'Selective GEM (Least)']
    
    baseline_mean = np.mean(ece_baseline)
    most_mean = np.mean(ece_most)
    least_mean = np.mean(ece_least)
    
    baseline_lower = np.percentile(ece_baseline, 2.5)
    baseline_upper = np.percentile(ece_baseline, 97.5)
    most_lower = np.percentile(ece_most, 2.5)
    most_upper = np.percentile(ece_most, 97.5)
    least_lower = np.percentile(ece_least, 2.5)
    least_upper = np.percentile(ece_least, 97.5)
    
    plt.bar(labels, [baseline_mean, most_mean, least_mean],
            yerr=[[baseline_mean - baseline_lower, most_mean - most_lower, least_mean - least_lower],
                  [baseline_upper - baseline_mean, most_upper - most_mean, least_upper - least_mean]],
            capsize=10, color=colors)
    plt.title(title)
    plt.ylabel('ECE')
    plt.xlabel('Experiment Variant')
    plt.grid(axis='y')
    # plt.savefig('ece.png')
    plt.show()


def expected_callibration_error(accuracies, confidences):
    """
    Computes the expected calibration error (ECE) given arrays of accuracies and confidences.
    """
    ece = 0
    for i in range(len(accuracies)):
        ece += abs(accuracies[i] - confidences[i])
    ece /= len(accuracies)
    return ece


def forgetting(sample_accuracies):
    """
    Computes a forgetting measure.
    Assumes sample_accuracies is a 400-element list which is reshaped to (N_TASKS, N_TASKS).
    For each task, we take the maximum accuracy observed across training phases
    and compute the drop in accuracy at the end.
    """
    sample_array = np.array(sample_accuracies).reshape(N_TASKS, N_TASKS)
    max_values = np.amax(sample_array, axis=0)
    
    # Use the final training phase (last row) for all tasks.
    final_accuracies = sample_array[N_TASKS - 1]
    forgetting_measure = [abs(max_values[i] - final_accuracies[i]) for i in range(N_TASKS)]
    return np.mean(forgetting_measure)


def forward_transfer(sample_accuracies):
    """
    Computes the forward transfer. For each task (except the first),
    we measure the difference in test accuracy on that task before and after training on it.
    """
    sample = np.array(sample_accuracies).reshape(N_TASKS, N_TASKS)
    forward_transfers = []
    for i in range(1, N_TASKS):
        forward_transfers.append(sample[i - 1][i] - sample[0][i])
    return np.mean(forward_transfers)


def plot_forward_transfer(sample_accuracies_baseline, sample_accuracies_most, sample_accuracies_least):
    """
    Plots the forward transfer metric for each experiment variant with 95% confidence error bars.
    """
    forward_transfer_baseline = []
    forward_transfer_most = []
    forward_transfer_least = []
    
    for i in range(CONFIDENCE_SAMPLES):
        forward_transfer_baseline.append(forward_transfer(sample_accuracies_baseline[i]))
        forward_transfer_most.append(forward_transfer(sample_accuracies_most[i]))
        forward_transfer_least.append(forward_transfer(sample_accuracies_least[i]))
    
    colors = ['blue', 'green', 'red']
    labels = ['Baseline GEM', 'Selective GEM (Most)', 'Selective GEM (Least)']
    
    baseline_mean = np.mean(forward_transfer_baseline)
    most_mean = np.mean(forward_transfer_most)
    least_mean = np.mean(forward_transfer_least)
    
    baseline_lower = np.percentile(forward_transfer_baseline, 2.5)
    baseline_upper = np.percentile(forward_transfer_baseline, 97.5)
    most_lower = np.percentile(forward_transfer_most, 2.5)
    most_upper = np.percentile(forward_transfer_most, 97.5)
    least_lower = np.percentile(forward_transfer_least, 2.5)
    least_upper = np.percentile(forward_transfer_least, 97.5)
    
    plt.bar(labels, [baseline_mean, most_mean, least_mean],
            yerr=[[baseline_mean - baseline_lower, most_mean - most_lower, least_mean - least_lower],
                  [baseline_upper - baseline_mean, most_upper - most_mean, least_upper - least_mean]],
            capsize=10, color=colors)
    plt.title('Forward Transfer')
    plt.ylabel('Forward Transfer')
    plt.xlabel('Experiment Variant')
    plt.grid(axis='y')
    # plt.savefig('forward_transfer.png')
    plt.show()


def backward_transfer(sample_accuracies):
    """
    Computes the backward transfer for a given experiment condition.
    For each task (except the last), we calculate the drop in test accuracy
    from when that task was trained to after the final task.
    """
    sample = np.array(sample_accuracies).reshape(N_TASKS, N_TASKS)
    backward_transfers = []
    for i in range(N_TASKS - 1):
        backward_transfers.append(sample[N_TASKS - 1][i] - sample[i][i])
    return np.mean(backward_transfers)


def plot_backward_transfer(sample_accuracies_baseline, sample_accuracies_most, sample_accuracies_least):
    """
    Plots the backward transfer metric for each experiment variant with 95% confidence error bars.
    """
    backward_transfer_baseline = []
    backward_transfer_most = []
    backward_transfer_least = []
    
    for i in range(CONFIDENCE_SAMPLES):
        backward_transfer_baseline.append(backward_transfer(sample_accuracies_baseline[i]))
        backward_transfer_most.append(backward_transfer(sample_accuracies_most[i]))
        backward_transfer_least.append(backward_transfer(sample_accuracies_least[i]))
    
    colors = ['blue', 'green', 'red']
    labels = ['Baseline GEM', 'Selective GEM (Most)', 'Selective GEM (Least)']
    
    baseline_mean = np.mean(backward_transfer_baseline)
    most_mean = np.mean(backward_transfer_most)
    least_mean = np.mean(backward_transfer_least)
    
    baseline_lower = np.percentile(backward_transfer_baseline, 2.5)
    baseline_upper = np.percentile(backward_transfer_baseline, 97.5)
    most_lower = np.percentile(backward_transfer_most, 2.5)
    most_upper = np.percentile(backward_transfer_most, 97.5)
    least_lower = np.percentile(backward_transfer_least, 2.5)
    least_upper = np.percentile(backward_transfer_least, 97.5)
    
    plt.bar(labels, [baseline_mean, most_mean, least_mean],
            yerr=[[baseline_mean - baseline_lower, most_mean - most_lower, least_mean - least_lower],
                  [baseline_upper - baseline_mean, most_upper - most_mean, least_upper - least_mean]],
            capsize=10, color=colors)
    plt.title('Backward Transfer')
    plt.ylabel('Backward Transfer')
    plt.xlabel('Experiment Variant')
    plt.grid(axis='y')
    # plt.savefig('backward_transfer.png')
    plt.show()


def plot_forgetting(sample_accuracies_baseline, sample_accuracies_most, sample_accuracies_least, title):
    """
    Plots the forgetting measure for each experiment variant with 95% confidence error bars.
    """
    forgetting_baseline = []
    forgetting_most = []
    forgetting_least = []
    
    for i in range(CONFIDENCE_SAMPLES):
        forgetting_baseline.append(forgetting(sample_accuracies_baseline[i]))
        forgetting_most.append(forgetting(sample_accuracies_most[i]))
        forgetting_least.append(forgetting(sample_accuracies_least[i]))
    
    colors = ['blue', 'green', 'red']
    labels = ['Baseline GEM', 'Selective GEM (Most)', 'Selective GEM (Least)']
    
    baseline_mean = np.mean(forgetting_baseline)
    most_mean = np.mean(forgetting_most)
    least_mean = np.mean(forgetting_least)
    
    baseline_lower = np.percentile(forgetting_baseline, 2.5)
    baseline_upper = np.percentile(forgetting_baseline, 97.5)
    most_lower = np.percentile(forgetting_most, 2.5)
    most_upper = np.percentile(forgetting_most, 97.5)
    least_lower = np.percentile(forgetting_least, 2.5)
    least_upper = np.percentile(forgetting_least, 97.5)
    
    plt.bar(labels, [baseline_mean, most_mean, least_mean],
            yerr=[[baseline_mean - baseline_lower, most_mean - most_lower, least_mean - least_lower],
                  [baseline_upper - baseline_mean, most_upper - most_mean, least_upper - least_mean]],
            capsize=10, color=colors)
    plt.title(title)
    plt.ylabel('Forgetting Measure')
    plt.xlabel('Experiment Variant')
    plt.grid(axis='y')
    # plt.savefig('forgetting.png')
    plt.show()
