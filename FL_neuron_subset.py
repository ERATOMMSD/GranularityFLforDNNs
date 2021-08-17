import os
import csv
from collections import defaultdict
from results_analysis.plotter import *


def intersections(dataset):
    if dataset == "mnist":
        nets = ["mnist_test_model_5_30_leaky_relu",
                "mnist_test_model_6_25_leaky_relu",
                "mnist_test_model_8_20_leaky_relu"]
        net_names = ["MNIST_1", "MNIST_2", "MNIST_3"]
        SNs = [1, 2, 3, 5, 10, 15]
        shortNNs = ["5_30_leaky", "6_25_leaky", "8_20_leaky"]
        layers = [[1, 3, 5, 7, 9], [1, 3, 5, 7, 9, 11], [1, 3, 5, 7, 9, 11, 13, 15]]
        idxs_x_layer = [30, 25, 20]
        methods = ["tarantula", "ochiai", "dstar", "random"]
    else:
        dataset = "cifar10"
        nets = ["cifar10_test_model_4_128_leaky_relu-normalized",
                "cifar10_test_model_2_256_leaky_relu-normalized",
                "cifar10_test_model_1_512_leaky_relu-normalized"
                ]
        net_names = ["CIFAR_1", "CIFAR_2", "CIFAR_3"]

        methods = ["tarantula", "ochiai", "dstar", "random"]
        SNs = [10, 20, 30, 40, 50, 70]
    """
    ========================================
    Subset intersection experiment
    ========================================
    """
    if not os.path.exists("results_analysis/intersections"):
        os.makedirs("results_analysis/intersections")
    if not os.path.exists("results_analysis/csv_values"):
        os.makedirs("results_analysis/csv_values")
    if not os.path.exists("results_analysis/csv_values/intersections"):
        os.makedirs("results_analysis/csv_values/intersections")

    rec_dd = lambda: defaultdict(rec_dd)
    all_intersections = defaultdict(rec_dd)
    for metric in methods:
        for K in SNs:
            heatmaps = []
            filepath = f"experiment_results/{dataset}/SN{K}_{metric}/"
            for net, net_name in zip(nets, net_names):
                print(f"{net} started")
                sets = {}
                ans = {}
                for c in range(0, 10):
                    sets[c] = {}  # the most suspicious neurons for each C, MC
                    ans[c] = {}  # amount of suspicious neurons in intersection of C, MC and C, -1
                    for mc in range(-1, 10):
                        file_name = filepath + f'{net}_C{c}_MC{mc}_{metric}_SN{K}.csv'
                        try:
                            with open(file_name, 'r', newline='') as csvfile:
                                reader = csv.reader(csvfile, delimiter=',')
                                current = set()
                                neuron = next(reader)
                                amount = K
                                for i in range(amount):
                                    neuron = next(reader)
                                    current.add((neuron[0], neuron[1]))
                                sets[c][mc] = current
                        except FileNotFoundError:
                            print(f"no csv file for net={net}, C={c} and MC={mc}")
                            continue
                    # check inclusion of sets with the current c
                    for mc in range(0, 10):
                        if mc == c:
                            continue
                        ans[c][mc] = len(sets[c][-1].intersection(sets[c][mc]))
                        print(f"for {net}, C={c} and MC={mc} there are {ans[c][mc]} out of {K} neurons " +
                              f"included in top {K} neurons of  MC -1")

                file_name = filepath + f'0FL_Table_{net}.csv'
                try:
                    with open(file_name, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow(["C\\MC", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
                        for c in range(0, 10):
                            row = [c]
                            for mc in range(0, 10):
                                if c == mc:
                                    row.append(0)
                                else:
                                    row.append(ans[c][mc])
                            writer.writerow(row)
                        writer.writerow(["\n"])
                        writer.writerow(["each value in the table is the intersection size of top 5 of that C/MC with C/-1"])
                except FileNotFoundError:
                    print(f"cant write FL table for net={shortNN}, C={c} and MC={mc}")
                    assert False

                current_matrix = np.zeros((10, 10))
                for c in range(10):
                    for mc in range(10):
                        if c == mc:
                            current_matrix[c][mc] = 0
                        else:
                            current_matrix[c][mc] = ans[c][mc]
                heatmaps.append(current_matrix)

                all_intersections[metric][K][net_name] = current_matrix

            for i in range(len(net_names)):
                plt.close("all")
                values = [heatmaps[i]]
                names = [["MC", "C", net_names[i]]]
                #suptitle = f"intersection of top {K} FL neurons for model {net_names[i]} and metric {metric}"
                plot_3d_bars(values, names, (1, 1), None) #suptitle)
                multipage("results_analysis/intersections/" + f"{net_names[i]}_{metric}_{K}_intersection_heatmaps.pdf")

                # csv_names = [f"results_analysis/csv_values/intersections/{net_names[i]}_{metric}_{K}_intersection_heatmaps.csv"]
                # save_csv_vals(values, csv_names, complete_name=True)

    # use all_intersections to build summary table
    for net_name in net_names:
        current_matrix = np.zeros((len(SNs)*4, len(methods)))
        for i in range(len(SNs)):
            for j in range(len(methods)):
                current_matrix[(i*4)][j] = all_intersections[methods[j]][SNs[i]][net_name].mean()
                current_matrix[(i*4)+1][j] = all_intersections[methods[j]][SNs[i]][net_name].min()
                current_matrix[(i*4)+2][j] = all_intersections[methods[j]][SNs[i]][net_name].max()
                current_matrix[(i*4)+3][j] = all_intersections[methods[j]][SNs[i]][net_name].std()
        csv_names = [f"results_analysis/csv_values/intersections/{net_name}_intersection_summary.csv"]
        save_csv_vals([current_matrix], csv_names, complete_name=True)

