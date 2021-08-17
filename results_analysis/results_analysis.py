import csv
from collections import defaultdict
import re
from os import path, makedirs, mkdir

import pingouin as pt
from scipy.stats import wilcoxon

from .make_tex_tables import make_tex, t_test_table
from .plotter import *


def analyze(dataset_name):
    if not path.exists("csv_values"):
        makedirs("csv_values")
    if not path.exists("heatmaps_per_configuration"):
        makedirs("heatmaps_per_configuration")
    if not path.exists("intersections"):
        makedirs("intersections")
    if not path.exists("tables"):
        makedirs("tables")

    if dataset_name == "mnist":
        # For MNIST: =======================
        filepath = "../experiment_results/mnist/"
        save_file = "mnist_results.pdf"
        dataset = "MNIST"

        methods = ["tarantula", "ochiai", "dstar", "random"]
        SNs = [1, 2, 3, 5, 10, 15]

        nets = ["mnist_test_model_5_30_leaky_relu",
                "mnist_test_model_6_25_leaky_relu",
                "mnist_test_model_8_20_leaky_relu"]
        net_names = ["MNIST_1", "MNIST_2", "MNIST_3"]
    else:
        # For CIFAR10 =======================
        filepath = "../experiment_results/cifar10/"
        save_file = "cifar_results.pdf"
        dataset = "CIFAR"

        methods = ["tarantula", "ochiai", "dstar", "random"]
        SNs = [10, 20, 30, 40, 50, 70]
        nets = ["cifar10_test_model_4_128_leaky_relu-normalized",
                "cifar10_test_model_2_256_leaky_relu-normalized",
                "cifar10_test_model_1_512_leaky_relu-normalized"
                ]
        net_names = ["CIFAR_1", "CIFAR_2", "CIFAR_3"]

    def build_dataframe():
        rec_dd = lambda: defaultdict(rec_dd)
        results = defaultdict(rec_dd)
        results_df = []
        for method in methods:
            for K in SNs:
                for net in nets:
                    for C in range(10):
                        for MC in range(-1, 10):
                            if C == MC:
                                continue
                            filename = filepath + f"SN{K}_{method}/{net}_C{C}_MC{MC}_{method}_SN{K}.csv"
                            with open(filename) as f:
                                # read contents
                                content = f.read().splitlines()
                                column_names = content[0]
                                suspicious_neurons = content[1:K + 1]
                                original_scores = content[1 + K]
                                suspicious_mutation_scores = content[1 + K + 1]
                                suspicious_predictions = content[1 + K + 2]
                                if MC != -1:
                                    guided_synthesis_amount_suspicious = content[1 + K + 3]

                                # parse each line
                                neurons = [[float(f) for f in neuron.split(",")] for neuron in suspicious_neurons]
                                neuron_0 = neurons[0]
                                if method == "random":
                                    amount_faulty = 0  # we dont save this result for random method since there is no spectrum
                                    amount_correct = 0
                                else:
                                    amount_faulty = neuron_0[3] + neuron_0[4]
                                    amount_correct = neuron_0[5] + neuron_0[6]
                                match = re.findall("[+-]?(?:[0-9]*[.])?[0-9]+(?:e[+-]?[0-9]+)?", original_scores)
                                original_scores = [float(match[0]), float(match[1])]
                                match = re.findall("[+-]?(?:[0-9]*[.])?[0-9]+(?:e[+-]?[0-9]+)?", suspicious_mutation_scores)
                                suspicious_mutation_scores = [float(match[0]), float(match[1])]
                                match = re.findall("[0-9]", suspicious_predictions)
                                suspicious_predictions = [int(i) for i in match]
                                if MC != -1:
                                    guided_synthesis_amount_suspicious = int(guided_synthesis_amount_suspicious[0])
                                else:
                                    guided_synthesis_amount_suspicious = -1

                                # save results
                                current_results = {'neurons': neurons,
                                                   'orig_scores': original_scores,
                                                   'sus_scores': suspicious_mutation_scores,
                                                   'sus_pred': suspicious_predictions,
                                                   'num_wrong_inputs': amount_faulty,
                                                   'num_right_inputs': amount_correct,
                                                   'guided_sus_pred': guided_synthesis_amount_suspicious}
                                results[method][K][net][C][MC] = current_results
                                current_results.update({"method": method, "K": K, "net": net, "C": C, "MC": MC})
                                results_df.append(current_results)
        df = pd.DataFrame(results_df, columns=["method", "K", "net", "C", "MC", 'neurons', 'orig_scores',
                                               'sus_scores', 'sus_pred', 'num_wrong_inputs', 'num_right_inputs',
                                               'guided_sus_pred'])
        if not path.exists('csv_values/general_raw_dataframe'):
            mkdir("csv_values/general_raw_dataframe")
        df.to_csv(f"csv_values/general_raw_dataframe/{dataset}_whole_dataframe.csv", index=False)
        return df

    # ================================
    # Analyze the results
    df = build_dataframe()

    # table of number of original and synthesized errors
    def get_DeepFault_synthesis(selected_property, K=None, dataframe=df, method=None, net=None):
        result = np.zeros((10, 10))
        if K is not None:
            dataframe = dataframe.loc[dataframe['K'] == K]
        if method is not None:
            dataframe = dataframe.loc[dataframe['method'] == method]
        if net is not None:
            dataframe = dataframe.loc[dataframe['net'] == net]
        for C in range(10):
            current = dataframe.loc[(dataframe['C'] == C) & (dataframe['MC'] == -1)]
            current = current[selected_property]
            current_res = [sum([lst.count(i) for lst in current]) for i in range(10)]
            current_res = np.true_divide(current_res, len(current))
            current_res[C] = 0
            result[C] = current_res
        return result

    def heatmap_for(selected_property, net, K, method, dataframe=df):
        result = np.zeros((10, 10))
        if net is not None:
            dataframe = dataframe.loc[dataframe['net'] == net]
        if K is not None:
            dataframe = dataframe.loc[dataframe['K'] == K]
        if method is not None:
            dataframe = dataframe.loc[dataframe['method'] == method]
        for C in range(10):
            for MC in range(0, 10):
                if C == MC:
                    continue
                else:
                    current = dataframe.loc[(dataframe['C'] == C) & (dataframe['MC'] == MC)]
                    assert len(current[selected_property]) == 1
                    result[C][MC] = current[selected_property].mean()
        return result

    def plot_all_heatmaps_without_averaging(name, selected_property, ignore_no_data_failures):
        for net, net_name in zip(nets, net_names):
            for K in SNs:
                for method in methods:
                    if name == "DF":
                        current = get_DeepFault_synthesis(selected_property, K=K, method=method, net=net)
                    else:
                        current = heatmap_for(selected_property, net, K, method)
                    if ignore_no_data_failures:
                        origs = heatmap_for('num_wrong_inputs', net, SNs[0], "tarantula")  # get orig_failures
                        mask = (origs == 0)
                        for c in range(10):
                            mask[c][c] = False
                        current[mask] = None  # mask cases(C,MC) which had no original faulty inputs
                    plt.close("all")
                    names = [["MC", "C", ""]]
                    plot_heatmaps([current], names, (1, 1), None, vmaxs=[10], mask_zeros=[True])
                    multipage("heatmaps_per_configuration/" +
                              f"{name}_{net_name}_{method}_{K}_amount_synthesized.pdf")

    def build_failure_types_dataframe():
        # count how many types of failures we generate. To t-test if granularity gives more failures
        counted_failures = []
        for method in methods:
            for K in SNs:
                for net in nets:
                    for C in range(10):
                        current = df.loc[(df['method'] == method) & (df['K'] == K) & (df['net'] == net) & (df['C'] == C)]
                        current_results = {"method": method, "K": K, "net": net, "C": C}
                        # deep fault
                        current_deep_fault = current.loc[(current['MC'] == -1)]['sus_pred'].values[0]
                        fails_df = np.unique(current_deep_fault)  # count number of different failures synthesized
                        num_fails_df = len(fails_df)
                        if C in fails_df:  # synthesizing an input predicted as the original class is not a failure
                            num_fails_df -= 1
                        current_results.update({f'DeepFault_sus': num_fails_df})

                        # original failures
                        orig_fails_foreach_mc = 0
                        for MC in range(10):  # count how many MCs have at least one originally faulty input
                            if MC == C:
                                continue
                            current_orig_MC = current.loc[(current['MC'] == MC)]['num_wrong_inputs'].values[0]
                            if current_orig_MC != 0:
                                orig_fails_foreach_mc += 1
                        current_results.update({f'orig_fail_types': orig_fails_foreach_mc})

                        # granular approach
                        fails_foreach_mc = 0
                        for MC in range(10):  # count how many MCs have at least one originally faulty input
                            if MC == C:
                                continue
                            current_orig_MC = current.loc[(current['MC'] == MC)][f'guided_sus_pred'].values[0]
                            if current_orig_MC != 0:
                                fails_foreach_mc += 1
                        current_results.update({f'granular_sus': fails_foreach_mc})

                        # save results
                        counted_failures.append(current_results)

        num_failure_types = pd.DataFrame(counted_failures, columns=["method", "K", "net", "C", 'orig_fail_types',
                                                              'granular_sus', 'DeepFault_sus'])
        
        if not path.exists("csv_values/t_tests"):
            mkdir("csv_values/t_tests")
                          
        num_failure_types.to_csv(f"csv_values/t_tests/{dataset}raw_dataframe.csv", index=False)
        return num_failure_types

    def t_test_granularity_vs_by_method_and_K():
        print("\n=======Compare DFGr vs DF with same method and K\n")
        table = [[None for j in range(len(methods))] for i in range(len(SNs))]
        for i, K in enumerate(SNs):
            for j, method in enumerate(methods):
                granular = num_failure_types.loc[(num_failure_types['method'] == method) &
                                                 (num_failure_types['K'] == K)]['granular_sus'].values
                deepfault = num_failure_types.loc[(num_failure_types['method'] == method) &
                                                  (num_failure_types['K'] == K)]['DeepFault_sus'].values
                stat, p = wilcoxon(granular, deepfault)
                print(f"results for K={K} for {method} of DeepFaultGr_sus and DeepFault_sus: stat={stat}, p={p}")
                if p > 0.05:
                    print("not enough evidence of statistically different methods")
                else:
                    print("one method is higher than the other")

                # Python paired sample t-test:
                print(pt.ttest(granular, deepfault, paired=True))
                effsize = pt.compute_effsize(granular, deepfault, paired=True)
                print(effsize)
                # compute_effsize is >0 if the first param is bigger than the second one.
                table[i][j] = (p, (p <= 0.05), round(effsize, 2))  # (enough statistical difference, cohen_d)
        t_test_table(table, SNs, methods, f"{dataset}-DF-vs-DFGr-method-K-comparison")
        print("\n=======Finished that comparison\n")

    def compute_tables():
        # make table like table3 of DeepFault paper,
        def table3_avgs(k, net, method, dataframe, selected_property, use_MC):
            if not use_MC:
                current = dataframe.loc[(dataframe['K'] == k) & (dataframe['net'] == net) & (dataframe['method'] == method)
                                        & (dataframe['MC'] == -1)]
            else:
                current = dataframe.loc[(dataframe['K'] == k) & (dataframe['net'] == net) & (dataframe['method'] == method)
                                        & (dataframe['C'] != dataframe['MC']) & (dataframe['MC'] != -1)]
            # we average all combinations of C and MC for the selected parameters, except for C==MC which is always noise
            loss = round(current[selected_property].map(lambda x: x[0]).mean(), 2)
            acc = round(current[selected_property].map(lambda x: x[1]).mean(), 2)
            return loss, acc

        tables = []
        for use_MC in [True, False]:
            table = [[" ", " ", "T", "O", "D", "R", "T", "O", "D", "R", "T", "O", "D", "R"]]
            for K in SNs:
                row = [f"{K} \n", "Loss\nAccuracy"]
                for net in nets:
                    for method in methods:
                        loss, acc = table3_avgs(K, net, method, df, "sus_scores", use_MC)
                        row.append(f"{loss}\n{acc}")
                table.append(row)
            if use_MC:
                name = f"================ DeepSingleFaults table ================"
            else:
                name = f"================ DeepFault table ================"
            tables.append((table, name))

        # make tex file for viewing
        headers = [["k", "Measure", dataset+"\\_1", dataset+"\\_2", dataset+"\\_3"]]
        make_tex(tables, headers, dataset)

    # ========== CALL FUNCTIONS ===============
    plot_all_heatmaps_without_averaging('DFGr', 'guided_sus_pred', True)
    plot_all_heatmaps_without_averaging('DF', 'sus_pred', False)

    compute_tables()

    num_failure_types = build_failure_types_dataframe()
    t_test_granularity_vs_by_method_and_K()
