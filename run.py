"""
This is the main file that executes the flow of DeepFault
"""
import csv
import os

from test_nn import test_model
from os import path, makedirs
from spectrum_analysis import *
from utils import load_inputs, save_inputs
from utils import create_experiment_dir, get_trainable_layers
from utils import construct_spectrum_matrices
from utils import load_MNIST, load_CIFAR, load_model
from utils import filter_val_set, filter_val_set_by_predicted
from input_synthesis import synthesize
from sklearn.model_selection import train_test_split
import datetime
import argparse
import random

experiment_path = "experiment_results"
model_path = "neural_networks"
group_index = 1
__version__ = "v1.0"


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Spectrum Based Fault Localization for Deep Neural Networks'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version", help="show program version",
                        action="version", version="DeepFault " + __version__)
    parser.add_argument("-M", "--model", help="The model to be loaded. The \
                        specified model will be analyzed.", required=True)
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        or cifar10).", choices=["mnist", "cifar10"])
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to localize dominant neurons")
    parser.add_argument("-D", "--distance", help="the distance between the \
                        original and the mutated image.", type=float)
    parser.add_argument("-C", "--true_class", help="the real label of inputs to \
                        analyze. with -1 all labels are analyzed", type=int)
    parser.add_argument("-MC", "--misclassified_class", help="the wrongly predicted \
                            label of inputs to analyze. with -1 all labels are analyzed", type=int)
    parser.add_argument("-AC", "--activation", help="activation function \
                        or hidden neurons. it can be \"relu\" or \"leaky_relu\"")
    parser.add_argument("-SN", "--suspicious_num", help="number of suspicious \
                        neurons we consider. with -1 all neurons are suspicious", type=int)
    parser.add_argument("-SS", "--step_size", help="multiplication of \
                        gradients by step size", type=float)
    parser.add_argument("-S", "--seed", help="Seed for random processes. \
                        If not provided seed will be selected randomly.", type=int)
    parser.add_argument("-ST", "--star", help="DStar\'s Star \
                        hyperparameter. Has an effect when selected approach is\
                        DStar", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-Load", "--load_selected_images", type=bool, help="if true, tries to load images to perturb"
                                                                           " from previous runs, else selects new ones")
    parser.add_argument("-SSyn", "--save_synthesized", type=bool,
                        help="if true, saves the synthesized images, to be able to compare with the originals"
                             "and see how much they differ")

    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    model_name = args['model']
    dataset = args['dataset'] if not args['dataset'] is None else 'mnist'
    selected_true_class = args['true_class'] if not args['true_class'] is None else 0
    selected_misclassified_class = args['misclassified_class'] if not args['misclassified_class'] is None else -1
    default_step_size = 10 if dataset == 'cifar10' else 1  # the "else" value is for mnist
    default_distance = 0.033 if dataset == 'cifar10' else 0.1
    # for other datasets not mnist or cifar, find empirically best step value and distance
    step_size = args['step_size'] if not args['step_size'] is None else default_step_size
    distance = args['distance'] if not args['distance'] is None else default_distance
    approach = args['approach'] if not args['approach'] is None else 'random'
    susp_num = args['suspicious_num'] if not args['suspicious_num'] is None else 1
    if args['seed'] is None:
        raise NotImplementedError("the seed argument is necesary, use 8 to reproduce the paper's results")
    else:
        seed = args['seed']
    star = args['star'] if not args['star'] is None else 3
    logfile_name = args['logfile'] if not args['logfile'] is None else experiment_path + '/result.log'
    try_load = args['load_selected_images'] if not args['load_selected_images'] is None else False
    save_synthesized_imgs = args['save_synthesized'] if not args['save_synthesized'] is None else False

    ####################
    # 0) Load MNIST or CIFAR10 data
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(one_hot=True)
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=1 / 6.0,
                                                      random_state=seed)

    if not path.exists(experiment_path):
        makedirs(experiment_path)

    logfile = open(logfile_name, 'a')
    logfile.write("\n New run: \n")

    ####################
    # 1) Load the pretrained network.
    try:
        model = load_model(path.join(model_path, model_name))
    except:
        logfile.write("Model not found! Provide a pre-trained model as input. \n")
        exit(1)

    experiment_name = create_experiment_dir(experiment_path, approach, susp_num, dataset)

    logfile.write('--Model: ' + model_name + ', Class: ' + str(selected_true_class) +
                  ', Misclassified as: ' + str(
        selected_misclassified_class) + ', Approach: ' + approach + ', susp_neurons(k):' + str(susp_num) + '\n')

    X_val, Y_val = X_test, Y_test
    # Fault localization is done per class.
    if selected_true_class != -1:
        X_val, Y_val = filter_val_set(selected_true_class, X_val, Y_val)

    mssg = f'there are {len(X_val)} inputs for the true selected class ({selected_true_class})'
    logfile.write(mssg + '\n')
    print(mssg)

    # To analyze single faults, we also separate between wrongly predicted classes
    if selected_misclassified_class != -1:
        X_val, Y_val = filter_val_set_by_predicted(selected_misclassified_class, X_val, Y_val, model)

    mssg = f'of those, {len(X_val)} are classified with the selected label ({selected_misclassified_class})' \
           'or correctly classified'
    logfile.write(mssg + '\n')
    print(mssg)

    ####################
    # 2)test the model and receive the indexes of correct and incorrect classifications
    # Also provide output of each neuron in each layer for test input x.
    filename = experiment_name + '/' + model_name + '_C' + str(selected_true_class) + \
               '_MC' + str(selected_misclassified_class)

    correct_classifications, misclassifications, layer_outs, predictions, score = \
        test_model(model, X_val, Y_val)

    logfile.write('--original [loss, accuracy] -> ' + str(score) + '\n')
    orig_score = score

    ####################
    # 3) Receive the correct classifications  & misclassifications and identify
    # the suspicious neurons per layer

    trainable_layers = get_trainable_layers(model)
    scores, num_cf, num_uf, num_cs, num_us = construct_spectrum_matrices(model,
                                                                         trainable_layers,
                                                                         correct_classifications,
                                                                         misclassifications,
                                                                         layer_outs)

    filename = experiment_name + '/' + model_name + '_C' + str(selected_true_class) + \
               '_MC' + str(selected_misclassified_class) + '_' + approach + '_SN' + str(susp_num)

    print("will use the different approaches for spectrum analysis")
    if approach == 'tarantula':
        suspicious_neuron_idx = tarantula_analysis(trainable_layers, scores,
                                                   num_cf, num_uf, num_cs, num_us,
                                                   susp_num)

    elif approach == 'ochiai':
        suspicious_neuron_idx = ochiai_analysis(trainable_layers, scores,
                                                num_cf, num_uf, num_cs, num_us,
                                                susp_num)

    elif approach == 'dstar':
        suspicious_neuron_idx = dstar_analysis(trainable_layers, scores,
                                               num_cf, num_uf, num_cs, num_us,
                                               susp_num, star)

    elif approach == 'random':
        # Random should truly be random, it makes no sense asking that it doesn't choose the same neurons as other
        # methods.

        filename = experiment_name + '/' + model_name + '_C' + str(selected_true_class) \
                   + '_MC' + str(selected_misclassified_class) + '_random_' + 'SN' + str(susp_num)

        suspicious_neuron_idx = []
        while len(suspicious_neuron_idx) < susp_num:
            l_idx = random.choice(trainable_layers)
            n_idx = random.choice(range(model.layers[l_idx].output_shape[1]))

            if [l_idx, n_idx] not in suspicious_neuron_idx:
                suspicious_neuron_idx.append([l_idx, n_idx])

    else:
        raise ValueError('approach is not one of the valid options')

    logfile.write('--the ' + str(susp_num) + ' Suspicious neurons: ' + str(suspicious_neuron_idx) + '\n')

    # save suspicious neurons
    rows = [str(n) for n in suspicious_neuron_idx]
    f = open(f"{filename}.csv", "w")
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["layer", "neuron_idx", "FL_score", "num_cf", "num_uf", "num_cs", "num_us"])
    for neuron in suspicious_neuron_idx:
        writer.writerow(neuron)

    ####################
    # 4) Run Suspiciousness-Guided Input Synthesis Algorithm
    # Receive the set of suspicious neurons for each layer from Step 3 # and
    # will produce new inputs based on the correct classifications (from the
    # testing set) that exercise the suspicious neurons

    perturbed_xs = []
    perturbed_ys = []

    assert (len(correct_classifications) > 0), "We aren't correctly classifying any input!"

    # save selected inputs to compare classic deepFault with deepSingleFaults perturbing the same images
    if not path.exists(experiment_path + '/selected_inputs'):
        makedirs(experiment_path + '/selected_inputs')

    # for each pair of C and MC, choose the 10 most suspicious inputs to perturb.
    suspicious_inputs_file = experiment_path + '/selected_inputs/' + model_name + '_C' + str(selected_true_class) + \
                             '_MC' + str(selected_misclassified_class) + 'suspicious-selected-inputs.csv'

    if selected_misclassified_class == -1:
        if try_load:
            # if running DeepFault, load the same targeted selected inputs used by DeepSingleFaults
            # i.e. 90 inputs, 10 for each MC
            x_original_suspicious = []
            for mc in range(10):
                if mc == selected_true_class:
                    continue
                x_original_suspicious += load_inputs(f"experiment_results/selected_inputs/{model_name}_C" +
                                                     f"{str(selected_true_class)}_MC{str(mc)}"
                                                     + "suspicious-selected-inputs.csv", dataset,
                                                     selection="targeted", non_granular=False)
            y_original_suspicious = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])] * len(x_original_suspicious)
            for p in y_original_suspicious:  # we know they were of class C
                p[selected_true_class] = 1
        else:
            # this is to use DeepFault alone, loading the images used by granular approach is better for comparison
            # select 10 inputs randomly from the correct classification set.
            selected = np.random.choice(list(correct_classifications), 10)
            x_original_suspicious = list(np.array(X_val)[selected])
            y_original_suspicious = list(np.array(Y_val)[selected])
    else:
        # select 10 inputs from those that are close to being misclassified as MC
        correct_inputs = X_val[correct_classifications]
        temp_predictions = zip(model.predict(correct_inputs), correct_classifications)
        temp_predictions = sorted(temp_predictions, key=lambda pred: pred[0][selected_misclassified_class],
                                  reverse=True)
        suspicious_selected = [elem[1] for elem in temp_predictions[:10]]
        x_original_suspicious = list(np.array(X_val)[suspicious_selected])
        y_original_suspicious = list(np.array(Y_val)[suspicious_selected])

    save_inputs(suspicious_inputs_file, x_original_suspicious, dataset)
    syn_start = datetime.datetime.now()
    x_perturbed_suspicious = synthesize(model, x_original_suspicious, suspicious_neuron_idx, step_size, distance)
    syn_end = datetime.datetime.now()

    if save_synthesized_imgs:
        save_inputs(suspicious_inputs_file.replace("suspicious-selected-inputs", "synthesized-images"),
                    x_perturbed_suspicious, dataset)

    ####################
    # 5) Test if the mutated inputs are adversarial
    score_suspicious = model.evaluate([x_perturbed_suspicious], [y_original_suspicious], verbose=0)
    logfile.write('--Model: ' + model_name + ', Class: ' + str(selected_true_class) + ', Misclassified as: ' +
                  str(selected_misclassified_class) + ', Approach: ' + approach + ', Distance: ' + str(distance) +
                  ', \n mutated suspicious adversarial Score: [loss, accuracy] ' + str(score_suspicious) + '\n')

    print('mutated suspicious adversarial inputs [loss, accuracy] -> ' + str(score_suspicious))
    f.write('original inputs [loss, accuracy] -> ' + str(orig_score) + '\n')
    f.write('mutated suspicious adversarial inputs [loss, accuracy] -> ' + str(score_suspicious) + '\n')
    logfile.write('--Input Synthesis Time: ' + str(syn_end - syn_start) + '\n')

    Y_pred_suspicious = model.predict([x_perturbed_suspicious])
    predictions_suspicious = np.argmax(Y_pred_suspicious, axis=1)
    f.write('the predictions for the suspicious mutated inputs are: ' + str(predictions_suspicious).replace("\n",
                                                                                                            "") + '\n')
    logfile.write('the predictions for the suspicious mutated inputs are: ' +
                  str(predictions_suspicious).replace("\n", "") + '\n')
    if selected_misclassified_class != -1:
        adversarial_count_suspicious = list(predictions_suspicious).count(selected_misclassified_class)
        f.write(f'{adversarial_count_suspicious} times the suspicious mutated adversarial inputs made the model'
                f' think it was class {selected_misclassified_class} \n')
        logfile.write(f'{adversarial_count_suspicious} times the suspicious mutated adversarial inputs made the model'
                      f' think it was class {selected_misclassified_class} \n')

    logfile.close()
    f.close()

    '''
    Currently not available
    ####################
    # 6) retrain the model
    # train_model_fault_localisation(model, x_perturbed, y_perturbed, len(x_perturbed))
    model.fit(x_perturbed, y_perturbed, batch_size=32, epochs=10, verbose=1)

    ####################
    # 7) retest the model
    test_model(model, X_test, Y_test)
    '''
