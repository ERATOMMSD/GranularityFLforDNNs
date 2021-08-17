import sys

from matplotlib import pyplot as plt

from utils import load_inputs


# an example csv_name "cifar10_test_model_2_256_leaky_relu-normalized_C7_MC1suspicious-selected-inputs.csv"
def view(csv_name, imgs_ammount, dataset, C, MC):
    def read_numpy_encoding(images, C, MC, dataset):
        fig_s = plt.figure(figsize=(10, 5))
        fig_s.suptitle(f"selected images to perturb for class {C} and prediction {MC}")
        for i in range(10):
            fig_s.add_subplot(2, 5, i+1)
            if dataset == 'mnist':
                plt.imshow(images[i][0], interpolation='nearest')
            else:
                plt.imshow(images[i], interpolation='nearest')
        return

    def load_imags(file, dataset, imgs_num, filepath):
        filename = filepath + file
        return load_inputs(filename, dataset, imgs_num=imgs_num)

    filepath = f"experiment_results/selected_inputs/"
    images = load_imags(csv_name, dataset, imgs_ammount, filepath)
    read_numpy_encoding(images, C, MC, dataset)

    plt.show()


if __name__ == "__main__":
    view(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
