import os

nets = ["mnist_test_model_5_30_leaky_relu",
        "mnist_test_model_6_25_leaky_relu", "mnist_test_model_8_20_leaky_relu"]


def run_all_mnist(seed):
    for k in [1, 2, 3, 5, 10, 15]:
        for approach in ["tarantula", "ochiai", "dstar", "random"]:
            for neural_net in nets:
                for real_class in range(0, 10):
                    for predicted_class in range(9, -2, -1):
                        # predicted class -1 goes last since DeepFault uses the selected images of the other labels
                        # when calling non granular DeepFault, load the imaged used by  granular approach
                        load = (predicted_class == -1)
                        if real_class != predicted_class:
                            os.system(f"python run.py --model {neural_net} --dataset mnist \
                                    -C {real_class} -MC {predicted_class} -SN {k} --approach {approach} \
                                      --load_selected_images {load} --seed {seed}")
