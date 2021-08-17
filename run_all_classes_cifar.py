import os

nets = ["cifar10_test_model_4_128_leaky_relu-normalized",
        "cifar10_test_model_2_256_leaky_relu-normalized",
        "cifar10_test_model_1_512_leaky_relu-normalized"
        ]


def run_all_cifar(seed):
    for k in [10, 20, 30, 40, 50, 70]:
        for approach in ["tarantula", "ochiai", "dstar", "random"]:
            for neural_net in nets:
                for real_class in range(0, 10):
                    for predicted_class in range(9, -2, -1):
                        # predicted class -1 goes last since DeepFault uses the selected images of the other labels
                        # when calling non granular DeepFault, load the imaged used by  granular approach
                        load = (predicted_class == -1)
                        if real_class != predicted_class:
                            os.system(f"python run.py --model {neural_net} --dataset cifar10 \
                                    -C {real_class} -MC {predicted_class} -SN {k} --approach {approach} \
                                      --load_selected_images {load} --seed {seed}")
