import os
import json
import time


FILEDIR = os.path.dirname(os.path.realpath(__file__))
TEMPDIR = os.path.realpath(os.path.join(FILEDIR, "experiments/different_dropout"))


def _next_batch(log_dir):
    exp_numbers = []
    if not os.path.exists(log_dir):
        return 0
    for file in os.listdir(log_dir):
        if "batch" not in file:
            continue
        else:
            exp_numbers.append(int(file.split("_")[1]))
    return max(exp_numbers)+1 if len(exp_numbers) > 0 else 0


if __name__ == '__main__':
    num_batches = 6

    # Batch parameters
    params = {
        "--num_vertices": 100,
        "--num_frames": 128,
        "--num_classes": 6,
        "--num_train": 12000,
        "--num_test": 2400,
        "--sigma": 2,
        "--num_epochs": 10,
        "--learning_rate": 1e-4,
        "--batch_size": 100,
        "--vertex_filter_orders": [3, 3, 3],
        "--time_filter_orders": [3, 3, 3],
        "--num_filters": [8, 16, 32],
        "--time_poolings": [4, 4, 4],
        "--vertex_poolings": [2, 2, 2],
        "--f_h": 50,
        "--f_l": 15,
        "--lambda_h": 80,
        "--lambda_l": 15,
        "--action": "train",
        "--sigma_n": 0.75,
        "--zeros":0,
        "--shot_noise":1.,
        "--model_type":'deep_fir'
    }

   
    #shot_noises = [0.1, 0.3, 0.5, 0.75, 0.9]
    num_train = [600]
    ps = [0.01,0.1,0.5]

    

    for batch in range(num_batches):
        params["--log_dir"] = os.path.join(TEMPDIR, "batch_"+str(_next_batch(TEMPDIR)))
        if not os.path.exists(params["--log_dir"]):
            os.mkdir(params["--log_dir"])

        with open(os.path.join(params["--log_dir"], "global_params.json"), "w") as f:
            json.dump(params, f)

        for p in ps: 

            params["--p"] = p
            #for shot_noise in shot_noises:
            for num_t in num_train: 
                args = []
                params["--num_train"] = num_t

                for arg_name, value in params.items():
                    if isinstance(value, list):
                        args.append(arg_name + " " + " ".join(str(e) for e in value))
                    else:
                        args.append(arg_name + " " + str(value))

                print("****************************************************")
                #print("Simulating %s with shot_noise %.2f" % (model, shot_noise))
                print("Simulating deep_fir %s with num_train %.2f" % (p, num_t))

                #wait for computer to cool down a bit -- should be put in the compare cheb 
                print("30 second pause for cooldown")
                time.sleep(30)
                

                os.system("python3 test_2.py " + " ".join(args))


