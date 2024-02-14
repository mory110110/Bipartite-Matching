import os
import subprocess

# Refer to opts.py for details about the flags
#region graph/dataset flags 
# 9/13/23 3:57 P
# model_type = "inv-ff-hist"
model_type = "ff"

# #9/15/2023 2:02 A
# #region adwords triangular
# problem = "adwords"
# graph_family = "triangular"
# weight_distribution = "triangular"
# weight_distribution_param = "0.1 0.4"  # seperate by a space
# graph_family_parameters = "10"
# capacity_params='0 1' # add this to the flags for adwords only.
# #endregion 

# # 9/18/2023 10:41 A
# #region adwords thick-z
# problem = "adwords"
# graph_family = "thick-z"
# weight_distribution = "thick-z"
# weight_distribution_param = "0.1 0.4"  # seperate by a space
# graph_family_parameters = "10"
# capacity_params='0 1' # add this to the flags for adwords only.
# #endregion 

# 9/18/2023 10:41 A
#region e-obm er
problem = "e-obm"
graph_family = "er"
weight_distribution = "uniform"
weight_distribution_param = "0 1"
graph_family_parameters = "0.05"
#endregion 

# # 9/15/2023 9:02 A
# #region e-obm ba
# problem = "e-obm"
# graph_family = "ba"
# weight_distribution = "uniform"
# weight_distribution_param = "0 1"
# graph_family_parameters = "5"
# #endregion 

# # 9/19/2023 1:14 P
# #region e-sbm gmission
# problem = "e-obm"
# graph_family = "gmission"
# weight_distribution = "gmission"
# weight_distribution_param = "3 10"
# graph_family_parameters = "50"
# #endregion 

# # 9/19/2023 3:42 P
# #region osbm
# problem = "osbm"
# graph_family = "movielense"
# weight_distribution = "movielense"
# weight_distribution_param = "1 2"
# graph_family_parameters = "50"
# #endregion 

u_size = 10
v_size = 30
# 9/14/2023 12:57 P
# dataset_size = 1000
# val_size = 100
# eval_size = 100
# # 9/22/2023 8:34 A
dataset_size = 2000
val_size = 200
eval_size = 100


extention = "/{}_{}_{}_{}_{}by{}".format(
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    u_size,
    v_size,
).replace(" ", "")

train_dataset = "dataset/train" + extention

val_dataset = "dataset/val" + extention

eval_dataset = "dataset/eval" + extention

save_eval_data = True
#endregion

#region model flags

# batch_size = 20
# 9/14/2023 12:57 P
# batch_size = 100
# # 9/22/2023 8:34 A
batch_size = 200
eval_batch_size = 100

embedding_dim = 30  # 60
n_heads = 1  # 3
# # 9/22/2023 8:34 A
# n_epochs = 20
n_epochs = 300
checkpoint_epochs = 0
eval_baselines = "greedy"
if problem == "e-obm":
    eval_baselines += " greedy-rt greedy-t"
if problem == "adwords":
    eval_baselines += " msvv"
lr_model = 0.006
lr_decay = 0.97
beta_decay = 0.8
ent_rate = 0.0006
n_encode_layers = 1

baseline = "exponential"
#endregion
#region directory io flags
# 9/20/2023 1:00 A
output_dir = "saved_models"
# output_dir = "outputs"
log_dir = "logs_dataset"
#endregion
#region model evaluation flags
eval_models = "ff ff-hist inv-ff inv-ff-hist"
# eval_models = "greedy-rt greedy-t"

eval_output = "figures"
# this is a single checkpoint. Example: outputs_dataset/e-obm_20/run_20201226T171156/epoch-4.pt
load_path = None

test_transfer = False


def get_latest_model(
    m_type,
    u_size,
    v_size,
    problem,
    graph_family,
    weight_dist,
    w_dist_param,
    g_fams,
    eval_models,
):
    if m_type not in eval_models:
        return "None"
    m, v = w_dist_param.split(" ")
    models = ""
    if graph_family == "gmission-perm":
        graph_family = "gmission"
    for g_fam_param in g_fams.split(" "):
        # 9/20/2023 1:00 A
        # dir = f"saved_models/output_{problem}_{graph_family}_{u_size}by{v_size}_p={g_fam_param}_{graph_family}_m={m}_v={v}_a=3"
        dir = f"saved_models/{problem}_{graph_family}_{u_size}by{v_size}_p={g_fam_param}"
        list_of_files = sorted(
            os.listdir(dir + f"/{m_type}"), key=lambda s: int(s[4:12] + s[13:])
        )
        if models != "":
            models += " " + dir + f"/{m_type}/{list_of_files[-1]}/best-model.pt"
        else:
            models += dir + f"/{m_type}/{list_of_files[-1]}/best-model.pt"

    return models


arg = [
    u_size,
    v_size,
    problem,
    graph_family,
    weight_distribution,
    weight_distribution_param,
    graph_family_parameters,
    eval_models.split(" "),
]


# attention_models = get_latest_model("attention", *arg)

# ff_supervised_models = get_latest_model("ff-supervised", *arg)

# gnn_hist_models = get_latest_model("gnn-hist", *arg)

# gnn_models = get_latest_model("gnn", *arg)

# gnn_simp_hist_models = get_latest_model("gnn-simp-hist", *arg)

# inv_ff_models = get_latest_model("inv-ff", *arg)

# inv_ff_hist_models = get_latest_model("inv-ff-hist", *arg)

# ff_models = get_latest_model("ff", *arg)

# ff_hist_models = get_latest_model("ff-hist", *arg)

eval_set = graph_family_parameters
# endregion

def make_dir():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(eval_output):
        os.makedirs(eval_output)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/train"):
        os.makedirs("data/train")

    if not os.path.exists("data/val"):
        os.makedirs("data/val")

    if not os.path.exists("data/eval"):
        os.makedirs("data/eval")


def generate_data():
    for n in graph_family_parameters.split(" "):
        # the naming convention here should not be changed!
        train_dir = train_dataset + "/parameter_{}".format(n)
        val_dir = val_dataset + "/parameter_{}".format(n)
        eval_dir = eval_dataset + "/parameter_{}".format(n)

        generate_train = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {} \
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {} \
                            --weight_distribution_param {} --graph_family_parameter {} """.format(
            problem,
            dataset_size,
            train_dir,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        generate_val = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {}  \
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {} \
                            --weight_distribution_param {} --graph_family_parameter {} --seed 20000""".format(
            problem,
            val_size,
            val_dir,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        generate_eval = """python data/generate_data.py --problem {} --dataset_size {} --dataset_folder {} \
                            --u_size {} --v_size {} --graph_family {} --weight_distribution {} \
                            --weight_distribution_param {} --graph_family_parameter {} --seed 40000""".format(
            problem,
            eval_size,
            eval_dir,
            u_size,
            v_size,
            graph_family,
            weight_distribution,
            weight_distribution_param,
            n,
        )

        # print(generate_train)
        # os.system(generate_train)
        subprocess.run(generate_train, shell=True)

        # print(generate_val)
        # os.system(generate_val)
        subprocess.run(generate_val, shell=True)

        # print(generate_eval)
        # os.system(generate_eval)
        subprocess.run(generate_eval, shell=True)


def train_model():
    for n in graph_family_parameters.split(" "):
        # the naming convention here should not be changed!
        train_dir = train_dataset + "/parameter_{}".format(n)
        val_dir = val_dataset + "/parameter_{}".format(n)
        save_dir = output_dir + extention + "/parameter_{}".format(n)
        train = """python run.py --encoder mpnn --model {} --problem {} --batch_size {} --embedding_dim {} --n_heads {} --u_size {}  --v_size {} --n_epochs {} \
                    --train_dataset {} --val_dataset {} --dataset_size {} --val_size {} --checkpoint_epochs {} --baseline {} \
                    --lr_model {} --lr_decay {} --output_dir {} --log_dir {} --n_encode_layers {} --save_dir {} --graph_family_parameter {} --exp_beta {} --ent_rate {}""".format(
            model_type,
            problem,
            batch_size,
            embedding_dim,
            n_heads,
            u_size,
            v_size,
            n_epochs,
            train_dir,
            val_dir,
            dataset_size,
            val_size,
            checkpoint_epochs,
            baseline,
            lr_model,
            lr_decay,
            output_dir,
            log_dir,
            n_encode_layers,
            save_dir,
            n,
            beta_decay,
            ent_rate,
        )

        # print(train)
        subprocess.run(train, shell=True)


def tune_model():
    for n in graph_family_parameters.split(" "):
        # the naming convention here should not be changed!
        train_dir = train_dataset + "/parameter_{}".format(n)
        val_dir = val_dataset + "/parameter_{}".format(n)
        save_dir = output_dir + extention + "/parameter_{}".format(n)
        train = """python run.py --tune_baseline --graph_family {} --encoder mpnn --model {} --problem {} --batch_size {} --embedding_dim {} --n_heads {} --u_size {}  --v_size {} --n_epochs {} \
                    --train_dataset {} --val_dataset {} --dataset_size {} --val_size {} --checkpoint_epochs {} --baseline {} \
                    --lr_model {} --lr_decay {} --output_dir {} --log_dir {} --n_encode_layers {} --save_dir {} --graph_family_parameter {} --exp_beta {} --ent_rate {}""".format(
            graph_family,
            model_type,
            problem,
            batch_size,
            embedding_dim,
            n_heads,
            u_size,
            v_size,
            n_epochs,
            train_dir,
            val_dir,
            dataset_size,
            val_size,
            checkpoint_epochs,
            baseline,
            lr_model,
            lr_decay,
            output_dir,
            log_dir,
            n_encode_layers,
            save_dir,
            n,
            beta_decay,
            ent_rate,
        )

        # print(train)
        subprocess.run(train, shell=True)


def evaluate_model():
    evaluate = """python eval.py --problem {} --graph_family {} --embedding_dim {} --load_path {} --ff_models {} --attention_models {} --inv_ff_models {} --ff_hist_models {} \
        --inv_ff_hist_models {} --gnn_hist_models {} --gnn_models {} --gnn_simp_hist_models {} --ff_supervised_models {} --eval_baselines {} \
        --baseline {} --eval_models {} --eval_dataset {}  --u_size {} --v_size {} --eval_set {} --eval_size {} --eval_batch_size {} \
        --n_encode_layers {} --n_heads {} --output_dir {} --dataset_size {} --batch_size {} --encoder mpnn --weight_distribution {} --weight_distribution_param {}""".format(
        problem,
        graph_family,
        embedding_dim,
        load_path,
        ff_models,
        attention_models,
        inv_ff_models,
        ff_hist_models,
        inv_ff_hist_models,
        gnn_hist_models,
        gnn_models,
        gnn_simp_hist_models,
        ff_supervised_models,
        eval_baselines,
        baseline,
        eval_models,
        eval_dataset,
        u_size,
        v_size,
        eval_set,
        eval_size,
        eval_batch_size,
        n_encode_layers,
        n_heads,
        output_dir,
        eval_size,
        eval_batch_size,
        weight_distribution,
        weight_distribution_param,
    )
    if test_transfer:
        evaluate += " --test_transfer"
    elif save_eval_data:
        evaluate += " --save_eval_data"
    # print(evaluate)
    subprocess.run(evaluate, shell=True)


if __name__ == "__main__":
    # make the directories if they do not exist
    # make_dir()
    # generate_data()
    train_model()
    # tune_model()
    # evaluate_model()
