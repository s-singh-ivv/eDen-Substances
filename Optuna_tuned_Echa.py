__version__ = "2.0"
__author__ = "Singh, Satnam"
__maintainer__ = "Singh, Satnam"
__contact__ = "satnam.singh@ivv.fraunhofer.de"

from custom_loader import CubeECHADataset
import optuna
import torch.optim as optim
import json
from tqdm import tqdm
import random
from datetime import datetime
from unet3d import *
from load_data import *
from cube_reader import *
from monai import losses
from sklearn.metrics import (
    jaccard_score as js,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from transforms import *
import torchmetrics

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
json_file = "params/training_settings.json"
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_json_settings(json_file):
    """Loads and returns the json file with the
    training parameters
    """
    with open(json_file) as json_file:
        params = json.load(json_file)
    return params


params = load_json_settings(json_file)
home_dir = params["params"]["home_dir"]
train_path = params["params"]["train_path"]
test_path = params["params"]["test_path"]
class_count = params["params"]["class_count"]
val_path = params["params"]["val_path"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"

# torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def save_ckp(state, checkpoint_dir, pars):
    """Saves the state of the model
    Params:
    state (dict): the state of the model
    checkpoint_dir (str): path to the directory to save the model
    pars (dict): parameters to be included in the checkpoint naming
    """
    f_path = os.path.join(
        checkpoint_dir,
        "checkpoint_Optune_"
        + str(pars["lr"])
        + "_"
        + str(pars["ep"])
        + "_"
        + str(pars["rate_decay_ep"])
        + "_"
        + str(pars["filter_size"])
        + "_"
        + str(pars["fin_neurons"])
        + "_"
        + str(pars["features"])
        + "_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".pt",
    )

    print(f'Saved at {datetime.now().strftime("%Y%m%d-%H%M%S")}')
    torch.save(state, f_path)


##
def dice_coef(intersection, union, smooth, eps):
    """Returns dice coefficient between two tensors"""
    return (2 * intersection + smooth) / (union + smooth + eps)


def objective(trial):
    """
    Objective function for Optuna.
    This objective function is called for hyperparameter tuning chosen from the params dictionary.
    For each run, the validation loss is returned and the parameters corresponding to the least validation loss are the
    optimal parameters of the model
    """
    device = torch.device("cuda:3")
    dataloaders = get_dataloaders()

    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "weight_decay": trial.suggest_float("weight_decay", 0, 1e-6),
        "rate_decay_ep": trial.suggest_int("epochs", 10, 50),
        "weight_0": trial.suggest_float("weight_0", 0.5, 2.0),
        "weight_1": trial.suggest_float("weight_1", 0.5, 2.0),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "feature_size": trial.suggest_int("feature_size", 4, 32, 4),
        "filter_size": trial.suggest_int("filter_size", 4, 16, 4),
        "fin_neurons": trial.suggest_int("fin_neurons", 8, 64, 8),
        "dice_type": trial.suggest_categorical("dice_type", ["Dice", "Gen_Dice"]),
    }
    model = U_net_3d(
        in_channels=1,
        out_channels=3,
        filter_size=params["filter_size"],
        fin_neurons=params["fin_neurons"],
        features=params["feature_size"],
        checkpointing=True,
    ).to(device)
    metric = torchmetrics.classification.BinaryRecall().to(device)
    loss_function_ = nn.CrossEntropyLoss(
        weight=torch.tensor([params["weight_0"], params["weight_1"]], device=device)
    )
    if params["dice_type"] == "Dice":
        dice_loss = losses.DiceCELoss(sigmoid=True, lambda_ce=0)
    else:
        dice_loss = losses.GeneralizedDiceLoss(sigmoid=True)
    checkpoint_dir = "models/checkpoint_dir"
    EPOCHS = params["epochs"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    start_epoch = 0
    u_net_3dmodel = model
    optimizer = optim.Adam(
        u_net_3dmodel.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params["rate_decay_ep"], gamma=0.1
    )
    for epoch in range(start_epoch, EPOCHS):
        lr_scheduler.step()
        if epoch == EPOCHS - 1:  # save last epoch details
            checkpoint = {
                "epoch": epoch,
                "state_dict": u_net_3dmodel.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            pars = {
                "lr": params["learning_rate"],
                "ep": EPOCHS,
                "decay": params["weight_decay"],
                "rate_decay_ep": params["rate_decay_ep"],
                "filter_size": params["filter_size"],
                "fin_neurons": params["fin_neurons"],
                "features": params["feature_size"],
            }
            save_ckp(checkpoint, checkpoint_dir, pars)
        running_loss = 0
        val_running_loss = 0
        dices_tr = [0] * 3
        echa_acc_tr = 0
        echa_acc_val = 0
        echa_recall_val = 0
        echa_recall_train = 0
        dices_val = [0] * 3
        phases = ["train", "val"]
        for phase in phases:
            print(f"Phase: {phase}, Epoch: {epoch + 1} of {EPOCHS}")
            for iter_, data in tqdm(
                enumerate(dataloaders[phase]), total=len(dataloaders[phase])
            ):
                if phase == "train":
                    u_net_3dmodel.train()
                X_train = torch.as_tensor(data[0], dtype=torch.float, device=device)
                y_train = torch.as_tensor(data[1], dtype=torch.long, device=device)
                y_class = torch.as_tensor(data[2], dtype=torch.long, device=device)
                idx = data[3]  # for identifying file

                if phase == "train":
                    eneg_output, echa_class = u_net_3dmodel(X_train)
                    g_loss = dice_loss(eneg_output, y_train)
                    classification_loss = loss_function_(echa_class, y_class)
                    optimizer.zero_grad()
                    loss = classification_loss + g_loss
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                if phase == "val":
                    u_net_3dmodel.eval()
                    with torch.no_grad():
                        eneg_output, echa_class = u_net_3dmodel(X_train)
                        g_loss = dice_loss(eneg_output, y_train)
                        classification_loss = loss_function_(echa_class, y_class)
                        optimizer.zero_grad()
                        loss = classification_loss + g_loss
                        val_running_loss += loss.item()

                pred_1 = (torch.sigmoid(eneg_output.squeeze()) > 0.5).float().squeeze()
                pred_echa = torch.softmax(echa_class, dim=1).argmax(dim=1)
                if phase == "train":
                    correct_results_sum = (pred_echa == y_class).sum().float()
                    echa_acc_tr += correct_results_sum / y_class.shape[0]
                    echa_recall_train += metric(pred_echa, y_class)
                    for channel in range(pred_1.shape[1]):
                        intersection_ch = (
                            pred_1[:, channel, :, :, :].squeeze()
                            * y_train[:, channel, :, :, :]
                        ).sum()
                        union_ch = (
                            pred_1[:, channel, :, :, :].squeeze().sum()
                            + y_train[:, channel, :, :, :].sum()
                        )
                        dices_tr[channel] += (
                            dice_coef(
                                intersection=intersection_ch,
                                union=union_ch,
                                smooth=1,
                                eps=1e-7,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                if phase == "val":
                    correct_results_sum = (pred_echa == y_class).sum().float()
                    echa_acc_val += correct_results_sum / y_class.shape[0]
                    echa_recall_val += metric(pred_echa, y_class)
                    for channel in range(pred_1.shape[1]):
                        intersection_ch = (
                            pred_1[:, channel, :, :, :].squeeze()
                            * y_train[:, channel, :, :, :]
                        ).sum()
                        union_ch = (
                            pred_1[:, channel, :, :, :].squeeze().sum()
                            + y_train[:, channel, :, :, :].sum()
                        )
                        dices_val[channel] += (
                            dice_coef(
                                intersection=intersection_ch,
                                union=union_ch,
                                smooth=1,
                                eps=1e-7,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

            if phase == "train":
                epoch_loss = running_loss / len(dataloaders["train"])
                avg_dices = [a / len(dataloaders["train"]) for a in dices_tr]
                print(f"Avg Dices: {avg_dices}")
                print(
                    f"Acc: {echa_acc_tr / len(dataloaders['train'])}, Recall: {echa_recall_train / len(dataloaders['train'])}"
                )
                print(f"Epoch: {epoch + 1}/{EPOCHS}, Ep Loss: {epoch_loss}")

            if phase == "val":
                val_epoch_loss = val_running_loss / len(dataloaders["val"])
                avg_dices_v = [a / len(dataloaders["val"]) for a in dices_val]
                print(f"Avg Dices: {avg_dices_v}")
                print(
                    f"Acc: {echa_acc_val / len(dataloaders['val'])}, Recall: {echa_recall_val / len(dataloaders['val'])}"
                )
                print(f"Epoch: {epoch + 1}/{EPOCHS}, Val Ep Loss: {val_epoch_loss}")

            trial.report(echa_acc_val / len(dataloaders["val"]), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    test_acc = test(u_net_3dmodel, dataloaders, device, optimizer)
    print("test acc: ", test_acc)
    return val_running_loss / len(dataloaders["val"])


def test(network, dataloaders, device, optimizer):
    """Test loop for 3D-UNet
    Initializes the saved model using saved parameters for testing purposes, prints and returns test metrics
    """
    u_net_3dmodel = network
    dices_test = [0] * 3
    phases = ["test"]
    preds = []
    trues = []
    val_acc = 0
    test_loss_g = [0] * 3
    for phase in phases:
        for iter_, data in tqdm(
            enumerate(dataloaders[phase]), total=len(dataloaders[phase])
        ):
            u_net_3dmodel.eval()
            X_test = torch.as_tensor(data[0], dtype=torch.float, device=device)
            y_test = torch.as_tensor(data[1], dtype=torch.long, device=device)
            y_class = torch.as_tensor(data[2], dtype=torch.long, device=device)
            idx = data[3]  # for identifying file
            u_net_3dmodel.eval()
            with torch.no_grad():
                u_net_3dmodel.eval()
                eneg_output, echa_class = u_net_3dmodel(X_test)
                optimizer.zero_grad()
                trues.append(y_class.item())

                pred_1 = (torch.sigmoid(eneg_output) > 0.5).float()
                pred_echa = torch.softmax(echa_class, dim=1).argmax(dim=1)
                preds.append(pred_echa.item())
                correct_results_sum = (pred_echa == y_class).sum().float()
                val_acc += correct_results_sum / y_class.shape[0]
                for channel in range(pred_1.shape[1]):
                    intersection_ch = (
                        pred_1[:, channel, :, :, :] * y_test[:, channel, :, :, :]
                    ).sum()
                    union_ch = (
                        pred_1[:, channel, :, :, :].sum()
                        + y_test[:, channel, :, :, :].sum()
                    )
                    dices_test[channel] += (
                        dice_coef(
                            intersection=intersection_ch,
                            union=union_ch,
                            smooth=1,
                            eps=1e-7,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
        print(accuracy_score(trues, preds))
        test_loss_g = [a / len(dataloaders["test"]) for a in test_loss_g]
        print("Loss: ", test_loss_g)
        avg_dices = [a / len(dataloaders["test"]) for a in dices_test]
        print(f"Avg Dices: {avg_dices}")
        print(f"Acc: {val_acc / len(dataloaders['test'])}")


def get_dataloaders():
    """Generates and returns the dataloaders"""

    training_data = CubeECHADataset(
        csv_file="Training_data_ECHA_unsorted.csv",
        eden_path="ECHA_cubes",
        eneg_path="ECHA_cubes",
        mode="train",
        transform=None,
    )
    val_data = CubeECHADataset(
        csv_file="Val_data_ECHA_unsorted.csv",
        eden_path="ECHA_cubes",
        eneg_path="ECHA_cubes",
        transform=None,
        mode="val",
    )

    trainloader = DataLoader(
        training_data,
        batch_size=20,
        num_workers=15,
        pin_memory=False,
        collate_fn=my_collate,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=20,
        num_workers=15,
        pin_memory=False,
        collate_fn=my_collate,
        shuffle=True,
        drop_last=True,
    )
    test_dataset = CubeECHADataset(
        csv_file="Test_data_ECHA_unsorted.csv",
        eden_path="ECHA_cubes",
        eneg_path="ECHA_cubes",
        transform=None,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        collate_fn=my_collate,
        shuffle=True,
        drop_last=False,
    )
    dataloaders = {"train": trainloader, "val": val_loader, "test": test_loader}
    return dataloaders


def my_collate(batch):
    """Custom collate function for dataloader that pads the data to the largest cube in the batch"""
    # this for normal sizes, if items have same size
    sizes = [item[0].shape for item in batch]
    if all(x == sizes[0] for x in sizes):  # seems ok
        data = torch.stack([item[0] for item in batch])
        target = torch.stack([item[1] for item in batch])
        target_2 = torch.stack([torch.tensor(item[2]) for item in batch])
        target_3 = torch.stack([torch.tensor(item[3]) for item in batch])

    else:
        data = [item[0] for item in batch]
        max_dim_1 = max([x[1] for x in sizes])
        max_dim_2 = max([x[2] for x in sizes])
        max_dim_3 = max([x[3] for x in sizes])
        data = torch.stack(
            [pad_input(data_, (1, max_dim_1, max_dim_2, max_dim_3)) for data_ in data]
        )

        target = [item[1] for item in batch]
        target_2 = [item[2] for item in batch]
        target_3 = [item[3] for item in batch]
        target = torch.stack(
            [
                pad_input(target_, (1, max_dim_1, max_dim_2, max_dim_3))
                for target_ in target
            ]
        )
        target_2 = torch.stack(
            [
                pad_input(target_, (1, max_dim_1, max_dim_2, max_dim_3))
                for target_ in target_2
            ]
        )
        target_3 = torch.stack([target_ for target_ in target_3])

    return [data, target, target_2, target_3]


def to_scientific_notation(number):
    a, b = "{:.4E}".format(number).split("E")
    return "{:.5f}E{:+03d}".format(float(a) / 10, int(b) + 1)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


if __name__ == "__main__":
    ## Create an optuna object with the desired optimisation specifications
    study = optuna.create_study(
        storage="sqlite:///echa_optim.sqlite3",
        study_name="echa_optim",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    ## optimisation carried out for 50 trials and best parameters are saved
    study.optimize(objective, n_trials=50)
    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
