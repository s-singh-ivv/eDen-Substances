__version__ = "2.0"
__author__ = "Singh, Satnam"
__maintainer__ = "Singh, Satnam"
__contact__ = "satnam.singh@ivv.fraunhofer.de"

from open_foodtox_dataloader import OpenTox
import torch.optim as optim
import optuna
from tqdm import tqdm
import random
from datetime import datetime
from unet3d import *
from load_data import *
from cube_reader import *
from monai import losses
from sklearn.metrics import (
    jaccard_score as js,
    matthews_corrcoef as mc,
    recall_score as rs,
    confusion_matrix,
    classification_report,
)

from transforms import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


optimize = False


def save_ckp(state, checkpoint_dir, pars):
    """Saves the state of the model
    Params:
    state (dict): the state of the model
    checkpoint_dir (str): path to the directory to save the model
    pars (dict): parameters to be included in the checkpoint naming
    """
    f_path = os.path.join(
        checkpoint_dir,
        "checkpoint_Optuna_OpenTox_"
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


def dice_coef(intersection, union, smooth, eps):
    """Returns dice coefficient between two tensors"""
    return (2 * intersection + smooth) / (union + smooth + eps)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


def objective(trial):
    """
    Objective function for Optuna.
    This objective function is called for hyperparameter tuning chosen from the params dictionary.
    For each run, the validation loss is returned and the parameters corresponding to the least validation loss are the
    optimal parameters of the model
    """
    config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "weight_decay": trial.suggest_float("weight_decay", 0, 1e-3),
        "rate_decay_ep": trial.suggest_int("rate_decay_ep", 10, 50),
        "epochs": trial.suggest_int("epochs", 10, 20),
        "batch_size": trial.suggest_int("batch_size", 10, 20),
        "weight_0": trial.suggest_float("weight_0", 0.5, 2.0),
        "weight_1": trial.suggest_float("weight_1", 0.5, 2.0),
        "feature_size": trial.suggest_int("feature_size", 4, 8, 4),
        "filter_size": trial.suggest_int("filter_size", 4, 8, 4),
        "fin_neurons": trial.suggest_int("fin_neurons", 8, 32, 8),
    }
    device = torch.device("cuda:1")
    loss_function_ = nn.CrossEntropyLoss(
        weight=torch.tensor([config["weight_0"], config["weight_1"]]).to(device)
    )
    return_metric = []
    dice_loss = losses.GeneralizedDiceLoss(sigmoid=True)

    checkpoint_dir = "models/checkpoint_dir"
    EPOCHS = config["epochs"]
    start_epoch = 0
    learning_rate = config["learning_rate"]
    rate_decay_ep = config["rate_decay_ep"]
    weight_decay = config["weight_decay"]
    dataloaders = get_dataloaders(config["batch_size"])
    model = U_net_3d(
        in_channels=1,
        out_channels=3,
        filter_size=config["feature_size"],
        fin_neurons=config["fin_neurons"],
        features=config["feature_size"],
        checkpointing=True,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=rate_decay_ep, gamma=0.1
    )
    for epoch in range(start_epoch, EPOCHS):
        lr_scheduler.step()
        if epoch == EPOCHS - 1:  # save last epoch details
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            pars = {
                "lr": config["learning_rate"],
                "ep": EPOCHS,
                "decay": config["weight_decay"],
                "rate_decay_ep": config["rate_decay_ep"],
                "filter_size": config["filter_size"],
                "fin_neurons": config["fin_neurons"],
                "features": config["feature_size"],
            }
            save_ckp(checkpoint, checkpoint_dir, pars)

        running_loss = 0
        val_running_loss = 0
        dices_tr = [0] * 3
        echa_acc_tr = 0
        echa_acc_val = 0
        dices_val = [0] * 3
        phases = ["train", "val"]
        for phase in phases:
            print(f"Phase: {phase}, Epoch: {epoch + 1} of {EPOCHS}")

            for iter_, data in tqdm(
                enumerate(dataloaders[phase]), total=len(dataloaders[phase])
            ):
                if phase == "train":
                    model.train()
                X_train = torch.as_tensor(data[0], dtype=torch.float, device=device)
                y_train = torch.as_tensor(data[1], dtype=torch.long, device=device)
                y_class = torch.as_tensor(data[2], dtype=torch.long, device=device)
                idx = data[3]

                if phase == "train":
                    eneg_output, label_output = model(X_train)
                    g_loss = dice_loss(eneg_output, y_train)
                    classification_loss = loss_function_(
                        label_output, y_class.squeeze()
                    )
                    optimizer.zero_grad()
                    loss = classification_loss + g_loss
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                if phase == "val":
                    model.eval()
                    with torch.no_grad():
                        eneg_output, label_output = model(X_train)
                        g_loss = dice_loss(eneg_output, y_train)
                        classification_loss = loss_function_(
                            label_output, y_class.squeeze()
                        )
                        optimizer.zero_grad()
                        loss = classification_loss + g_loss
                        val_running_loss += loss.item()

                pred_1 = (torch.sigmoid(eneg_output.to(device)) > 0.5).float().squeeze()
                pred_label = torch.softmax(label_output, dim=1).argmax(dim=1)
                if phase == "train":
                    correct_results_sum = (
                        (pred_label == y_class.squeeze()).sum().float()
                    )
                    echa_acc_tr += correct_results_sum / y_class.shape[0]

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
                    correct_results_sum = (
                        (pred_label == y_class.squeeze()).sum().float()
                    )
                    echa_acc_val += correct_results_sum / y_class.shape[0]

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
                print(f"Acc: {echa_acc_tr / len(dataloaders['train'])}")
                print(f"Epoch: {epoch + 1}/{EPOCHS}, Ep Loss: {epoch_loss}")

            if phase == "val":
                val_epoch_loss = val_running_loss / len(dataloaders["val"])
                avg_dices_v = [a / len(dataloaders["val"]) for a in dices_val]
                print(f"Avg Dices: {avg_dices_v}")
                print(f"Acc: {echa_acc_val / len(dataloaders['val'])}")
                print(f"Epoch: {epoch + 1}/{EPOCHS}, Val Ep Loss: {val_epoch_loss}")

        return_metric.append(val_running_loss / len(dataloaders["val"]))
        return_metric.append(echa_acc_val / len(dataloaders["val"]))

    test_loss = 0
    test_acc = 0
    predicted_list = []
    original_list = []
    for iter_, data in tqdm(
        enumerate(dataloaders["test"]), total=len(dataloaders["test"])
    ):
        X_test = torch.as_tensor(data[0], dtype=torch.float, device=device)
        y_test = torch.as_tensor(data[1], dtype=torch.long, device=device)
        y_class = torch.as_tensor(data[2], dtype=torch.long, device=device)
        idx = data[3]  # for identifying file

        model.eval()
        with torch.no_grad():
            eneg, label_output = model(X_test)
            g_loss = dice_loss(eneg_output, y_test)
            classification_loss = loss_function_(label_output, y_class.squeeze())
            optimizer.zero_grad()
            loss = classification_loss
            test_loss += loss.item()

            # pred_1 = (torch.sigmoid(eneg_output) > 0.5).float().squeeze()
            pred_label = torch.softmax(label_output, dim=1).argmax(dim=1)
            numpy_preds = pred_label.detach().cpu().numpy()
            numpy_gt = y_class.squeeze().detach().cpu().numpy()
            original_list.append(numpy_gt)
            predicted_list.append(numpy_preds)
            correct_results_sum = (pred_label == y_class.squeeze()).sum().float()
            test_acc += correct_results_sum / y_class.shape[0]

    pr = [x[0] for x in predicted_list]
    og = [int(x) for x in original_list]
    test_mcc = mc(og, pr)
    test_recall = rs(og, pr)

    if test_acc / len(dataloaders["test"]) >= 0.65:
        print(classification_report(og, pr))
        print(confusion_matrix(og, pr))
    print(
        f"Test: MCC: {test_mcc}, Acc: {test_acc / len(dataloaders['test'])}, Recall: {test_recall}"
    )

    return return_metric[0]


def get_dataloaders(batch_size=10):
    """Generates and returns the dataloaders"""

    val_data = OpenTox(
        csv_file="opentox_val.csv",
        eden_path="tox_datasets/opentox",
        eneg_path="tox_datasets/opentox",
        mode="val",
    )

    training_data = OpenTox(
        csv_file="opentox_train.csv",
        eden_path="tox_datasets/opentox",
        eneg_path="tox_datasets/opentox",
        mode="train",
    )

    test_data = OpenTox(
        csv_file="opentox_test.csv",
        eden_path="tox_datasets/opentox",
        eneg_path="tox_datasets/opentox",
        mode="test",
    )

    trainloader = DataLoader(
        training_data,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        collate_fn=my_collate,
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=15,
        pin_memory=False,
        collate_fn=my_collate,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        collate_fn=my_collate,
        shuffle=False,
        drop_last=False,
    )

    dataloaders = {"train": trainloader, "val": val_loader, "test": test_loader}
    return dataloaders


def my_collate(batch):
    """Custom collate function for dataloader that pads the data to the largest cube in the batch"""
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
        target_2 = torch.stack([target_ for target_ in target_2])
        target_3 = torch.stack([torch.tensor(target_) for target_ in target_3])

    return [data, target, target_2, target_3]


def test():
    """Test loop for 3D-UNet
    Initializes the saved models for testing purposes, prints and returns test metrics
    """
    config = {
        "batch_size": 12,
        "epochs": 50,
        "learning_rate": 0.00018398555942245975,
        "rate_decay_ep": 20,
        "weight_0": 1.0920170545355055,
        "weight_1": 0.9763172466387251,
        "weight_decay": 0.0008687991618007261,
        "path": "",
    }
    device = torch.device("cuda:2")
    models_ = [
        x for x in os.listdir("models/checkpoint_dir") if "checkpoint_Optuna" in x
    ]

    for m in models_:
        pars = m.split("_2023")[0].split("_")
        features = int(pars[-1])
        fin_neurons = int(pars[-2])
        filter_size = int(pars[-3])
        print(m)
        model = U_net_3d(
            in_channels=1,
            out_channels=3,
            filter_size=filter_size,
            fin_neurons=fin_neurons,
            features=features,
            checkpointing=True,
        ).to(device)

        learning_rate = config["learning_rate"]
        rate_decay_ep = config["rate_decay_ep"]
        weight_decay = config["weight_decay"]
        dataloaders = get_dataloaders(batch_size=1)

        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        try:
            model, optimizer, epoch = load_ckp(
                os.path.join("models/checkpoint_dir", m),
                model,
                optimizer,
            )
            save_features = True
            test_loss = 0
            test_acc = 0
            test_mcc = 0
            test_dices = [0] * 3
            predicted_list = []
            original_list = []
            loss_function_ = nn.CrossEntropyLoss(
                weight=torch.tensor([config["weight_0"], config["weight_1"]]).to(device)
            )
            dice_loss = losses.GeneralizedDiceLoss(sigmoid=True)
            singular_dice = [0] * 3
            for iter_, data in tqdm(
                enumerate(dataloaders["test"]), total=len(dataloaders["test"])
            ):
                X_test = torch.as_tensor(data[0], dtype=torch.float, device=device)
                y_test = torch.as_tensor(data[1], dtype=torch.long, device=device)
                y_class = torch.as_tensor(
                    data[2], dtype=torch.long, device=device
                )  # 0,1
                idx = data[3]  # for identifying file

                model.eval()
                with torch.no_grad():
                    eneg, label_output, enc = model(X_test)
                    g_loss = dice_loss(eneg, y_test)
                    classification_loss = loss_function_(label_output, y_class)
                    optimizer.zero_grad()
                    loss = classification_loss
                    test_loss += loss.item()
                    if not save_features:
                        pred_label = torch.softmax(label_output, dim=1).argmax(dim=1)
                        pred_1 = (torch.sigmoid(eneg.to(device)) > 0.5).float()
                        for channel in range(pred_1.shape[1]):
                            intersection_ch = (
                                pred_1[:, channel, :, :, :]
                                * y_test[:, channel, :, :, :]
                            ).sum()
                            union_ch = (
                                pred_1[:, channel, :, :, :].sum()
                                + y_test[:, channel, :, :, :].sum()
                            )
                            test_dices[channel] += (
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
                            singular_dice[channel] = (
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

                        numpy_preds = pred_label.detach().cpu().numpy()
                        numpy_gt = y_class.squeeze().detach().cpu().numpy()
                        original_list.append(numpy_gt)
                        predicted_list.append(numpy_preds)
                        correct_results_sum = (pred_label == y_class).sum().float()
                        test_acc += correct_results_sum / y_class.shape[0]

            pr = [x[0] for x in predicted_list]
            og = [int(x) for x in original_list]
            test_mcc = mc(og, pr)
            test_recall = rs(og, pr)
            if test_acc / len(dataloaders["test"]) > 0.6:
                print(classification_report(og, pr))
                print(confusion_matrix(og, pr))
                avg_dices = [a / len(dataloaders["test"]) for a in test_dices]
                print(avg_dices)
            print(
                f"Test: MCC: {test_mcc}, Acc: {test_acc / len(dataloaders['test'])}, Recall: {test_recall}"
            )
        except:
            print("Failed model loading due to mismatch:", m)
            continue


if __name__ == "__main__":
    optimize = False
    if optimize:
        study = optuna.create_study(
            storage="sqlite:///opentox_without_na.sqlite3",
            study_name="opentox_without_na",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=100)
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

    test()
