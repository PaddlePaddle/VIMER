# !/usr/bin/env python3
# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

################## LIBRARIES ##############################
import warnings

warnings.filterwarnings("ignore")
import paddle
import numpy as np, os, csv, datetime, faiss
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import numpy as np
from sklearn import metrics
import cv2


"""============================================================================================================="""
################### TensorBoard Settings ###################
def args2exp_name(args):
    exp_name = f"{args.dataset}_{args.loss}_{args.lr}_bs{args.bs}_spc{args.samples_per_class}_embed{args.embed_dim}_arch{args.arch}_decay{args.decay}_fclr{args.fc_lr_mul}_anneal{args.sigmoid_temperature}"
    return exp_name



################# SAVE TRAINING PARAMETERS IN NICE STRING #################
def gimme_save_string(args):
    """
    Taking the set of parameters and convert it to easy-to-read string, which can be stored later.

    Args:
        args: argparse.Namespace, contains all training-specific parameters.
    Returns:
        string, returns string summary of parameters.
    """
    varx = vars(args)
    base_str = ""
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += "\n\t" + str(sub_key) + ": " + str(sub_item)
        else:
            base_str += "\n\t" + str(varx[key])
        base_str += "\n\n"
    return base_str



def eval_metrics_one_dataset(model, test_dataloader, k_vals, args):
    """
    Compute evaluation metrics on test-dataset, e.g. NMI, F1 and Recall @ k.

    Args:
        model:              PyTorch network, network to compute evaluation metrics for.
        test_dataloader:    PyTorch Dataloader, dataloader for test dataset, should have no shuffling and correct processing.
        k_vals:             list of int, Recall values to compute
        args:                argparse.Namespace, contains all training-specific parameters.
    Returns:
        F1 score (float), NMI score (float), recall_at_k (list of float), data embedding (np.ndarray)
    """
    paddle.device.cuda.empty_cache()
    model.eval()
    n_classes = len(test_dataloader.dataset.avail_classes)
    with paddle.no_grad():
        ### For all test images, extract features
        target_labels, feature_coll = [], []
        final_iter = tqdm(test_dataloader, desc="Computing Evaluation Metrics...")
        image_paths = [x[0] for x in test_dataloader.dataset.image_list]
        for idx, inp in enumerate(final_iter):
            input_img, target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img)
            feature_coll.extend(out.cpu().detach().numpy().tolist())
        target_labels = np.hstack(target_labels).reshape(-1, 1)
        feature_coll = np.vstack(feature_coll).astype("float32")

        cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        kmeans = faiss.Clustering(feature_coll.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(feature_coll, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(
            n_classes, feature_coll.shape[-1]
        )

        ### Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        try:
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
        except AttributeError:
            # Only faiss CPU is installed
            print("only use cpu")
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)


        ### Recover max(k_vals) nehbours to use for recall computation
        faiss_search_index = faiss.IndexFlatL2(feature_coll.shape[-1])

        faiss_search_index.add(feature_coll)
        _, k_closest_points = faiss_search_index.search(
            feature_coll, int(np.max(k_vals) + 1)
        )
        k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]
        print("computing recalls")
        ### Compute Recall
        recall_all_k = []
        if args.eval:
            MAP_list = []
            _, k_closest_points = faiss_search_index.search(
                feature_coll, feature_coll.shape[0]
            )
            k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]
            for target, recalled_predictions in zip(target_labels, k_closest_classes):
                pos_nums = 0
                sum_correct = 0
                AP = 0
                all_pos_nums = np.sum(recalled_predictions == target[0])
                for pre_id, pre in enumerate(range(int(all_pos_nums))):
                    if target[0] == recalled_predictions[pre_id]:
                        sum_correct += 1.0
                        AP += sum_correct / (pre_id + 1.0)
                AP = AP / all_pos_nums
                # print("AP",AP)
                MAP_list.append(AP)
            print("map@r", np.mean(MAP_list))

        for k in k_vals:
            recall_at_k = np.sum(
                [
                    1
                    for target, recalled_predictions in zip(
                        target_labels, k_closest_classes
                    )
                    if target in recalled_predictions[:k]
                ]
            ) / len(target_labels)
            recall_all_k.append(recall_at_k)

    return 0, 0, recall_all_k, feature_coll, np.mean(MAP_list)


def eval_metrics_query_and_gallery_dataset(
    model, query_dataloader, gallery_dataloader, k_vals, args
):
    """
    Compute evaluation metrics on test-dataset, e.g. NMI, F1 and Recall @ k.

    Args:
        model:               PyTorch network, network to compute evaluation metrics for.
        query_dataloader:    PyTorch Dataloader, dataloader for query dataset, for which nearest neighbours in the gallery dataset are retrieved.
        gallery_dataloader:  PyTorch Dataloader, dataloader for gallery dataset, provides target samples which are to be retrieved in correspondance to the query dataset.
        k_vals:              list of int, Recall values to compute
        args:                 argparse.Namespace, contains all training-specific parameters.
    Returns:
        F1 score (float), NMI score (float), recall_at_ks (list of float), query data embedding (np.ndarray), gallery data embedding (np.ndarray)
    """
    paddle.device.cuda.empty_cache()
    _ = model.eval()
    n_classes = len(query_dataloader.dataset.avail_classes)

    with paddle.no_grad():
        ### For all query test images, extract features
        query_target_labels, query_feature_coll = [], []
        query_image_paths = [x[0] for x in query_dataloader.dataset.image_list]
        query_iter = tqdm(query_dataloader, desc="Extraction Query Features")
        for idx, inp in enumerate(query_iter):
            input_img, target = inp[-1], inp[0]
            query_target_labels.extend(target.numpy().tolist())
            out = model(input_img)
            query_feature_coll.extend(out.cpu().detach().numpy().tolist())

        ### For all gallery test images, extract features
        gallery_target_labels, gallery_feature_coll = [], []
        gallery_image_paths = [x[0] for x in gallery_dataloader.dataset.image_list]
        gallery_iter = tqdm(gallery_dataloader, desc="Extraction Gallery Features")
        for idx, inp in enumerate(gallery_iter):
            input_img, target = inp[-1], inp[0]
            gallery_target_labels.extend(target.numpy().tolist())
            out = model(input_img)
            gallery_feature_coll.extend(out.cpu().detach().numpy().tolist())

        query_target_labels, query_feature_coll = np.hstack(
            query_target_labels
        ).reshape(-1, 1), np.vstack(query_feature_coll).astype("float32")
        gallery_target_labels, gallery_feature_coll = np.hstack(
            gallery_target_labels
        ).reshape(-1, 1), np.vstack(gallery_feature_coll).astype("float32")

        paddle.device.cuda.empty_cache()

        ### Set CPU Cluster index
        stackset = np.concatenate([query_feature_coll, gallery_feature_coll], axis=0)
        stacklabels = np.concatenate(
            [query_target_labels, gallery_target_labels], axis=0
        )
        cpu_cluster_index = faiss.IndexFlatL2(stackset.shape[-1])
        kmeans = faiss.Clustering(stackset.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(stackset, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(
            n_classes, stackset.shape[-1]
        )

        ### Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        try:
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
        except AttributeError:
            # Only faiss CPU is installed
            print("only use cpu")
            pass
        _, model_generated_cluster_labels = faiss_search_index.search(stackset, 1)

        ### Recover max(k_vals) nearest neighbours to use for recall computation
        faiss_search_index = faiss.IndexFlatL2(gallery_feature_coll.shape[-1])
        faiss_search_index.add(gallery_feature_coll)
        _, k_closest_points = faiss_search_index.search(
            query_feature_coll, gallery_feature_coll.shape[0]
        )
        k_closest_classes = gallery_target_labels.reshape(-1)[k_closest_points]

        ### Compute Recall
        recall_all_k = []
        MAP_list = []
        for target, recalled_predictions in zip(query_target_labels, k_closest_classes):
            pos_nums = 0
            sum_correct = 0
            AP = 0

            all_pos_nums = np.sum(recalled_predictions == target[0])
            for pre_id, pre in enumerate(range(int(all_pos_nums))):
                if target[0] == recalled_predictions[pre_id]:
                    sum_correct += 1.0
                    AP += sum_correct / (pre_id + 1.0)
            AP = AP / all_pos_nums
            MAP_list.append(AP)
        print("map@r", np.mean(MAP_list))
        ### Compute Recall
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum(
                [
                    1
                    for target, recalled_predictions in zip(
                        query_target_labels, k_closest_classes
                    )
                    if target in recalled_predictions[:k]
                ]
            ) / len(query_target_labels)
            recall_all_k.append(recall_at_k)
        recall_str = ", ".join(
            "@{0}: {1:.4f}".format(k, rec) for k, rec in zip(k_vals, recall_all_k)
        )
    return (
        0,
        0,
        recall_all_k,
        query_feature_coll,
        gallery_feature_coll,
        np.mean(MAP_list),
    )


"""============================================================================================================="""
################## WRITE TO CSV FILE #####################
class CSV_Writer:
    """
    Class to append newly compute training metrics to a csv file
    for data logging.
    Is used together with the LOGGER class.
    """

    def __init__(self, save_path, columns):
        """
        Args:
            save_path: str, where to store the csv file
            columns:   list of str, name of csv columns under which the resp. metrics are stored.
        Returns:
            Nothing!
        """
        self.save_path = save_path
        self.columns = columns

        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(self.columns)

    def log(self, inputs):
        """
        log one set of entries to the csv.

        Args:
            inputs: [list of int/str/float], values to append to the csv. Has to be of the same length as self.columns.
        Returns:
            Nothing!
        """
        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)


################## PLOT SUMMARY IMAGE #####################
class InfoPlotter:
    """
    Plotter class to visualize training progression by showing
    different metrics.
    """

    def __init__(self, save_path, title="Training Log", figsize=(20, 15)):
        """
        Args:
            save_path: str, where to store the create plot.
            title:     placeholder title of plot
            figsize:   base size of saved figure
        Returns:
            Nothing!
        """
        self.save_path = save_path
        self.title = title
        self.figsize = figsize
        # Colors for validation lines
        self.v_colors = ["r", "g", "b", "y", "m", "k", "c"]
        # Colors for training lines
        self.t_colors = ["k", "b", "r", "g"]

    def make_plot(
        self,
        t_epochs,
        v_epochs,
        t_metrics,
        v_metrics,
        t_labels,
        v_labels,
        appendix=None,
    ):
        """
        Given a list of iterated epochs, visualize the progression of various training/testing metrics.

        Args:
            t_epochs:  [list of int/float], list of epochs for which training metrics were collected (e.g. Training Loss)
            v_epochs:  [list of int/float], list of epochs for which validation metrics were collected (e.g. Recall @ k)
            t_metrics: [list of float], list of training metrics per epoch
            v_metrics: [list of list of int/float], contains all computed validation metrics
            t_labels, v_labels: [list of str], names for each metric that is plotted.
        Returns:
            Nothing!
        """
        plt.style.use("ggplot")

        f, axes = plt.subplots(1, 2)

        # Visualize Training Loss
        for i in range(len(t_metrics)):
            axes[0].plot(
                t_epochs,
                t_metrics[i],
                "-{}".format(self.t_colors[i]),
                linewidth=1,
                label=t_labels[i],
            )
        axes[0].set_title("Training Performance", fontsize=19)

        axes[0].legend(fontsize=16)

        axes[0].tick_params(axis="both", which="major", labelsize=16)
        axes[0].tick_params(axis="both", which="minor", labelsize=16)

        # Visualize Validation metrics
        for i in range(len(v_metrics)):
            axes[1].plot(
                v_epochs,
                v_metrics[i],
                "-{}".format(self.v_colors[i]),
                linewidth=1,
                label=v_labels[i],
            )
        axes[1].set_title(self.title, fontsize=19)

        axes[1].legend(fontsize=16)

        axes[1].tick_params(axis="both", which="major", labelsize=16)
        axes[1].tick_params(axis="both", which="minor", labelsize=16)

        f.set_size_inches(2 * self.figsize[0], self.figsize[1])

        savepath = self.save_path
        f.savefig(self.save_path, bbox_inches="tight")

        plt.close()


################## GENERATE LOGGING FOLDER/FILES #######################
def set_logging(args):
    """
    Generate the folder in which everything is saved.
    If args.savename is given, folder will take on said name.
    If not, a name based on the start time is provided.
    If the folder already exists, it will by iterated until it can be created without
    deleting existing data.
    The current args.save_path will be extended to account for the new save_folder name.

    Args:
        args: argparse.Namespace, contains all training-specific parameters.
    Returns:
        Nothing!
    """
    checkfolder = args.save_path + "/" + args.savename

    # Create start-time-based name if args.savename is not give.
    if args.savename == "":
        date = datetime.datetime.now()
        time_string = "{}-{}-{}-{}-{}-{}".format(
            date.year, date.month, date.day, date.hour, date.minute, date.second
        )
        checkfolder = (
            args.save_path
            + "/{}_{}_".format(args.dataset.upper(), args.arch.upper())
            + time_string
        )

    # If folder already exists, iterate over it until is doesn't.
    counter = 1
    while os.path.exists(checkfolder):
        checkfolder = args.save_path + "/" + args.savename + "_" + str(counter)
        counter += 1

    # Create Folder
    os.makedirs(checkfolder)
    args.save_path = checkfolder

    # Store training parameters as text and pickle in said folder.
    with open(args.save_path + "/Parameter_Info.txt", "w") as f:
        f.write(gimme_save_string(args))
    pkl.dump(args, open(args.save_path + "/hypa.pkl", "wb"))


import pdb


class LOGGER:
    """
    This class provides a collection of logging properties that are useful for training.
    These include setting the save folder, in which progression of training/testing metrics is visualized,
    csv log-files are stored, sample recoveries are plotted and an internal data saver.
    """

    def __init__(self, args, metrics_to_log, name="Basic", start_new=True):
        """
        Args:
            args:               argparse.Namespace, contains all training-specific parameters.
            metrics_to_log:    dict, dictionary which shows in what structure the data should be saved.
                               is given as the output of aux.metrics_to_examine. Example:
                               {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
                                'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}
            name:              Name of this logger. Will be used to distinguish logged files from other LOGGER instances.
            start_new:         If set to true, a new save folder will be created initially.
        Returns:
            Nothing!
        """
        self.prop = args
        self.metrics_to_log = metrics_to_log

        ### Make Logging Directories
        if start_new:
            set_logging(args)

        ### Set INFO-PLOTS
        if self.prop.dataset != "vehicle_id":
            self.info_plot = InfoPlotter(
                args.save_path + "/InfoPlot_{}.svg".format(name)
            )
        else:
            self.info_plot = {
                "Set {}".format(i): InfoPlotter(
                    args.save_path + "/InfoPlot_{}_Set{}.svg".format(name, i + 1)
                )
                for i in range(3)
            }

        ### Set Progress Saver Dict
        self.progress_saver = self.provide_progress_saver(metrics_to_log)

        ### Set CSV Writters
        self.csv_loggers = {
            mode: CSV_Writer(
                args.save_path + "/log_" + mode + "_" + name + ".csv", lognames
            )
            for mode, lognames in metrics_to_log.items()
        }

    def provide_progress_saver(self, metrics_to_log):
        """
        Provide Progress Saver dictionary.

        Args:
            metrics_to_log: see __init__(). Describes the structure of Progress_Saver.
        """
        Progress_Saver = {
            key: {sub_key: [] for sub_key in metrics_to_log[key]}
            for key in metrics_to_log.keys()
        }
        return Progress_Saver

    def log(self, main_keys, metric_keys, values):
        """
        Actually log new values in csv and Progress Saver dict internally.
        Args:
            main_keys:      Main key in which data will be stored. Normally is either 'train' for training metrics or 'val' for validation metrics.
            metric_keys:    Needs to follow the list length of self.progress_saver[main_key(s)]. List of metric keys that are extended with new values.
            values:         Needs to be a list of the same structure as metric_keys. Actual values that are appended.
        """
        if not isinstance(main_keys, list):
            main_keys = [main_keys]
        if not isinstance(metric_keys, list):
            metric_keys = [metric_keys]
        if not isinstance(values, list):
            values = [values]

        # Log data to progress saver dict.
        for main_key in main_keys:
            for value, metric_key in zip(values, metric_keys):
                self.progress_saver[main_key][metric_key].append(value)

        # Append data to csv.
        self.csv_loggers[main_key].log(values)

    def update_info_plot(self):
        """
        Create a new updated version of training/metric progression plot.

        Args:
            None
        Returns:
            Nothing!
        """
        t_epochs = self.progress_saver["val"]["Epochs"]
        t_loss_list = [self.progress_saver["train"]["Train Loss"]]
        t_legend_handles = ["Train Loss"]

        v_epochs = self.progress_saver["val"]["Epochs"]
        # Because Vehicle-ID normally uses three different test sets, a distinction has to be made.
        if self.prop.dataset != "vehicle_id":
            title = " | ".join(
                key + ": {0:3.3f}".format(np.max(item))
                for key, item in self.progress_saver["val"].items()
                if key not in ["Time", "Epochs"]
            )
            self.info_plot.title = title
            v_metric_list = [
                self.progress_saver["val"][key]
                for key in self.progress_saver["val"].keys()
                if key not in ["Time", "Epochs"]
            ]
            v_legend_handles = [
                key
                for key in self.progress_saver["val"].keys()
                if key not in ["Time", "Epochs"]
            ]

            self.info_plot.make_plot(
                t_epochs,
                v_epochs,
                t_loss_list,
                v_metric_list,
                t_legend_handles,
                v_legend_handles,
            )
        else:
            # Iterate over all test sets.
            for i in range(3):
                title = " | ".join(
                    key + ": {0:3.3f}".format(np.max(item))
                    for key, item in self.progress_saver["val"].items()
                    if key not in ["Time", "Epochs"] and "Set {}".format(i) in key
                )
                self.info_plot["Set {}".format(i)].title = title
                v_metric_list = [
                    self.progress_saver["val"][key]
                    for key in self.progress_saver["val"].keys()
                    if key not in ["Time", "Epochs"] and "Set {}".format(i) in key
                ]
                v_legend_handles = [
                    key
                    for key in self.progress_saver["val"].keys()
                    if key not in ["Time", "Epochs"] and "Set {}".format(i) in key
                ]
                self.info_plot["Set {}".format(i)].make_plot(
                    t_epochs,
                    v_epochs,
                    t_loss_list,
                    v_metric_list,
                    t_legend_handles,
                    v_legend_handles,
                    appendix="set_{}".format(i),
                )


def metrics_to_examine(dataset, k_vals):
    """
    Please only use either of the following keys:
    -> Epochs, Time, Train Loss for training
    -> Epochs, Time, NMI, F1 & Recall @ k for validation

    Args:
        dataset: str, dataset for which a storing structure for LOGGER.progress_saver is to be made.
        k_vals:  list of int, Recall @ k - values.
    Returns:
        metric_dict: Dictionary representing the storing structure for LOGGER.progress_saver. See LOGGER.__init__() for an example.
    """
    metric_dict = {"train": ["Epochs", "Time", "Train Loss"]}

    metric_dict["val"] = ["Epochs", "Time", "NMI", "F1"]
    metric_dict["val"] += ["Recall @ {}".format(k) for k in k_vals]

    return metric_dict

