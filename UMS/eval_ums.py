"""eval_ums 
"""
import argparse
import os
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from paddle.fluid.dataloader.collate import default_collate_fn
from model.model_fusion import networkselect
from model.datasets import Product1MDataset
from model.datasets import give_dataloaders
from model.datasets import CTCDataset
from model.model_fusion_base import VistaBase
from model.model_fusion_base import VistaBaseShare
import eval.auxiliaries as aux
import eval.evaluate as eval
import datetime
import time
from collections import OrderedDict
import random
import pdb


parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval_batch_size", default=512, type=int, help="Total batch size for evaluation."
)
parser.add_argument(
    "--num_workers", default=16, type=int, help="Number of workers in the dataloader."
)
parser.add_argument(
    "--resume_file", default="", type=str, help="Resume from checkpoint"
)
parser.add_argument("--text_file", default="", type=str, help="predict file")
parser.add_argument("--ocr_path", default="", type=str, help="ocr path")
parser.add_argument("--image_root", default="", type=str, help="image root")
parser.add_argument(
    "--image_size", default=224, type=int, help="input image resolution"
)
parser.add_argument("--save_prefix", default="", type=str, help="save prefix")
parser.add_argument("--feature_type", default="both", type=str, help="feature type")
parser.add_argument(
    "--image_model_name",
    default="deit_base_patch16_224",
    type=str,
    help="Backbone for image feature.",
)
parser.add_argument(
    "--image_model_ckpt", default="", type=str, help="Backbone for image feature."
)
parser.add_argument(
    "--text_model_dir",
    default="./bert-base-uncased/",
    type=str,
    help="Backbone for text feature.",
)
parser.add_argument(
    "--max_seq_length",
    default=64,
    type=int,
    help="The maximum total input sequence length",
)
parser.add_argument(
    "--scene_text_model_dir",
    default="./bert-base-uncased/",
    type=str,
    help="Backbone for scene text feature.",
)
parser.add_argument(
    "--max_scene_text_length",
    default=64,
    type=int,
    help="The maximum total input sequence length",
)
parser.add_argument(
    "--projection_dim", default=512, type=int, help="Image and text projection dim."
)
parser.add_argument(
    "--text_embed_dim", default=768, type=int, help="Text embedding dim."
)
parser.add_argument(
    "--image_embed_dim", default=768, type=int, help="Image embedding dim."
)
parser.add_argument(
    "--scene_text_embed_dim", default=768, type=int, help="Scene text embedding dim."
)
parser.add_argument(
    "--pre_training_model",
    default="PretrainModel",
    type=str,
    help="pre traing model name.",
)
parser.add_argument(
    "--is_frozen",
    default=0,
    type=int,
    help="Image and Text pre-trained weights: fixed or not.",
)
parser.add_argument(
    "--return_ocr_pos", default=0, type=int, help="Input 2D pos information of ocr."
)
parser.add_argument(
    "--model_type",
    default="VL",
    type=str,
    help="model type.(VL or DRL)",
)

####### Main Parameter: Dataset to use for Training
parser.add_argument(
    "--dataset",
    default="Stanford_Online_Products",
    type=str,
    help="Dataset to use.",
    choices=[
        "Stanford_Online_Products",
        "inshop_dataset",
        "Product1M",
        "CTC",
    ],
)
### General Training Parameters
parser.add_argument(
    "--n_epochs", default=400, type=int, help="Number of training epochs."
)
parser.add_argument("--bs", default=112, type=int, help="Mini-Batchsize to use.")
parser.add_argument(
    "--samples_per_class",
    default=4,
    type=int,
    help="Number of samples in one class drawn before choosing the next class",
)
parser.add_argument(
    "--seed", default=1, type=int, help="Random seed for reproducibility."
)
parser.add_argument(
    "--infrequent_eval",
    default=0,
    type=int,
    help="only compute evaluation metrics every 10 epochs",
)
##### Evaluation Settings
parser.add_argument(
    "--k_vals", nargs="+", default=[1, 2, 4, 8], type=int, help="Recall @ Values."
)
##### Network parameters
parser.add_argument(
    "--embed_dim", default=512, type=int, help="Embedding dimensionality of the network"
)
parser.add_argument(
    "--fusion_depth", default=2, type=int, help="Number of fusion layers"
)
parser.add_argument(
    "--arch",
    default="vit-deit-base",
    type=str,
    help="Network backend choice: vit-deit-base",
)
parser.add_argument("--pretrained", default=None, type=str, help="Pretrain weights")
parser.add_argument("--infer_model", default=None, type=str, help="Pretrain weights")
parser.add_argument(
    "--eval",
    action="store_true",
    help="If added, the ratio between intra- and interclass distances is stored after each epoch.",
)
##### Setup Parameters
parser.add_argument("--gpu", default="0", type=str, help="GPU-id for GPU to use.")
### Paths to datasets and storage folder
parser.add_argument(
    "--source_path",
    default=".",
    type=str,
    help="Path to data",
)
parser.add_argument(
    "--save_path",
    default=os.getcwd() + "/Eval_Results",
    type=str,
    help="Where to save the checkpoints",
)
parser.add_argument(
    "--savename",
    default="",
    type=str,
    help="Save folder name if any special information is to be included.",
)


def my_collate_fn(batch):
    """my_collate_fn"""
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate_fn(batch)


def main():
    """main"""
    args = parser.parse_args()
    if args.model_type == "VL":
        text_files = args.text_file.split(";")

        test_files = []
        for text_file in text_files:
            test_files.append([text_file, None, None])

        feature_type = args.feature_type
        tokenizer_path = args.text_model_dir
        print("tokenizer_path: ", tokenizer_path)

        for test_file in test_files:
            text_file = test_file[0]
            if args.dataset == "Product1M":
                dataset = Product1MDataset(
                    text_file,
                    args.image_root,
                    tokenizer_path,
                    args.max_seq_length,
                    args.image_size,
                    args.scene_text_model_dir,
                    args.max_scene_text_length,
                    args.ocr_path,
                    args.return_ocr_pos,
                    is_train=False,
                )
            if args.dataset == "CTC":
                dataset = CTCDataset(
                    text_file,
                    args.image_root,
                    tokenizer_path,
                    args.max_seq_length,
                    args.image_size,
                    args.scene_text_model_dir,
                    args.max_scene_text_length,
                    args.ocr_path,
                    args.return_ocr_pos,
                    is_train=False,
                )
        test_dataset = dataset
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.eval_batch_size,
            collate_fn=my_collate_fn,
            num_workers=args.num_workers,
        )
        if args.pre_training_model == "ViSTA_BASE":
            model = VistaBase(args)
        elif args.pre_training_model == "ViSTA_BASE_SHARE":
            model = VistaBaseShare(args)

        if args.resume_file != "" and os.path.exists(args.resume_file):
            state_dict = paddle.load(args.resume_file)
            model.set_state_dict(state_dict)
            print("load model finished!")
        else:
            print("error:", args.resume_file)
            raise RuntimeError('load model error')

        model.eval()

        pairs_infos = []
        image_features = []
        text_features = []
        device = paddle.device.get_device()
        for step, batch in enumerate(test_loader):
            info = batch[-1]
            paddle.device.set_device(device)
            batch = tuple(t for t in batch[:-1])
            if len(batch) == 7:
                (
                    image,
                    input_ids,
                    input_mask,
                    segment_ids,
                    scene_text_ids,
                    scene_text_mask,
                    scene_text_segment_ids,
                ) = batch
            elif len(batch) == 8:
                (
                    image,
                    input_ids,
                    input_mask,
                    segment_ids,
                    scene_text_ids,
                    scene_text_mask,
                    scene_text_segment_ids,
                    ocr_pos,
                ) = batch
            else:
                print("[ERROR] illegal read data!")

            text_embeds, image_embeds, scene_text_embeds = model(
                image=image,
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                scene_text_ids=scene_text_ids,
                scene_text_attention_mask=scene_text_mask,
                scene_text_token_type_ids=scene_text_segment_ids,
                scene_text_pos=ocr_pos,
                is_train=False,
            )
            if args.feature_type == "image" or args.feature_type == "both":
                image_embeds = image_embeds.detach().cpu().numpy()
                image_features.append(image_embeds)

            if args.feature_type == "text" or args.feature_type == "both":
                text_embeds = text_embeds.detach().cpu().numpy()
                text_features.append(text_embeds)

            for data in zip(*info):
                pairs_infos.append(data)

        # save features. for eval
        if args.feature_type == "image" or args.feature_type == "both":
            image_features = np.concatenate(image_features)
            np.save("{}_image_features.npy".format(args.save_prefix), image_features)
            print("image_features: ", image_features.shape)

        if args.feature_type == "text" or args.feature_type == "both":
            text_features = np.concatenate(text_features)
            np.save("{}_text_features.npy".format(args.save_prefix), text_features)
            print("text_features: ", text_features.shape)

        print("pairs_infos:", len(pairs_infos))
        with open(
            "{}_pairs_info.txt".format(args.save_prefix), "w", encoding="utf-8"
        ) as f:
            for idx, info in enumerate(pairs_infos):
                str_info = "\t".join(info) + "\n"
                f.write(str_info)
    if args.model_type == "DRL":

        """============================================================================"""
        args.source_path += "/" + args.dataset
        args.save_path += "/" + args.dataset

        if args.dataset == "inshop_dataset":
            args.k_vals = [1, 10, 20, 30, 40, 50]

        if args.dataset == "Stanford_Online_Products":
            args.k_vals = [1, 10, 100]

        """==========================================================================="""
        random.seed(args.seed)
        np.random.seed(args.seed)

        """============================================================================"""
        ################### GPU SETTINGS ###########################
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        paddle.set_device("gpu")
        model = networkselect(args)

        dataloaders = give_dataloaders(args.dataset, args)

        metrics_to_log = aux.metrics_to_examine(args.dataset, args.k_vals)

        LOG = aux.LOGGER(args, metrics_to_log, name="Base", start_new=True)

        for epoch in range(args.n_epochs):
            if args.eval:
                print("loading infer weight {}".format(args.infer_model))
                if args.infer_model != "" and os.path.exists(args.infer_model):
                    state_dict = paddle.load(args.infer_model)
                else:
                    raise RuntimeError('load model error')

                load_state_dict = {}
                for key, value in model.state_dict().items():
                    key1 = key[6:]
                    if key1 not in state_dict:
                        print("{} is not found in modelpth".format(key1))
                    elif value.shape != state_dict[key1].shape:
                        print(
                            "the shape {} is unmatched: modelpath is {}, model is {}".format(
                                key,
                                state_dict[key1].shape,
                                value.shape,
                            )
                        )
                    else:
                        load_state_dict[key] = state_dict[key1]
                model.set_state_dict(load_state_dict)
            model.eval()

            if args.dataset == "Stanford_Online_Products":
                eval_params = {
                    "dataloader": dataloaders["testing"],
                    "model": model,
                    "args": args,
                    "epoch": epoch,
                }
            elif args.dataset == "inshop_dataset":
                eval_params = {
                    "query_dataloader": dataloaders["testing_query"],
                    "gallery_dataloader": dataloaders["testing_gallery"],
                    "model": model,
                    "args": args,
                    "epoch": epoch,
                }
            else:
                raise Exception("No Dataset >{}< available!".format(args.dataset))

            epoch_freq = args.infrequent_eval

            if (epoch + 1) % epoch_freq == 0:
                results = eval.evaluate(args.dataset, LOG, save=True, **eval_params)


if __name__ == "__main__":
    main()
