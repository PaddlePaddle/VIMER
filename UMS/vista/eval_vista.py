"""eval_vista
"""
import argparse
import os
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from paddle.fluid.dataloader.collate import default_collate_fn
from model.model_fusion import ViSTA
from model.datasets import LocalSceneTextDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "--eval_batch_size",
    default=512,
    type=int,
    help="Total batch size for training.",
)

parser.add_argument(
    "--num_workers",
    default=16,
    type=int,
    help="Number of workers in the dataloader.",
)

parser.add_argument(
    "--resume_file", default="", type=str, help="Resume from checkpoint"
)

parser.add_argument(
    "--text_file", default="./cupati_top200.txt", type=str, help="predict file"
)

parser.add_argument(
    "--image_root", default="./2021_Q1_product_300/", type=str, help="image root"
)

parser.add_argument(
    "--ocr_feature_files", default="./test.npy", type=str, help="ocr features file"
)

parser.add_argument(
    "--return_ocr_feature", default=224, type=int, help="input image resolution"
)

parser.add_argument(
    "--image_size", default=224, type=int, help="input image resolution"
)

parser.add_argument("--ocr_pos_path", default="./data", type=str, help="ocr pos path")

parser.add_argument(
    "--return_ocr_2D_pos", default=0, type=int, help="return ocr 2D pos"
)

parser.add_argument(
    "--save_prefix", default="./2021_Q1_product_300/", type=str, help="save prefix"
)

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
    default="./bert-base-chinsese/",
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
    "--max_scene_text_length",
    default=64,
    type=int,
    help="The maximum total input scene_text sequence length",
)

parser.add_argument(
    "--scene_text_model_dir",
    default="./bert-base-chinese/",
    type=str,
    help="Backbone for scene text feature.",
)

parser.add_argument(
    "--projection_dim", default=512, type=int, help="Image and text projection dim."
)

parser.add_argument(
    "--text_embed_dim", default=768, type=int, help="Text embedding dim."
)

parser.add_argument(
    "--scene_text_embed_dim", default=768, type=int, help="Scene Text embedding dim."
)

parser.add_argument(
    "--image_embed_dim", default=512, type=int, help="Image embedding dim."
)

parser.add_argument(
    "--pre_training_model",
    default="PretrainModel",
    type=str,
    help="pre traing model name.",
)

parser.add_argument("--fusion_depth", default=4, type=int, help="fusion depth.")

parser.add_argument(
    "--is_frozen",
    default=0,
    type=int,
    help="Image and Text pre-trained weights: fixed or not.",
)


def my_collate_fn(batch):
    """my_collate_fn"""
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate_fn(batch)


def main():
    """main"""
    args = parser.parse_args()
    return_ocr_feature = args.return_ocr_feature
    text_files = args.text_file.split(";")

    test_files = []
    if return_ocr_feature == 3:
        ocr_pos_paths = args.ocr_pos_path.split(";")
        for text_file, ocr_pos_path in zip(text_files, ocr_pos_paths):
            test_files.append([text_file, None, ocr_pos_path])
    elif return_ocr_feature == 2:
        ocr_feature_files = args.ocr_feature_files.split(";")
        for text_file, ocr_feature_file in zip(text_files, ocr_feature_files):
            test_files.append([text_file, ocr_feature_file, None])
    else:
        for text_file in text_files:
            test_files.append([text_file, None, None])

    feature_type = args.feature_type
    tokenizer_path = args.text_model_dir
    print("tokenizer_path: ", tokenizer_path)
    scene_text_tokenizer_path = args.scene_text_model_dir
    print("scene text tokenizer_path: ", scene_text_tokenizer_path)

    for test_file in test_files:
        text_file = test_file[0]
        ocr_features_path = test_file[1]
        ocr_pos_path = test_file[2]

        dataset = LocalSceneTextDataset(
            text_file,
            args.image_root,
            tokenizer_path,
            scene_text_tokenizer_path,
            args.max_seq_length,
            args.max_scene_text_length,
            args.image_size,
            ocr_features_path=ocr_features_path,
            return_ocr_feature=return_ocr_feature,
            pos_path=ocr_pos_path,
            return_2D_pos=args.return_ocr_2D_pos,
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
    if args.pre_training_model == "ViSTA":
        model = ViSTA(args)

    if args.resume_file != "" and os.path.exists(args.resume_file):
        state_dict = paddle.load(args.resume_file)
        model.set_state_dict(state_dict)
        print("load model finished!")
    else:
        print("error:", args.resume_file)

    model.eval()

    pairs_infos = []
    image_features = []
    text_features = []
    scene_text_features = []
    device = paddle.device.get_device()
    for step, batch in enumerate(test_loader):
        info = batch[-1]
        paddle.device.set_device(device)
        # batch = tuple(t.to(device=device, non_blocking=True) for t in batch[:-1])
        batch = tuple(t for t in batch[:-1])
        ocr_feature = None
        ocr_mask = None
        scene_text_ids = None
        scene_text_mask = None
        scene_text_segment_ids = None
        fusion_mask = None
        ocr_2D_pos = None

        if len(batch) == 4:
            image, input_ids, input_mask, segment_ids = batch
        elif len(batch) == 6:
            image, input_ids, input_mask, segment_ids, ocr_feature, ocr_mask = batch
        elif len(batch) == 8:
            (
                image,
                input_ids,
                input_mask,
                segment_ids,
                scene_text_ids,
                scene_text_mask,
                scene_text_segment_ids,
                fusion_mask,
            ) = batch
        elif len(batch) == 9:
            (
                image,
                input_ids,
                input_mask,
                segment_ids,
                scene_text_ids,
                scene_text_mask,
                scene_text_segment_ids,
                fusion_mask,
                ocr_2D_pos,
            ) = batch
        else:
            print("[ERROR] illegal read data!")
        text_embeds, image_embeds, scene_text_embeds = model(
            image=image,
            input_ids=input_ids,
            scene_text_ids=scene_text_ids,
            attention_mask=input_mask,
            scene_text_attention_mask=scene_text_mask,
            token_type_ids=segment_ids,
            scene_text_token_type_ids=scene_text_segment_ids,
            ocr_feature=ocr_feature,
            ocr_mask=ocr_mask,
            scene_text_pos=ocr_2D_pos,
            fusion_mask=fusion_mask,
            is_train=False,
        )
        if args.feature_type == "image" or args.feature_type == "both":
            image_embeds = image_embeds.detach().cpu().numpy()
            image_features.append(image_embeds)

        if args.feature_type == "text" or args.feature_type == "both":
            text_embeds = text_embeds.detach().cpu().numpy()
            text_features.append(text_embeds)

        if args.feature_type == "scene_text" or args.feature_type == "both":
            scene_text_embeds = scene_text_embeds.detach().cpu().numpy()
            scene_text_features.append(scene_text_embeds)

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

    if args.feature_type == "scene_text" or args.feature_type == "both":
        scene_text_features = np.concatenate(scene_text_features)
        np.save(
            "{}_scene_text_features.npy".format(args.save_prefix), scene_text_features
        )
        print("scene_text_features: ", scene_text_features.shape)

    if args.feature_type == "scene_text" or args.feature_type == "both":
        tmp = (image_features + scene_text_features) / 2
        np.save("{}_fusion_features.npy".format(args.save_prefix), tmp)

    print("pairs_infos:", len(pairs_infos))
    with open("{}_pairs_info.txt".format(args.save_prefix), "w", encoding="utf-8") as f:
        for idx, info in enumerate(pairs_infos):
            str_info = "\t".join(info) + "\n"
            f.write(str_info)


if __name__ == "__main__":
    main()
