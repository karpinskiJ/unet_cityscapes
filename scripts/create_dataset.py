import argparse
import multiprocessing
import glob
import os
import shutil
from PIL import Image
import logging
from cityscapesscripts.preparation import createTrainIdLabelImgs
DATASET_SUBSETS = ["train", "val"]
RESIZE_SHAPE = (1024, 512)


def copy(src_dst: tuple):
    shutil.copy(src_dst[0], src_dst[1])


def copy_file(raw_data_dir_path: str, sub_folder: str, output_root_dir: str, file_name_pattern: str):
    def create_dst_path(src_path: str) -> str:
        return os.path.join(output_root_dir, sub_folder, os.path.basename(src_path))

    raw_data_path_normalized = os.path.normpath(raw_data_dir_path)
    paths = glob.glob(f"{raw_data_path_normalized}/{sub_folder}/*/*{file_name_pattern}")
    src_dest_paths = [(src, create_dst_path(src)) for src in paths]
    with multiprocessing.Pool() as pool:
        pool.map(copy, src_dest_paths)


def resize_image(path: str):
    img = Image.open(path)
    img = img.resize(RESIZE_SHAPE,Image.Resampling.NEAREST)
    img.save(path)


def resize_images(final_dir_path: str):
    images = glob.glob(f"{final_dir_path}/*/*/*.png")
    with multiprocessing.Pool() as pool:
        pool.map(resize_image, images)

def delete_jsons(final_dir_path: str):
    jsons = glob.glob(f"{final_dir_path}/*/*/*.json")
    with multiprocessing.Pool() as pool:
        pool.map(os.remove, jsons)


def main(args):
    logger = logging.getLogger("Create Dataset")
    logger.setLevel(logging.INFO)
    mask_final_path = os.path.join(args.final_dir_path, "mask")
    img_final_path = os.path.join(args.final_dir_path, "img")
    if not os.path.exists(args.final_dir_path):
        os.mkdir(args.final_dir_path)
        os.mkdir(mask_final_path)
        os.mkdir(img_final_path)
        for subset in DATASET_SUBSETS:
            os.mkdir(os.path.join(mask_final_path, subset))
            os.mkdir(os.path.join(img_final_path, subset))
    logger.info("COPYING STARTED")
    for subset in DATASET_SUBSETS:
        copy_file(args.raw_data_dir_path_masks, subset, mask_final_path, "_gtFine_polygons.json")
        createTrainIdLabelImgs.main(os.path.join(mask_final_path,subset))
        copy_file(args.raw_data_dir_path_images, subset, img_final_path, "_leftImg8bit.png")

    delete_jsons(args.final_dir_path)

    logger.info("COPYING COMPLETED")
    if args.resize:
        logger.info("RESIZING STARTED")
        resize_images(args.final_dir_path)
        logger.info("RESIZING COMPLETED")
    shutil.make_archive(args.final_dir_path, "zip", args.final_dir_path)

    logger.info("ARCHIVING COMPLETED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract_JSON",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw_data_dir_path_masks', type=str, required=False,
                        default="C:\\data\\gtFine_trainvaltest\\gtFine")
    parser.add_argument('--final_dir_path', type=str, required=False,
                        default="C:\\projects\\unet_cityscapes\\data")
    parser.add_argument('--raw_data_dir_path_images', type=str, required=False,
                        default="C:\\data\\leftImg8bit_trainvaltest\\leftImg8bit")

    parser.add_argument('--resize', type=bool, required=False,
                        default=True)

    main(parser.parse_args())
