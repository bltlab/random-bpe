import sys
import requests
import zipfile
import os
import argparse

from pathlib import Path

# Based on download_data.py from
# https://github.com/turkic-interlingua/til-mt/blob/master/til_corpus/download_data.py

base_url = "gs://til-corpus/corpus"

# all the langs present in the corpus
langs = [
    "alt",
    "az",
    "ba",
    "cjs",
    "crh",
    "cv",
    "gag",
    "kaa",
    "kjh",
    "kk",
    "krc",
    "kum",
    "ky",
    "sah",
    "slr",
    "tk",
    "tr",
    "tt",
    "tyv",
    "ug",
    "uum",
    "uz",
    "en",
    "ru",
]

# downloads a zip file given a url
def download_url(url, save_path):
    try:
        os.system(f"gsutil -m cp -r {url} {save_path}")
    except CommandException as e:
        raise IOError("Requested test files are not found!")


def unzip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def create_folder(base_path, source, target, split):

    folder_path = Path(f"{base_path}/{source}-{target}/{split}")

    if not folder_path.exists():
        folder_path.mkdir(exist_ok=True, parents=True)

        return True
    else:
        print(f"{folder_path} already exists! Skipping...")

        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_language",
        default=None,
        type=str,
        required=True,
        help="2-letter code of the source language",
    )
    parser.add_argument(
        "--target_language",
        default=None,
        type=str,
        required=True,
        help="2-letter code of the target language",
    )
    parser.add_argument(
        "--split",
        default="all",
        type=str,
        required=False,
        help='which split to download. Options: \{"train", "dev", "test"\}',
    )
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="Where to download files.",
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    source = args.source_language
    target = args.target_language
    download_folder = Path(args.download_folder).expanduser().resolve()
    corpus_name = args.corpus_name

    # check if the language code exists in the corpus

    if source not in langs or target not in langs:
        print("Language code error!!!\nLanguage code must be one of these: ")
        print(langs)
        quit()

    if args.split not in ["train", "dev", "test", "all"]:
        print("Splits should be one of the following: [train, dev, test, all]")
        quit()

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    # base save path
    save_path = download_folder / f"{source}-{target}" / corpus_name / "download"

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    save_path_str = str(save_path)

    for split in splits:
        # create_folder(download_folder, source, target, split)

        download_path = f"{base_url}/{split}/{source}-{target}"
        split_folder = Path(f"{save_path_str}/{split}/")

        if not split_folder.exists():
            split_folder.mkdir(exist_ok=True, parents=True)
        print(f"Downloading {split} files from {download_path} to {split_folder}")
        download_url(download_path, str(split_folder))
