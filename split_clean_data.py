#!/usr/bin/env python
"""This is a program to randomly split acoustic measure file into training and test files."""

import csv
import re
import random
import argparse

from typing import List, Dict, Iterator

def clean_files(path: str, row):
    pt_spk = re.compile(r"\D+\d+")
    pt_measure = re.compile(r"\D+")
    if path.startswith("fricative"):
        if re.fullmatch(pt_spk, row["speaker"]):
            row["speaker"] = row["speaker"].lstrip("gd")
        if re.fullmatch(pt_measure, row["duration"]):
            row["duration"] = "missing"
            row["intensity"] = "missing"
            row["cog"] = "missing"
            row["sdev"] = "missing"
            row["skew"] = "missing"
            row["kurt"] = "missing"
    else:
        if re.fullmatch(pt_measure, row["F1_bark"]):
            row["F1_bark"] = "missing"
            row["F2_bark"] = "missing"
            row["F3_bark"] = "missing"
    return row


def read_files(path: str, number: int):
    """This function returns a list of lists of each speaker's dictionaries"""
    speaker_ls = []
    all_ls = []
    count = 0
    with open(path, "r") as source:
        for line in csv.DictReader(source, delimiter="\t"):
            speaker_ls.append(clean_files(path, line))
            count += 1
            if count == number:
                all_ls.append(speaker_ls)
                count = 0
                speaker_ls = []
        return all_ls




def write_files(path: str, splited_data: List):
    keys_dict = []
    all_dict = []
    with open(path, "w") as sink:
        for list in splited_data:
            for dict in list:
                for key in dict:
                    if (key in keys_dict):
                        break
                    else:
                        keys_dict.append(key)
                all_dict.append(dict)    
        writer = csv.DictWriter(sink, fieldnames=keys_dict, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_dict)


def main(args: argparse.Namespace) -> None:
    data_for_split = read_files(args.input, args.num)
    random.shuffle(data_for_split, random.seed(args.seed))
    ten = len(data_for_split) // 10
    write_files(args.train, data_for_split[ten:])
    write_files(args.test, data_for_split[:ten])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Provide the name for the input file")
    parser.add_argument("train", help="Provide the name for the training path")
    parser.add_argument("test", help="Provide the name for the test path")
    parser.add_argument(
        "--seed", type=int, required=True, help="Provide a seed number to PRNG"
    )
    parser.add_argument(
        "--num", type=int, required=True, help="Provide the number of tokens for each speaker"
    )
    main(parser.parse_args())
