import sys
import argparse
import subprocess
from multiprocessing import Pool
from tqdm import tqdm


def test_once(file_num):
    score_str = subprocess.getoutput(
        f"cd tools && cargo run --release --bin tester poetry run python3 ../main.py < in/{file_num}.txt > out/{file_num}.txt"
    )
    score = int(score_str.split()[-1])
    if score == 0:
        print(score_str)
    return score


def local_test(test_cnt, disable_tqdm):
    file_num_list = [format(i, "0>4") for i in range(test_cnt)]

    with Pool(processes=4) as p:
        scores = list(
            tqdm(p.imap(func=test_once, iterable=file_num_list), total=test_cnt, disable=disable_tqdm)
        )

    print(f"mean score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', default=10)
    parser.add_argument('--disable_tqdm', action="store_true")

    args = parser.parse_args()

    test_cnt = int(args.count)
    disable_tqdm = args.disable_tqdm

    local_test(test_cnt, disable_tqdm)
