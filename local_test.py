import argparse
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import time


def test_once(params):
    file_num = params[0]
    no_hints = params[1]

    start = time.time()

    if no_hints:
        command = f"cd tools && cargo run --release --bin tester poetry run python3 ../main_nohints.py < in/{file_num}.txt > out/{file_num}.txt"
    else:
        command = f"cd tools && cargo run --release --bin tester poetry run python3 ../main.py < in/{file_num}.txt > out/{file_num}.txt"

    score_str = subprocess.getoutput(command)

    seconds = time.time() - start

    score = int(score_str.split()[-1])
    if score == 0:
        print(score_str)
    return file_num, score, seconds


def local_test(test_cnt, disable_tqdm, no_hints):
    params_list = [(format(i, "0>4"), no_hints) for i in range(test_cnt)]

    with Pool(processes=4) as p:
        results = list(
            tqdm(
                p.imap(func=test_once, iterable=params_list),
                total=test_cnt,
                disable=disable_tqdm,
            )
        )

    file_nums = []
    scores = []
    secondss = []

    for file_num, score, seconds in results:
        print(file_num, score)
        file_nums.append(file_num)
        scores.append(score)
        secondss.append(seconds)

    print(
        f"mean score: {sum(scores) / len(scores)}, mean seconds: {sum(secondss) / len(secondss)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", default=10)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--no_hints", action="store_true")

    args = parser.parse_args()

    test_cnt = int(args.count)
    disable_tqdm = args.disable_tqdm
    no_hints = args.no_hints

    local_test(test_cnt, disable_tqdm, no_hints)
