from statistics import mean, median


def print_statistics(results):
    for d in results.keys():
        print(f"DECODING STRATEGY {d}")

        print(f"MAX: {max(results[d])} \t SAMPLE: {results[d].index(max(results[d]))}")
        print(f"MIN: {max(results[d])} \t SAMPLE: {results[d].index(min(results[d]))")
        print(f"MEAN: {mean(results[d])}")
        print(f"MEDIAN: {median(results[d])}")