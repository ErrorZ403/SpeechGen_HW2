from statistics import mean, median

def print_statistics(results):
    for d in results.keys():
        print(f"DECODING STRATEGY {d}")

        print(f"MAX: {max(results[d])} \t SAMPLE: {results[d].index(max(results[d]))}")
        print(f"MIN: {max(results[d])} \t SAMPLE: {results[d].index(min(results[d]))}")
        print(f"MEAN: {mean(results[d])}")
        print(f"MEDIAN: {median(results[d])}")

def aggregate_results(results):
    all_results_time = {
    'greedy': [],
    'beam': [],
    'beam_lm': [],
    'beam_lm_rescore': []
    }
    
    all_results_distance = {
        'greedy': [],
        'beam': [],
        'beam_lm': [],
        'beam_lm_rescore': []
    }

    for result_time, result_distance in results:
        for key in all_results_time:
            all_results_time[key].extend(result_time[key])
            all_results_distance[key].extend(result_distance[key])

    return all_results_time, all_results_distance