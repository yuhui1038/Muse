import librosa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def filter_lang(dataset:list[dict], langs:list[str]) -> list[dict]:
    """Filter dataset, only keep items with lang tag matching specified languages"""
    new_dataset = []
    for ele in tqdm(dataset, desc="Filtering Lang"):
        if 'lang' not in ele or ele['lang'] not in langs :
            continue
        new_dataset.append(ele)
    print(f"filter: {len(dataset)} -> {len(new_dataset)}")
    return new_dataset

def _check_duration(ele, lower_bound, upper_bound):
    """Subprocess task: Check if audio duration is within range"""
    duration = librosa.get_duration(filename=ele['path'])
    if lower_bound != -1 and duration < lower_bound:
        return None
    if upper_bound != -1 and duration > upper_bound:
        return None
    return ele

def filter_length(dataset:list[dict], lower_bound:int=-1, upper_bound:int=-1, max_worker:int=4) -> list[dict]:
    """Filter dataset, only keep items with length in [lower_bound, upper_bound], if set to -1 then no limit on that side"""
    new_dataset = []
    with ProcessPoolExecutor(max_workers=max_worker) as executor:
        futures = [
            executor.submit(_check_duration, ele, lower_bound, upper_bound)
            for ele in dataset
        ]
        with tqdm(total=len(futures), desc="Filtering Length") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    new_dataset.append(result)
                pbar.update(1)
        # for ele in tqdm(dataset, desc="Filtering Length"):
        #     duration = librosa.get_duration(filename=ele['path'])
        #     if lower_bound != -1 and duration < lower_bound:
        #         continue
        #     if upper_bound != -1 and duration > upper_bound:
        #         continue
        #     new_dataset.append(ele)
        print(f"filter: {len(dataset)} -> {len(new_dataset)}")
    return new_dataset