import json
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# Use HuggingFace mirror site
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Set HuggingFace cache directory so SentenceTransformer can recognize downloaded models
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")


def clean_lyrics(lyrics):
    """
    Clean lyrics by removing segment tags, timestamp tags, and newlines, keeping only pure lyric text
    
    Args:
        lyrics: Raw lyrics text (contains segment tags like [Verse 1], timestamps like [00:07.00], and newlines)
        
    Returns:
        Cleaned lyrics text (plain text, no tags and newlines)
    """
    # Use regex to remove all [tag] format content (including segment tags and timestamps)
    # Pattern matching [any content]
    cleaned = re.sub(r'\[.*?\]', '', lyrics)
    
    # Remove all newlines, replace with spaces
    cleaned = cleaned.replace('\n', ' ')
    
    # Remove extra spaces (replace multiple consecutive spaces with single space)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove leading and trailing spaces
    cleaned = cleaned.strip()
    
    return cleaned


def load_music_data(input_file, max_count=None):
    """
    Load music data from jsonl file
    
    Args:
        input_file: Path to input jsonl file
        max_count: Maximum number to read, None means read all
        
    Returns:
        List of music data
    """
    music_list = []
    print(f"Loading music data: {input_file}")
    if max_count:
        print(f"Limiting to first {max_count} songs")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                data = json.loads(line.strip())
                # Ensure required fields are present
                if 'description' in data and 'lyrics' in data:
                    music_list.append(data)
                    # If reached maximum count, stop reading
                    if max_count and len(music_list) >= max_count:
                        break
            except json.JSONDecodeError:
                continue
    print(f"Successfully loaded {len(music_list)} songs")
    return music_list


def deduplicate_music(music_list, texts, model, threshold=0.90, output_file=None, save_interval=10000, matrix_save_dir=None):
    """
    Deduplicate music data based on text similarity
    
    Args:
        music_list: List of music data
        texts: List of texts for comparison
        model: SentenceTransformer model
        threshold: Similarity threshold
        output_file: Output file path, if provided supports incremental saving
        save_interval: Save every N valid songs processed
        matrix_save_dir: Directory to save matrices
        
    Returns:
        Deduplicated music data list
    """
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Computing similarity matrix...")
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    
    # Save similarity matrix and embeddings
    if matrix_save_dir:
        os.makedirs(matrix_save_dir, exist_ok=True)
        embeddings_path = os.path.join(matrix_save_dir, 'embeddings.pt')
        cos_scores_path = os.path.join(matrix_save_dir, 'cos_scores.pt')
        print(f"Saving embeddings to: {embeddings_path}")
        torch.save(embeddings.cpu(), embeddings_path)
        print(f"Saving similarity matrix to: {cos_scores_path}")
        torch.save(cos_scores.cpu(), cos_scores_path)
        print("Matrix saving complete!")
    
    print(f"Deduplicating (threshold: {threshold})...")
    keep_idx = []
    removed = set()
    
    # If output file provided, open in write mode
    f = None
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        f = open(output_file, 'w', encoding='utf-8')
    
    saved_count = 0
    
    for i in tqdm(range(len(music_list)), desc="Deduplication progress"):
        if i in removed:
            continue
        keep_idx.append(i)
        
        # If incremental saving enabled, save every save_interval songs
        if f and len(keep_idx) - saved_count >= save_interval:
            # Save all valid songs from saved_count to current
            for idx in range(saved_count, len(keep_idx)):
                music = music_list[keep_idx[idx]]
                f.write(json.dumps(music, ensure_ascii=False) + '\n')
            f.flush()  # Ensure write to disk
            saved_count = len(keep_idx)
            print(f"Saved {saved_count} valid songs to file")
        
        for j in range(i+1, len(music_list)):
            if cos_scores[i][j] > threshold:
                removed.add(j)
    
    # Save remaining valid songs
    if f:
        for idx in range(saved_count, len(keep_idx)):
            music = music_list[keep_idx[idx]]
            f.write(json.dumps(music, ensure_ascii=False) + '\n')
        f.close()
        print(f"Saved all {len(keep_idx)} valid songs to file")
    
    deduped_music_list = [music_list[i] for i in keep_idx]
    print(f"Deduplication complete: {len(music_list)} -> {len(deduped_music_list)} (removed {len(removed)} songs)")
    
    return deduped_music_list


def dedup_by_description_and_lyrics(input_file, output_file, threshold=0.95, max_count=None, device='cuda:1', save_interval=10000, matrix_save_dir=None):
    """
    Method 2: Deduplicate based on description + lyrics
    
    Args:
        input_file: Path to input jsonl file
        output_file: Path to output jsonl file
        threshold: Similarity threshold
        max_count: Maximum number to read, None means read all
        device: Device to use, default cuda:1 (GPU1)
        save_interval: Save every N valid songs processed, default 10000
        matrix_save_dir: Directory to save matrices, if provided saves embeddings and similarity matrix
    """
    print("\n========== Method 2: Deduplicate based on description + lyrics ==========")
    print(f"Using device: {device}")
    
    # Load data
    music_list = load_music_data(input_file, max_count=max_count)
    
    # Extract combined text from description + lyrics
    combined_texts = []
    for music in music_list:
        description = music.get('description', '')
        lyrics = music.get('lyrics', '')
        # Clean lyrics, remove structure tags
        cleaned_lyrics = clean_lyrics(lyrics)
        # Concatenate description and cleaned lyrics (separated by delimiter)
        combined_text = f"{description} [SEP] {cleaned_lyrics}"
        combined_texts.append(combined_text)
    
    # Load Chinese model and specify device
    # Check if local model exists, if so use local path directly to avoid re-downloading
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_cache_dir = os.path.join(hf_home, "hub", "models--shibing624--text2vec-bge-large-chinese", "snapshots")
    
    # Find model snapshot directory
    local_model_path = None
    if os.path.exists(model_cache_dir):
        snapshots = [d for d in os.listdir(model_cache_dir) if os.path.isdir(os.path.join(model_cache_dir, d))]
        if snapshots:
            # Use latest snapshot (usually unique)
            local_model_path = os.path.join(model_cache_dir, snapshots[0])
            if os.path.exists(os.path.join(local_model_path, "config.json")):
                print(f"Detected local model, using path: {local_model_path}")
                model = SentenceTransformer(local_model_path, device=device)
            else:
                local_model_path = None
    
    if local_model_path is None:
        print(f"Loading model to {device}...")
        model = SentenceTransformer('shibing624/text2vec-bge-large-chinese', device=device)
    
    # Deduplicate (supports incremental saving)
    deduped_music_list = deduplicate_music(
        music_list, 
        combined_texts, 
        model, 
        threshold, 
        output_file=output_file,
        save_interval=save_interval,
        matrix_save_dir=matrix_save_dir
    )
    
    # Deduplication function already handled saving, just print info here
    if output_file:
        print(f"✓ Save complete! Remaining {len(deduped_music_list)} songs after deduplication\n")
    else:
        # If no output file provided, save once here (compatibility with old code)
        print(f"Saving results to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for music in deduped_music_list:
                f.write(json.dumps(music, ensure_ascii=False) + '\n')
        print(f"✓ Save complete! Remaining {len(deduped_music_list)} songs after deduplication\n")
    
    return deduped_music_list


if __name__ == '__main__':
    # Input file path
    input_file = 'lrc_4w_single_pro_des.jsonl'
    
    # Output file path
    output_file = 'filter_all_4w.jsonl'
    
    # Matrix save directory
    matrix_save_dir = 'generate_lrc'
    
    # Set maximum read count (for testing, None means read all)
    max_count = None  # Test first 5 songs
        
    # Deduplicate based on description + lyrics
    print("\nDeduplicating based on description + lyrics")
    dedup_by_description_and_lyrics(
        input_file, 
        output_file, 
        threshold=0.90, 
        max_count=max_count, 
        device='cuda:7',
        save_interval=10000,  # Save every 10000 valid songs
        matrix_save_dir=matrix_save_dir  # Save similarity matrix
    )
    print(f"\nComplete! Results saved to: {output_file}")
    print(f"Similarity matrix saved to: {matrix_save_dir}")
    # Test lyrics cleaning effect
    # print("\n========== Test Lyrics Cleaning Effect ==========")
    # music_list = load_music_data(input_file, max_count=max_count)
    
    # print("\n" + "="*80)
    # for i, music in enumerate(music_list, 1):
    #     print(f"\n[Song {i}]")
    #     print(f"Description: {music.get('description', '')}")
    #     print("\n--- Original Lyrics ---")
    #     original_lyrics = music.get('lyrics', '')
    #     print(original_lyrics[:500] + "..." if len(original_lyrics) > 500 else original_lyrics)
    #     print("\n--- Cleaned Lyrics ---")
    #     cleaned_lyrics = clean_lyrics(original_lyrics)
    #     print(cleaned_lyrics)
    #     print("\n" + "-"*80)

