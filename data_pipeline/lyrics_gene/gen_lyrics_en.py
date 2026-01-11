import os
import json
import time
import random
import re
from openai import OpenAI
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables
# Note: Set these environment variables before running the script
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your custom API URL
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ""  # Replace with your API key or set via environment variable
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"  # Replace with API URL or set via environment variable

# Initialize client
client = OpenAI()


def _extract_lyrics_timestamps(lyrics_text):
    """
    Extract timestamps from lyrics and convert to seconds
    Args:
        lyrics_text: Lyrics string
    Returns:
        List[float]: Timestamps in order (seconds)
    """
    if not isinstance(lyrics_text, str):
        return []
    pattern = re.compile(r'\[(\d{2}):(\d{2})(?:\.(\d{2}))?\]')
    timestamps = []
    for match in pattern.finditer(lyrics_text):
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction = match.group(3)
        total_seconds = minutes * 60 + seconds
        if fraction is not None:
            divisor = 100 if len(fraction) == 2 else 10 ** len(fraction)
            total_seconds += int(fraction) / divisor
        timestamps.append(total_seconds)
    return timestamps


def _validate_timestamps(lyrics_text, min_last_timestamp=170, max_interval=35):
    """
    Validate if timestamps in lyrics meet requirements
    Args:
        lyrics_text: Lyrics string
        min_last_timestamp: Minimum value of last timestamp (seconds)
        max_interval: Maximum interval between last two timestamps (seconds)
    Returns:
        bool: Whether validation passed
    """
    timestamps = _extract_lyrics_timestamps(lyrics_text)
    if len(timestamps) < 2:
        print("Validation failed: Timestamp count less than 2")
        return False
    last = timestamps[-1]
    second_last = timestamps[-2]
    if last < min_last_timestamp:
        print(f"Validation failed: Last timestamp {last:.2f}s is less than {min_last_timestamp}s")
        return False
    if last - second_last > max_interval:
        print(f"Validation failed: Interval between last two timestamps {last - second_last:.2f}s is greater than {max_interval}s")
        return False
    return True


def chat_gpt(text, model='gpt-4o-mini'):
    while True:
        try:
            # Call OpenAI chat completions API
            completion = client.chat.completions.create(
                model=model,  # Use GPT-4o-mini model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ]
            )
            # Get response content
            if getattr(completion.choices[0].message, 'content', None):
                content = completion.choices[0].message.content.strip()
                return content
            else:
                print('error_wait_2s')
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)


def chat_gpt_call(text, model='gpt-4o-mini'):
    # Call OpenAI chat completions API
    completion = client.chat.completions.create(
        model=model,  # Use GPT-4o-mini model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    )
    # Get response content
    if getattr(completion.choices[0].message, 'content', None):
        content = completion.choices[0].message.content.strip()
        return content
    else:
        print('error_wait_2s')


def generate_music_descriptions(all_music_data, index_pool, output_file, file_lock, sample_size=20, model='gpt-4o-mini', max_retries=0):
    """
    Read music data file, randomly sample, call GPT to generate new music descriptions and lyrics
    
    Args:
        all_music_data: List of all music data
        index_pool: Index pool object (thread-safe)
        output_file: Path to output jsonl file
        file_lock: File write lock
        sample_size: Number of samples to randomly extract
        model: Model name to use
        max_retries: Maximum retry count
    
    Returns:
        (used_indices, success_count): List of used indices and number of successful generations
    """
    # duration_ranges = [
    #     ("3.00", "3.15", 50), ("3.15", "3.30", 50), ("3.30", "3.45", 60),
    #  ("3.45", "4.00", 60),
    #     ("4.00", "4.15", 70), ("4.15", "4.30", 70), ("4.30", "4.45", 70)
    # ]
    duration_ranges = [
        ("3.00", "3.15", 60), ("3.15", "3.30", 70), ("3.30", "3.45", 80),("3.45", "4.00", 90),
        ("4.00", "4.15", 100), ("4.15", "4.30", 100), ("4.30", "4.45", 100)
    ]
    selected_range = random.choice(duration_ranges)
    require_length = selected_range[2]
    
    # Directly convert to timestamp format (strictly corresponding to left and right ends of tuple)
    start_timestamp = f"[{selected_range[0].replace('.', ':')}.00]"
    end_timestamp = f"[{selected_range[1].replace('.', ':')}.00]"
    
    # Generate duration description
    # Convert seconds to minutes and seconds format
    start_seconds = float(selected_range[0])
    start_minutes = int(start_seconds // 60)
    start_secs = int(start_seconds % 60)
    start_duration = f"{start_minutes}min {start_secs}sec"
    
    end_seconds = float(selected_range[1])
    end_minutes = int(end_seconds // 60)
    end_secs = int(end_seconds % 60)
    end_duration = f"{end_minutes}min {end_secs}sec"
    
    # Generate random timestamps in examples (randomly generated within time range)
    # Parse time string to minutes and seconds
    start_parts = selected_range[0].split('.')
    end_parts = selected_range[1].split('.')
    
    start_minutes = int(start_parts[0])
    start_seconds = int(start_parts[1])
    end_minutes = int(end_parts[0])
    end_seconds = int(end_parts[1])
    
    # Convert to total seconds
    start_total_seconds = start_minutes * 60 + start_seconds
    end_total_seconds = end_minutes * 60 + end_seconds
    
    # Randomly generate within range
    example1_seconds = random.randint(start_total_seconds, end_total_seconds)
    example2_seconds = random.randint(start_total_seconds, end_total_seconds)
    
    example1_minutes = example1_seconds // 60
    example1_secs = example1_seconds % 60
    example2_minutes = example2_seconds // 60
    example2_secs = example2_seconds % 60
    
    example1_timestamp = f"[{example1_minutes:02d}:{example1_secs:02d}.00]"
    example2_timestamp = f"[{example2_minutes:02d}:{example2_secs:02d}.00]"
    
    # Get sample indices from index pool (thread-safe)
    selected_indices = index_pool.get_indices(sample_size)
    if not selected_indices:
        return [], 0
    
    sample_data = [all_music_data[i] for i in selected_indices]
    
    # Extract all unique styles
    styles = []
    for data in sample_data:
        style = data.get('style', '')
        if style and style not in styles:
            styles.append(style)
    
    styles_text = "、".join(styles)
    
    # Build example text - include all sampled data (excluding style)
    examples = []
    for i, data in enumerate(sample_data, 1):  
        lyrics_text = " ".join(data.get('lyrics', [])) if isinstance(data.get('lyrics'), list) else data.get('lyrics', '')
        description = data.get('description', '')
        examples.append(f"Example {i}:\ndescription: {description}\nlyrics: {lyrics_text}")
    
    examples_text = "\n\n".join(examples)

    prompt = f"""Generate 2 complete songs. Each song must meet the following hard requirements:
- Strictly forbidden to generate lyrics with fewer than {require_length} lines!
- The number of lyric lines for each song must be strictly greater than {require_length}. This is a hard requirement!
- The timestamp of the final line must be between {start_timestamp} and {end_timestamp}.
- The two songs must differ in duration and line count; their final timestamps must not be identical.
- The timestamp interval between adjacent lyric lines must not exceed 10 seconds! Timestamps must be continuous and progress naturally.
- Awkward gaps like "[03:25.00]in the heart[04:25.00]the last lyric" are strictly forbidden. Do not exceed a 10-second interval.
- It is strictly forbidden to repeat the entire structure or its sections after one iteration is complete. It is also strictly forbidden to repeat the same lyric line multiple times.
If any of the above requirements are not met, the generation is considered a failure. Please regenerate.
Please generate 2 new, diverse music descriptions and LRC format lyrics. The language should be English.


Creative Requirements:
1. Style and Genre must be diverse.
2. Description Tagging Requirements (Must be strictly followed):
   The description field must use a structured tag format, including the following tags, separated by commas:
   - Music Style tag
   - Music Genre tag
   - Instruments tag
   - Emotional Tone tag
   - Mood/Atmosphere tag
   - Vocal Style and Voice tag, limited to either "male voice" or "female voice", solo performance only.
   Note: Each tag should be concise. Multiple tags of the same category can be separated by a slash (e.g., "Piano/Violin").
3. Lyric Creativity: The lyrics should have depth and artistry:
   - Themes can cover various aspects such as love, life, society, nature, philosophy, dreams, memories, etc.
   - Use rich literary devices: metaphors, imagery, contrast, parallelism, etc.
   - Express sincere emotions with a focus on rhyme and rhythm.
   - The style can be narrative, lyrical, or stream-of-consciousness.
4. Lyric Structure and Length Requirements (Must be strictly followed):
   - The lyrics must be organized using the following structure, with section tags annotating each part.
   - The structure must strictly follow this order, for a total of 8 section tags: [Verse 1] → [Pre-Chorus] → [Chorus] → [Verse 2] → [Pre-Chorus] → [Chorus] → [Bridge] → [Chorus (Outro)].
   - A single song can only have these 8 section tags. [Verse 1] and [Verse 2] appear once; [Pre-Chorus] and [Chorus] appear twice; [Bridge] and [Chorus (Outro)] appear once. Do not add or repeat extra section tags.
   - Each section tag (e.g., [Verse 1], [Chorus]) must be on its own line, immediately followed by the LRC format lyrics for that section.
   - Separate sections with a blank line.
   - **Total Line Count Requirement**: The entire song must contain at least {require_length} lines of timestamped lyrics (not including section tags or blank lines).
5. LRC Format Mandatory Rules (Must be strictly followed):
   - Each line of lyrics must be in the format `[mm:ss.xx]Lyric content`, with no space between the timestamp and the lyrics. The lyric content should be coherent.
   - **Each line must contain only one short phrase of lyrics.** Start a new line when encountering punctuation like commas or periods.
   - **Strictly forbidden to merge multiple sentences or clauses onto the same line.**
   - Timestamps must be distributed naturally. **The first line's timestamp must not be [00:00.00]**. Allow for an instrumental intro (suggestion: start between [00:05.00] and [00:15.00]).
   - Timestamp intervals must be varied: The intervals within each song must be diverse, often using decimal values. Do not use a fixed interval:
     * A single song must contain a variety of different intervals; do not use the same interval for all lines (e.g., not all 4-second gaps).
     * Dynamically adjust intervals based on the emotional intensity and rhythm of the lyrics.
     * The gap between adjacent lines should vary to reflect the musical rhythm.
   - Timestamp allocation should be reasonably inferred based on the song's style, emotion, and rhythm, not mechanically assigned based on lyric length.
   - The length of each lyric line should vary naturally; do not make them all uniform.
   - **The total song duration must be between {start_duration} and {end_duration} (meaning the final line's timestamp must be between {start_timestamp} and {end_timestamp}). This is a hard requirement!**
6. Lyric Length Requirement: The number of lyric lines in the lyrics field must be greater than {require_length}. If the generated length is too short, please regenerate.
7. Uniqueness and Originality: Each piece should be unique. Avoid simply repeating the content from examples.
8. Format Requirements:
   - Directly return a JSON array containing 2 song objects. Each object must have only "description" and "lyrics" fields.
   - `description` field: Must be in tag format, not narrative text.
   - `lyrics` field: A string in LRC format with section tags.
   - Strictly forbidden to insert any extra symbols, markers, comments, or explanatory text within the JSON.

LRC Format Example (with section tags):
[Verse 1]
[00:08.00]First line of lyrics
[00:12.50]Second line of lyrics
[00:17.20]Third line of lyrics

[Pre-Chorus]
[00:22.00]Pre-chorus lyrics
[00:26.50]Pre-chorus lyrics

[Chorus]
[00:31.00]Chorus lyrics
[00:35.50]Chorus lyrics

Negative Examples (to avoid):
- Incorrect: [01:30.00](Piano Interlude) - Do not add parenthetical comments after the timestamp.
- Incorrect: [00:00.00]Starting lyric - The first line cannot start at 00:00.00.
- Incorrect: [00:05.00]In the familiar field, the sun casts golden rays upon the wheat - Strictly forbidden to place multiple clauses on the same line.
- Incorrect: [03:00.00] In the light of hope[03:05.50] In the light of hope[03:10.20] In the light of hope -Excessive repetition of the exact same lyric line is strictly forbidden. Lyrical content must show variation.
Now, please fully unleash your creativity and generate 2 new, complete works of music descriptions and LRC format lyrics.
Special Reminder: Each song must be complete, not abbreviated or omitted! It must contain the full 8 sections (Verse 1, Pre-Chorus, Chorus, Verse 2, Pre-Chorus, Chorus, Bridge, Chorus Outro) and strictly ensure more than {require_length} lines of lyrics.

Directly return in JSON array format:
[
  {{"description": "...", "lyrics": "..."}},
  {{"description": "...", "lyrics": "..."}}
]"""
    # Try to generate with retry mechanism
    for attempt in range(max_retries + 1):
        try:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are a creative music lyricist and composer. Please generate diverse and creative music tag-based descriptions and LRC format lyrics with song structure tags. CRITICAL REQUIREMENTS: 1) Description must be structured tags separated by commas, NOT narrative text. 2) Return ONLY pure, valid JSON format without any extra symbols, markers, or comments. 3) Each song must include structure tags like [Verse 1], [Chorus], [Bridge], etc., followed by LRC format lyrics [mm:ss.xx]lyric_content. 4) MANDATORY: Each song must have MORE than {require_length} lines of lyrics with timestamps. "},
                    {"role": "user", "content": prompt}
                ],
                n=1,  
                temperature=1.0,  
            )
            #print(prompt)
            # Extract all responses
            results = []
            filtered_count = 0
            last_content = None
            
            for i, choice in enumerate(completion.choices, 1):
                try:
                    content = choice.message.content.strip()
                    last_content = content
                    print(f"\n=== GPT Response {i} ===")
                    print(content)
                    print("=" * 50)
                    # Try to extract JSON content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    # Clean trailing commas in JSON (extra commas)
                    # Remove commas after last element of object/array
                    content = re.sub(r',(\s*[}\]])', r'\1', content)
                    
                    # Parse JSON array
                    result_array = json.loads(content)
                    
                    # Ensure it's a list
                    if isinstance(result_array, list):
                        # Validate each object in array
                        for song in result_array:
                            if isinstance(song, dict) and 'description' in song and 'lyrics' in song:
                                if _validate_timestamps(song.get('lyrics', '')):
                                    results.append(song)
                                else:
                                    filtered_count += 1
                    # If returned a single object (compatibility with old format)
                    elif isinstance(result_array, dict) and 'description' in result_array and 'lyrics' in result_array:
                        if _validate_timestamps(result_array.get('lyrics', '')):
                            results.append(result_array)
                        else:
                            filtered_count += 1
                        
                except json.JSONDecodeError:
                    continue
            
            if filtered_count:
                print(f"Total {filtered_count} songs filtered due to timestamp validation failure")
            
            # Print parsing results
            print(f"\nParsing complete, results length: {len(results)}")
            print(f"Results content: {results}")
            print(start_duration, end_duration,example1_timestamp,example2_timestamp,require_length)
            
            # If parsed result length is not 2, write model response content to test.txt
            if len(results) != 2:
                print(f"Warning: Parsed result length is not 2, actual is {len(results)}, will write to test.txt")
                with open('test.txt', 'w', encoding='utf-8') as f:
                    if last_content is not None:
                        f.write(last_content)
                print("Written to test.txt file")
            
            # Check if successfully generated 50 songs (10 responses * 5 each)
            if len(results) >= 50:
                # Append save results to file (use lock to ensure thread safety)
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for result in results[:50]:  # Only save first 50 songs
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                return selected_indices, min(len(results), 50)
            elif attempt < max_retries:
                print(f"Only successfully parsed {len(results)}/50 songs, retrying...")
                time.sleep(2)
            else:
                # Last attempt, save even if not 50 songs
                if len(results) > 0:
                    with file_lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            for result in results:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                return selected_indices, len(results)
                
        except Exception as e:
            if attempt < max_retries:
                print(f"Error occurred during generation: {e}, retrying...")
                time.sleep(2)
            else:
                print(f"Generation failed: {e}")
                return selected_indices, 0
    
    return selected_indices, 0


class IndexPool:
    """Thread-safe index pool with automatic reset support"""
    
    def __init__(self, total_size, selected_file):
        self.total_size = total_size
        self.selected_file = selected_file
        self.lock = threading.Lock()
        self.available_indices = []
        self.selected_indices = set()
        self.reset_count = 0  # Record reset count
        
        # Load selected indices from file
        self._load_selected_indices()
        # Initialize available indices
        self._reset_pool()
    
    def _load_selected_indices(self):
        """Load selected indices from file"""
        if os.path.exists(self.selected_file):
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.selected_indices.add(int(line.strip()))
    
    def _reset_pool(self):
        """Reset index pool"""
        # Calculate available indices
        self.available_indices = [i for i in range(self.total_size) if i not in self.selected_indices]
        random.shuffle(self.available_indices)  # Shuffle order
        
        if len(self.available_indices) == 0:
            # If no available indices, all have been used, reset selected_indices
            self.reset_count += 1
            print(f"\nIndex pool exhausted, resetting pool for the {self.reset_count}th time, re-selecting from {self.total_size} songs")
            self.selected_indices.clear()
            self.available_indices = list(range(self.total_size))
            random.shuffle(self.available_indices)
    
    def get_indices(self, count):
        """
        Thread-safe get specified number of indices
        
        Args:
            count: Number of indices needed
            
        Returns:
            List of selected indices
        """
        with self.lock:
            # Check if pool needs to be reset
            if len(self.available_indices) < count:
                self._reset_pool()
            
            # Get indices
            selected = self.available_indices[:count]
            self.available_indices = self.available_indices[count:]
            
            # Add to selected set
            for idx in selected:
                self.selected_indices.add(idx)
            
            # Write to file
            with open(self.selected_file, 'a', encoding='utf-8') as f:
                for idx in selected:
                    f.write(f"{idx}\n")
            
            return selected
    
    def get_stats(self):
        """Get statistics"""
        with self.lock:
            return {
                'available': len(self.available_indices),
                'selected': len(self.selected_indices),
                'reset_count': self.reset_count
            }


def batch_generate_music(input_file, output_file, selected_file, total_songs=1000, sample_size=20, model='gpt-4o-mini', num_threads=10):
    """
    Batch generate music descriptions and lyrics (multi-threaded version)
    
    Args:
        input_file: Path to input jsonl file
        output_file: Path to output jsonl file
        selected_file: Path to file recording selected indices
        total_songs: Total number of songs to generate
        sample_size: Number of samples to extract each time
        model: Model name to use
        num_threads: Number of threads
    """
    # Load all music data
    print("Loading music data...")
    all_music_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_music_data.append(data)
    print(f"Loaded {len(all_music_data)} songs")
    
    # Create thread-safe index pool
    index_pool = IndexPool(len(all_music_data), selected_file)
    stats = index_pool.get_stats()
    print(f"Currently selected indices: {stats['selected']}")
    print(f"Currently available indices: {stats['available']}")
    
    # Calculate number of calls needed (5 songs per call)
    num_iterations = (total_songs + 1) // 2  # Round up
    print(f"Need to call {num_iterations} times to generate approximately {total_songs} songs (5 per call)")
    print(f"Using {num_threads} threads for parallel processing\n")
    
    # Create file write lock
    file_lock = threading.Lock()
    
    # Statistics
    total_generated = 0
    generated_lock = threading.Lock()
    
    def worker_task(task_id):
        """Worker thread task"""
        try:
            used_indices, success_count = generate_music_descriptions(
                all_music_data=all_music_data,
                index_pool=index_pool,
                output_file=output_file,
                file_lock=file_lock,
                sample_size=sample_size,
                model=model,
                max_retries=0  # Retry
            )
            return success_count
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            return 0
    
    # Use thread pool and progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {executor.submit(worker_task, i): i for i in range(num_iterations)}
        
        # Use tqdm to show progress
        with tqdm(total=num_iterations, desc="Generation progress", unit="batch") as pbar:
            for future in as_completed(futures):
                success_count = future.result()
                
                with generated_lock:
                    total_generated += success_count
                
                # Get current statistics
                stats = index_pool.get_stats()
                
                # Update progress bar
                pbar.set_postfix({
                    'Batch': f'{success_count}/5',
                    'Total': total_generated,
                    'Remaining': stats['available'],
                    'Resets': stats['reset_count']
                })
                pbar.update(1)
    
    # Final statistics
    stats = index_pool.get_stats()
    print(f"\nGeneration complete!")
    print(f"Total generated: {total_generated} songs")
    print(f"Used {stats['selected']} indices")
    print(f"Remaining available indices: {stats['available']}")
    print(f"Pool reset count: {stats['reset_count']}")


if __name__ == '__main__':
    input_file = 'tagged_musics.jsonl'
    output_file = 'generate_en_lrc.jsonl'
    selected_file = 'selected.txt'
    # n=1, max_retries=0, sample 10 songs each time, generate 5 new songs
    batch_generate_music(
        input_file=input_file,
        output_file=output_file,
        selected_file=selected_file,
        total_songs=100,
        sample_size=4,
        model='gpt-4o-mini',
        num_threads=20  # Test with 1 thread first
    )
    # Append to txt file