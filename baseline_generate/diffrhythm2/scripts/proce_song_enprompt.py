import json
import os

def extract_user_prompt(messages):
    """
    Extract all content from messages where role is user, and concatenate them
    When concatenating each content, check if the concatenated prompt length exceeds 1600,
    if so, do not concatenate and skip subsequent segments to ensure paragraph integrity
    
    Args:
        messages: List of dictionaries, each containing role and content fields
        
    Returns:
        Concatenated prompt string
    """
    # Collect all user message content, but need to check length limit
    user_contents = []
    current_length = 0  # Current concatenated length
    
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                # Calculate total length if this content is added
                # Need to consider newline: if content already exists, need to add a newline
                if user_contents:
                    # Content already exists, need to add newline and current content
                    new_length = current_length + 1 + len(content)  # 1 is newline length
                else:
                    # First content, no newline needed
                    new_length = len(content)
                
                # If adding this content doesn't exceed 1600, add it
                if new_length <= 1600:
                    user_contents.append(content)
                    current_length = new_length
                else:
                    # Exceeds 1600, don't add this content and stop processing subsequent segments
                    break
    
    # Concatenate all content with newlines
    if user_contents:
        prompt = "\n".join(user_contents)
        return prompt
    
    # If no user message found, return empty string
    return ""

def update_song_file(file_path, new_prompt):
    """
    Update style_prompt field in song file
    
    Args:
        file_path: Path to song file
        new_prompt: New prompt content
    """
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        print(f"  Warning: File {file_path} is empty, skipping")
        return
    
    # Read first JSON data
    try:
        data = json.loads(lines[0])
        # Update style_prompt field
        data['style_prompt'] = new_prompt
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            # If there's a second empty line, keep it
            if len(lines) > 1:
                f.write('\n')
        
        print(f"  ✓ Updated {file_path}")
    except json.JSONDecodeError as e:
        print(f"  Error: JSON parsing failed {file_path}: {e}")
    except Exception as e:
        print(f"  Error: Failed to update file {file_path}: {e}")

def main():
    # File paths
    input_file = "xxx/diffrhythm2/scripts/test_messages.jsonl"
    zh_songs_dir = "xxx/diffrhythm2/example/zh_songs"
    en_songs_dir = "xxx/diffrhythm2/example/en_songs"
    
    print(f"Reading file: {input_file}")
    
    # Read all data
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Read {len(lines)} entries")
    
    # Process each entry
    for idx, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # Extract prompt
            prompt = extract_user_prompt(messages)
            
            if not prompt:
                print(f"Processing entry {idx}: No user content found, skipping")
                continue
            
            # Determine if Chinese or English
            if idx <= 50:
                # First 50 entries: Chinese songs
                song_num = idx
                target_dir = zh_songs_dir
                lang = "Chinese"
            else:
                # Entries 51-100: English songs
                song_num = idx - 50  # 51->1, 52->2, ..., 100->50
                target_dir = en_songs_dir
                lang = "English"
            
            # Build file path
            song_file = os.path.join(target_dir, f"song_{song_num}.jsonl")
            
            print(f"Processing entry {idx} ({lang}, song_{song_num})...")
            print(f"  Prompt length: {len(prompt)} characters")
            
            # Update file
            update_song_file(song_file, prompt)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for entry {idx}: {e}")
            continue
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete! Processed {len(lines)} entries")

if __name__ == "__main__":
    main()

