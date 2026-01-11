import json
import os

def extract_user_prompt(messages):
    """
    从messages列表中提取第一个role为user的content，提取风格字符串
    
    格式通常是："Please generate a song in the following style:（风格描述）\n"
    只保留冒号后面的风格字符串部分
    
    Args:
        messages: 字典列表，每个字典包含role和content字段
        
    Returns:
        风格字符串
    """
    # 找到第一个role为user的消息
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                # 查找 "Please generate a song in the following style:" 的位置
                style_prefix = "Please generate a song in the following style:"
                style_index = content.find(style_prefix)
                
                if style_index != -1:
                    # 找到冒号后面的内容起始位置
                    start_index = style_index + len(style_prefix)
                    # 查找换行符的位置
                    newline_index = content.find("\n", start_index)
                    
                    if newline_index != -1:
                        # 提取冒号后到换行符前的内容
                        style_text = content[start_index:newline_index].strip()
                    else:
                        # 如果没有换行符，提取到字符串末尾
                        style_text = content[start_index:].strip()
                    
                    return style_text
                else:
                    # 如果没有找到标准格式，返回空字符串
                    return ""
    
    # 如果没有找到user消息，返回空字符串
    return ""

def update_song_file(file_path, new_prompt):
    """
    更新song文件中的style_prompt字段
    
    Args:
        file_path: song文件的路径
        new_prompt: 新的prompt内容
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        print(f"  警告: 文件 {file_path} 为空，跳过")
        return
    
    # 读取第一条JSON数据
    try:
        data = json.loads(lines[0])
        # 更新style_prompt字段
        data['style_prompt'] = new_prompt
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            # 如果有第二行空行，保留
            if len(lines) > 1:
                f.write('\n')
        
        print(f"  ✓ 已更新 {file_path}")
    except json.JSONDecodeError as e:
        print(f"  错误: 解析JSON失败 {file_path}: {e}")
    except Exception as e:
        print(f"  错误: 更新文件失败 {file_path}: {e}")

def main():
    # 文件路径
    input_file = "xxx/diffrhythm2/scripts/test_messages.jsonl"
    zh_songs_dir = "xxx/diffrhythm2/example/zh_songs"
    en_songs_dir = "xxx/diffrhythm2/example/en_songs"
    
    print(f"正在读取文件: {input_file}")
    
    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"共读取到 {len(lines)} 条数据")
    
    # 处理每条数据
    for idx, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # 提取prompt
            prompt = extract_user_prompt(messages)
            
            if not prompt:
                print(f"处理第 {idx} 条数据: 未找到user content，跳过")
                continue
            
            # 判断是中文还是英文
            if idx <= 50:
                # 前50条：中文歌曲
                song_num = idx
                target_dir = zh_songs_dir
                lang = "中文"
            else:
                # 51-100条：英文歌曲
                song_num = idx - 50  # 51->1, 52->2, ..., 100->50
                target_dir = en_songs_dir
                lang = "英文"
            
            # 构建文件路径
            song_file = os.path.join(target_dir, f"song_{song_num}.jsonl")
            
            print(f"处理第 {idx} 条数据 ({lang}，song_{song_num})...")
            print(f"  Prompt长度: {len(prompt)} 字符")
            
            # 更新文件
            update_song_file(song_file, prompt)
            
        except json.JSONDecodeError as e:
            print(f"处理第 {idx} 条数据时JSON解析失败: {e}")
            continue
        except Exception as e:
            print(f"处理第 {idx} 条数据时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n处理完成！共处理 {len(lines)} 条数据")

if __name__ == "__main__":
    main()

