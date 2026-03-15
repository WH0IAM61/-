import pandas as pd
import json
import random
from datetime import datetime

# 初始化随机种子（保证每次运行结果不同）
random.seed(datetime.now().timestamp())

def load_trajectory_data(file_path):
    """
    读取量化轨迹数据（适配你的空格分隔格式，处理0值/特殊direction格式）
    :param file_path: 轨迹文件路径
    :return: 清洗后的轨迹DataFrame
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 去除首尾空格，按空格分割（适配你的数据格式）
            line_stripped = line.strip()
            if not line_stripped:  # 过滤空行
                continue
            
            # 分割规则：按空格分割，但保证最终是6列（适配direction含°′的情况）
            parts = line_stripped.split()
            if len(parts) != 6:
                print(f"⚠️ 第{line_num}行格式异常（列数≠6），跳过：{line_stripped}")
                continue
            
            # 提取字段并做类型转换（增加异常捕获）
            try:
                user_id = parts[0].strip()
                date = parts[1].strip()
                time = parts[2].strip()
                direction = parts[3].strip() if parts[3].strip() != 'None' else None
                distance = float(parts[4].strip())
                speed = float(parts[5].strip())
            except ValueError as e:
                print(f"⚠️ 第{line_num}行数值转换失败，跳过：{e} | 内容：{line_stripped}")
                continue
            
            # 存储数据（保留0值行，后续生成样本时过滤）
            data.append({
                'user_id': user_id,
                'date': date,
                'time': time,
                'direction': direction,
                'distance': distance,
                'speed': speed
            })
    
    # 转为DataFrame并按用户+时间排序
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("❌ 未读取到任何有效轨迹数据，请检查文件路径/格式！")
    
    # 解析datetime（处理时间格式异常）
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df = df.dropna(subset=['datetime'])  # 过滤时间解析失败的行
    df = df.sort_values(['user_id', 'datetime']).reset_index(drop=True)
    
    # 过滤全0轨迹段（仅保留有有效运动的轨迹点）
    df_valid = df[(df['distance'] > 0) | (df['speed'] > 0)]
    if len(df_valid) == 0:
        raise ValueError("❌ 数据中无有效轨迹点（distance/speed均为0）！")
    
    print(f"✅ 读取完成：总行数{len(df)} → 有效轨迹点{len(df_valid)}")
    return df_valid

def generate_training_samples(df, output_jsonl, samples_per_user=5):
    """
    生成随机N的训练样本（随机起始位置+随机N，避免0值样本）
    :param df: 清洗后的轨迹DataFrame
    :param output_jsonl: 输出JSONL路径
    :param samples_per_user: 每个用户生成的样本数
    """
    # 按用户分组
    user_groups = df.groupby('user_id')
    samples = []
    
    for user_id, group in user_groups:
        # 提取该用户的轨迹序列（按时间排序）
        trajectory = group.to_dict('records')
        traj_len = len(trajectory)
        
        # 过滤轨迹长度不足的用户（至少需要2个历史点+1个预测点）
        if traj_len < 3:
            print(f"⚠️ 用户{user_id}轨迹长度{traj_len}，不足3个点，跳过")
            continue
        
        # 为每个用户生成多个随机样本
        for _ in range(samples_per_user):
            # 步骤1：随机选择起始索引（避免只取开头）
            max_start_idx = traj_len - 3  # 保证能取到N个历史点+1个预测点
            start_idx = random.randint(0, max_start_idx)
            
            # 步骤2：随机选择N值（2 ≤ N ≤ 剩余轨迹长度-1）
            remaining_len = traj_len - start_idx
            random_n = random.randint(2, remaining_len - 1)
            
            # 步骤3：截取历史点和目标点
            history_points = trajectory[start_idx:start_idx+random_n]
            target_point = trajectory[start_idx+random_n]
            
            # 构建历史轨迹文本（英文，适配大模型）
            history_text = []
            for point in history_points:
                dir_str = point['direction'] if point['direction'] else 'None'
                history_text.append(
                    f"{point['date']} {point['time']}: direction {dir_str}, distance {point['distance']:.2f} meters, speed {point['speed']:.2f} meters per second"
                )
            history_str = '; \n'.join(history_text) + ';'
            
            # 构建指令（instruction）
            instruction = f"""Given the following {random_n} trajectory features of the user with ID {user_id} on {history_points[0]['date']} (arranged in chronological order):
{history_str}
Please predict the user's moving direction (in degrees and minutes format, e.g., 179°60′), moving distance (in meters, keep two decimal places), and moving speed (in meters per second, keep two decimal places) at the next time point."""
            
            # 构建答案（output）
            target_dir = target_point['direction'] if target_point['direction'] else 'None'
            output = f"""The user's moving direction at the next time point is {target_dir}, the moving distance is {target_point['distance']:.2f} meters, and the speed is {target_point['speed']:.2f} meters per second."""
            
            # 生成样本（大模型微调标准格式）
            sample = {
                "instruction": instruction,
                "input": "",
                "output": output
            }
            samples.append(sample)
    
    # 保存为JSONL文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 样本生成完成！共生成{len(samples)}个样本，保存至：{output_jsonl}")

# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 配置文件路径（替换为你的实际路径）
    TRAJECTORY_FILE = r"D:/浙大/启真问学/数据/OpenPFLOW/DataSet/quantified_trajectory01.tsv"
    OUTPUT_JSONL = r"D:/浙大/启真问学/数据/OpenPFLOW/DataSet/train_samples_random_n.jsonl"
    
    # 执行数据读取和样本生成
    try:
        df = load_trajectory_data(TRAJECTORY_FILE)
        generate_training_samples(df, OUTPUT_JSONL, samples_per_user=5)
    except Exception as e:
        print(f"❌ 程序执行失败：{e}")
