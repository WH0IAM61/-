import pandas as pd
import math as m
import sys
from datetime import datetime

# ===================== 核心计算函数 =====================
def deg2rad(deg):
    return deg * m.pi / 180.0

def calculate_distance(lon1, lat1, lon2, lat2):
    R = 6371000  # 地球半径（米）
    lon1_rad = deg2rad(lon1)
    lat1_rad = deg2rad(lat1)
    lon2_rad = deg2rad(lon2)
    lat2_rad = deg2rad(lat2)
    
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    
    a = m.sin(delta_lat/2)**2 + m.cos(lat1_rad) * m.cos(lat2_rad) * m.sin(delta_lon/2)**2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
    distance = R * c  # 单位：米
    return round(distance, 2)

def calculate_direction(lon1, lat1, lon2, lat2):
    if pd.isna(lon1) or pd.isna(lat1) or pd.isna(lon2) or pd.isna(lat2):
        return None
    
    lon1_rad = deg2rad(lon1)
    lat1_rad = deg2rad(lat1)
    lon2_rad = deg2rad(lon2)
    lat2_rad = deg2rad(lat2)
    
    delta_lon = lon2_rad - lon1_rad
    y = m.sin(delta_lon) * m.cos(lat2_rad)
    x = m.cos(lat1_rad) * m.sin(lat2_rad) - m.sin(lat1_rad) * m.cos(lat2_rad) * m.cos(delta_lon)
    bearing_deg = (m.degrees(m.atan2(y, x)) + 360) % 360
    
    deg = int(bearing_deg)
    min = round((bearing_deg - deg) * 60)
    bearing_str = f"{deg}°{min}′"
    return bearing_str

def calculate_speed(distance_m, time_diff_seconds):
    if time_diff_seconds <= 0 or distance_m <= 0:
        return 0.0
    speed = distance_m / time_diff_seconds
    return round(speed, 2)

# ===================== 主处理函数 =====================
def process_trajectory_tsv(input_file, output_file, read_rows=0):
    try:
        # 1. 读取原始数据
        col_names = ["user_id", "datetime_str", "longitude", "latitude", "transport_mode"]
        read_kwargs = {
            "sep": "\t",
            "names": col_names,
            "dtype": {
                "user_id": str,
                "datetime_str": str,
                "longitude": float,
                "latitude": float,
                "transport_mode": str
            },
            "na_filter": False,
        }
        if read_rows > 0:
            read_kwargs["nrows"] = read_rows
        
        print(f"开始读取数据（{'全部行' if read_rows == 0 else f'前{read_rows}行'}）...")
        df = pd.read_csv(input_file, **read_kwargs)
        print(f"✅ 成功读取 {len(df)} 行原始数据")
        
        # 2. 数据清洗
        df = df.dropna(subset=["longitude", "latitude"])
        # 可选：非东京数据请注释下面两行
        df = df[
            (df["longitude"].between(139.29, 140.09)) &
            (df["latitude"].between(35.39, 35.89))
        ].reset_index(drop=True)
        print(f"✅ 清洗后剩余 {len(df)} 行有效数据")
        
        # 3. 时间解析
        print("⏳ 解析时间格式...")
        df["datetime"] = pd.to_datetime(
            df["datetime_str"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
        print(f"✅ 时间解析完成，剩余 {len(df)} 行有效数据")
        
        # 4. 初始化新列
        df["move_direction"] = None
        df["move_distance_m"] = 0.0
        df["speed_m_s"] = 0.0
        
        # 5. 按用户分组计算特征
        print("⏳ 计算方向/距离/速度...")
        user_ids = df["user_id"].unique()
        total_users = len(user_ids)
        
        for idx, user_id in enumerate(user_ids, 1):
            group = df[df["user_id"] == user_id].sort_values("datetime").reset_index(drop=True)
            group_len = len(group)
            
            if group_len < 2:
                continue
            
            for i in range(1, group_len):
                prev_row = group.iloc[i-1]
                curr_row = group.iloc[i]
                
                distance = calculate_distance(prev_row["longitude"], prev_row["latitude"],
                                             curr_row["longitude"], curr_row["latitude"])
                direction = calculate_direction(prev_row["longitude"], prev_row["latitude"],
                                               curr_row["longitude"], curr_row["latitude"])
                time_diff = (curr_row["datetime"] - prev_row["datetime"]).total_seconds()
                speed = calculate_speed(distance, time_diff)
                
                df.loc[curr_row.name, "move_direction"] = direction
                df.loc[curr_row.name, "move_distance_m"] = distance
                df.loc[curr_row.name, "speed_m_s"] = speed
            
            if idx % 100 == 0 or idx == total_users:
                print(f"已处理 {idx}/{total_users} 个用户")
        
        # 6. 构建输出数据
        output_df = df[["user_id", "date", "time", "move_direction", "move_distance_m", "speed_m_s"]]
        
        # --- 核心修改：生成固定宽度的对齐文本 ---
        print("⏳ 生成对齐格式的输出文件...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in output_df.iterrows():
                # 为每一列设定固定宽度，用空格填充对齐
                # 格式: 左对齐，总宽度分别为 10, 12, 10, 15, 12, 10
                line = (
                    f"{row['user_id']:<10} "
                    f"{row['date']:<12} "
                    f"{row['time']:<10} "
                    f"{str(row['move_direction']):<15} "
                    f"{row['move_distance_m']:>12.2f} "  # 数字右对齐
                    f"{row['speed_m_s']:>10.2f}\n"       # 数字右对齐
                )
                f.write(line)
        
        print(f"✅ 处理完成！输出文件已保存至：{output_file}")
        print(f"📊 最终输出 {len(output_df)} 行数据")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_file}，请检查路径是否正确")
        sys.exit(1)
    except PermissionError:
        print(f"❌ 错误：没有权限访问 {input_file} 或 {output_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理出错：{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ===================== 运行配置 =====================
if __name__ == "__main__":
    INPUT_TSV = r"D:/浙大/启真问学/数据/OpenPFLOW/DataSet/trajectory01.tsv"
    OUTPUT_TSV = r"D:/浙大/启真问学/数据/OpenPFLOW/DataSet/quantified_trajectory01.tsv"
    
    # 先测试前10000行，确认没问题后改为0（全量处理）
    READ_ROWS = 10000
    
    process_trajectory_tsv(INPUT_TSV, OUTPUT_TSV, read_rows=READ_ROWS)
