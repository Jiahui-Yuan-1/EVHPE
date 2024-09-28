import sys
import numpy as np
from utils.event_util import generate_input_representation
import os
import numpy as np
from pathlib import Path



# 定义根目录路径
root_path = Path('/home/yuanjh/EventHeadpose/data/Davis')

# 定义事件表示，形状和其他参数，具体值根据需要进行调整
event_representation = 'voxel_grid'
# shape = (600, 600)  # 假定目标形状
shape = (260,346)

# 遍历根目录下的所有子目录
# for category in ['car', 'indoor']:
for category in ['indoor']:
    category_path = root_path / category
    if category_path.exists() and category_path.is_dir():
        for person_name in category_path.iterdir():
            if person_name.is_dir():  # 确保是目录
                for subfolder in person_name.iterdir():
                    if subfolder.is_dir():
                        event_folder = subfolder / 'frame_events'
                        voxel_folder = subfolder / 'voxel_5_260_346_frame'
                        voxel_folder.mkdir(parents=True, exist_ok=True)  # 确保voxel目录存在
                        
                        # 处理event目录下的每个.npy文件
                        for event_npy in event_folder.glob('*.npy'):

                            # 构造输出路径并保存处理后的数组
                            output_path = voxel_folder / event_npy.name
                            if not output_path.exists():
                                # 加载.npy文件
                                array = np.load(event_npy)
                                # 调用处理函数
                                array = generate_input_representation(array, event_representation, shape, 5)
                                
                                np.save(output_path, array)
                                print(f"Processed and saved: {output_path}")

print("Finished processing and saving arrays.")
