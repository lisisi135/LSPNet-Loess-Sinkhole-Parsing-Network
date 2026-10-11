from function.main import SinkholeProcessor
import os

folder_path = r"/single_sinkhole"# 替换为你的实际路径
output_folder = r"/result"# 替换为你的实际路径

os.makedirs(output_folder, exist_ok=True)

processor = SinkholeProcessor(
    folder_path=folder_path,
    output_folder=output_folder,
    excel_filename="数据表.xlsx",
    shp_filename="图层.shp"
)
processor.process_all_files()