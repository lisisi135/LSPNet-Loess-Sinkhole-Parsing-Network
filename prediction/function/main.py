import openpyxl
import time
import os
import re
import numpy as np
from .hole_parameters import hole_parameters
from .hole_extraction import hole_extraction
import geopandas as gpd
from shapely.geometry import Polygon



class SinkholeProcessor:
    def __init__(self, folder_path, output_folder, excel_filename="参数数据库.xlsx", shp_filename="地图.shp"):
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.excel_filename = excel_filename
        self.shp_filename = shp_filename
        os.makedirs(self.output_folder, exist_ok=True)
        self.workbook = openpyxl.Workbook()
        self.sheet = self.workbook.active
        self.set_excel_header()

    def set_excel_header(self):
        """设置 Excel 文件的表头"""
        headers = [
            '落水洞编号', '体积', '点数', '深度', '长轴长度', '短轴长度', '面积', '周长', '延伸率', '圆度指数'
        ]
        for col, header in enumerate(headers, 1):
            self.sheet.cell(row=1, column=col).value = header

    def extract_column_number(self, file_path):
        """从文件路径中提取目录编号"""
        try:
            file_name = os.path.basename(file_path)
            match = re.search(r'\d+', file_name)
            return int(match.group()) if match else None
        except Exception as e:
            print(f"提取编号时出错 {file_path}: {e}")
            return None

    def read_points(self, file_path):
        """从文件中读取点云数据（提前分配数组大小）"""
        try:
            row_count = 0
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            float(parts[0])
                            float(parts[1])
                            float(parts[2])
                            row_count += 1
                        except ValueError:
                            continue

            if row_count == 0:
                return np.array([])

            points = np.empty((row_count, 3), dtype=np.float64)
            index = 0

            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            z = float(parts[2])
                            points[index] = [x, y, z]
                            index += 1
                        except ValueError:
                            continue

            return points
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return np.array([])

    def process_file(self, file_path):
        """处理单个文件，计算各种几何参数"""
        try:
            point_cloud = self.read_points(file_path)

            column_number = self.extract_column_number(file_path)
            if column_number is None:
                print(f"{file_path} 序号提取错误，可能是格式不满足要求请检查。")
                return {
                    "column_number": column_number,
                    "file_path": file_path,
                    "success": False
                }

            if len(point_cloud) < 10:
                print(f"文件 {file_path} 点云数量少于10个，返回默认值0")
                return {
                    "column_number": column_number,
                    "total_volume": 0,
                    "point_count": len(point_cloud),
                    "max_elevation": 0,
                    "min_elevation": 0,
                    "depth": 0,
                    "major_axis": 0,
                    "minor_axis": 0,
                    "cave_area": 0,
                    "perimeter": 0,
                    "extension_rate": 0,
                    "roundness_index": 0,
                    "points": [],
                    "file_path": file_path,
                    "success": True
                }

            point_count = len(point_cloud)

            max_elevation = np.max(point_cloud[:, 2]) if len(point_cloud) > 0 else 0
            min_elevation = np.min(point_cloud[:, 2]) if len(point_cloud) > 0 else 0

            cave_points_processor = hole_extraction(slice_thickness=0.5)
            cropped_points, _ = cave_points_processor.process(point_cloud)

            '''洞口参数计算'''
            analyzer = hole_parameters(cropped_points).analyze()

            '''凹包'''
            cave_area = analyzer.concave_hull.area
            perimeter = analyzer.concave_hull.perimeter
            points = analyzer.concave_hull.hull_points
            '''凸包'''
            major_axis = analyzer.bounding_rect.long_axis
            minor_axis = analyzer.bounding_rect.short_axis

            depth = max_elevation - min_elevation
            total_volume = depth * cave_area
            extension_rate = major_axis / minor_axis if minor_axis != 0 else 0
            roundness_index = (4 * np.pi * cave_area) / (perimeter ** 2) if perimeter != 0 else 0


            return {
                "column_number": column_number,
                "total_volume": total_volume,
                "point_count": point_count,
                "max_elevation": max_elevation,
                "min_elevation": min_elevation,
                "depth": depth,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "cave_area": cave_area,
                "perimeter": perimeter,
                "extension_rate": extension_rate,
                "roundness_index": roundness_index,
                "points": points,
                "file_path": file_path,
                "success": True
            }

        except FileNotFoundError:
            print(f"文件 {file_path} 不存在，请检查！")
            return {"success": False, "file_path": file_path}
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return {"success": False, "file_path": file_path}

    def process_all_files(self):
        """单线程顺序处理文件夹内所有文件，并存储结果"""
        start_time = time.time()
        results = []

        txt_files = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')]
        total_files = len(txt_files)
        processed_files = 0
        success_count = 0

        if total_files == 0:
            print("未发现任何txt文件，请检查文件夹路径")
            return

        print(f"发现 {total_files} 个txt文件，开始单线程处理...")
        print("-" * 50)

        for file_name in txt_files:
            processed_files += 1
            file_path = os.path.join(self.folder_path, file_name)

            try:
                result = self.process_file(file_path)
                if result and result.get("success", False):
                    success_count += 1
                    if result["column_number"] is not None:
                        results.append(result)
                        print(f"已处理 {processed_files}/{total_files} 个，成功 {success_count} 个 | 成功：{file_name}")
                    else:
                        print(f"已处理 {processed_files}/{total_files} 个，成功 {success_count} 个 | 无编号：{file_name}")
                else:
                    print(f"已处理 {processed_files}/{total_files} 个，成功 {success_count} 个 | 失败：{file_name}")
            except Exception as e:
                print(f"已处理 {processed_files}/{total_files} 个，成功 {success_count} 个 | 异常：{file_name}，错误：{e}")

        print("-" * 50)

        results.sort(key=lambda x: x["column_number"])
        self.create_excel_file(results)
        self.create_shapefile(results)

        end_time = time.time()
        print(f"\n处理完成！共处理 {total_files} 个文件，成功 {success_count} 个，耗时 {end_time - start_time:.2f} 秒")

    def create_excel_file(self, results):
        """将结果保存到 Excel 文件（按排序后的顺序写入，避免0编号覆盖）"""

        for row_idx, result in enumerate(results, start=2):
            self.sheet.cell(row=row_idx, column=1).value = result["column_number"]
            self.sheet.cell(row=row_idx, column=2).value = round(result["total_volume"], 2)
            self.sheet.cell(row=row_idx, column=3).value = result["point_count"]
            self.sheet.cell(row=row_idx, column=4).value = round(result["depth"], 2)
            self.sheet.cell(row=row_idx, column=5).value = round(result["major_axis"], 2)
            self.sheet.cell(row=row_idx, column=6).value = round(result["minor_axis"], 2)
            self.sheet.cell(row=row_idx, column=7).value = round(result["cave_area"], 2)
            self.sheet.cell(row=row_idx, column=8).value = round(result["perimeter"], 2)
            self.sheet.cell(row=row_idx, column=9).value = round(result["extension_rate"], 2)
            self.sheet.cell(row=row_idx, column=10).value = round(result["roundness_index"], 2)

        excel_path = os.path.join(self.output_folder, self.excel_filename)
        self.workbook.save(excel_path)
        print(f"Excel 文件已保存：{excel_path}")

    def create_shapefile(self, results):
        """将结果保存到 Shapefile 文件（只包含坡度参数，改用英文短字段名）"""
        polygons = []

        for result in results:
            points = result["points"]
            if len(points) > 0:
                try:
                    polygon = Polygon(points)
                    polygons.append({
                        'FID': result["column_number"],
                        'Volume': result["total_volume"],
                        'PointCount': result["point_count"],
                        'Depth': result["depth"],
                        'MajorAxis': result["major_axis"],
                        'MinorAxis': result["minor_axis"],
                        'Area': result["cave_area"],
                        'Perimeter': result["perimeter"],
                        'ExtRatio': result["extension_rate"],
                        'Roundness': result["roundness_index"],
                        'geometry': polygon
                    })
                except Exception as e:
                    print(f"创建多边形时出错 (编号 {result['column_number']}): {e}")

        if not polygons:
            print("没有可保存的多边形数据，跳过Shapefile创建")
            return

        all_polygons_gdf = gpd.GeoDataFrame(polygons, geometry='geometry')

        shp_path = os.path.join(self.output_folder, self.shp_filename)
        try:
            all_polygons_gdf.to_file(shp_path, encoding='utf-8')
            print(f"Shapefile 文件已保存：{shp_path}")
        except Exception as e:
            print(f"保存Shapefile时出错: {e}")


if __name__ == "__main__":
    folder_path = r"/LSPNet/prediction/predict_result/single_sinkhole"# 替换为你的实际路径
    output_folder = r"/LSPNet/prediction/predict_result/result"# 替换为你的实际路径

    os.makedirs(output_folder, exist_ok=True)

    processor = SinkholeProcessor(
        folder_path=folder_path,
        output_folder=output_folder,
        excel_filename="数据表.xlsx",
        shp_filename="图层.shp"
    )
    processor.process_all_files()