from PIL import Image, UnidentifiedImageError
from surya.table_rec import TableRecPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from shapely.geometry import Polygon
import os
import re
import argparse
import shutil # 用于复制文件

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- 辅助函数 (部分来自之前的版本) ---
def calculate_overlap_ratio(bbox1, bbox2):
    poly1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3]), (bbox1[0], bbox1[3])])
    poly2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3]), (bbox2[0], bbox2[3])])
    if not poly1.is_valid or not poly2.is_valid: return 0
    intersection_area = poly1.intersection(poly2).area
    smaller_area = min(poly1.area, poly2.area)
    return intersection_area / smaller_area if smaller_area > 0 else 0

def fill_table_with_text(cells, text_lines):
    """
    根据重叠面积占更小一方面积的比例将文字填入表格单元格中。
    (此版本来自用户原始的 table.py 逻辑，填充 cell.text_lines)
    """
    for cell in cells:
        cell_bbox = cell.bbox
        cell_text_list = [] # Changed variable name for clarity
        for text_line in text_lines:
            text_bbox = text_line.bbox
            overlap_ratio = calculate_overlap_ratio(cell_bbox, text_bbox)
            if overlap_ratio > 0.5:  # 重叠比例阈值
                cell_text_list.append(text_line.text)
        # 为 cell 对象动态添加属性来存储提取的文本行
        cell.text_lines = cell_text_list if cell_text_list else []

def process_math_tags(text):
    if not text: return ""
    def replace_superscript(match): return f"<sup>{match.group(1)}</sup>"
    processed_text = re.sub(r"<math>(.*?)</math>", r"\1", text)
    processed_text = re.sub(r"\^\{(.*?)\}", replace_superscript, processed_text)
    return processed_text

def save_table_as_html(table_data, output_path):
    """
    将表格保存为 HTML 文件，并使用 CSS 格式化。
    (此版本基于用户原始的 table.py 中的 save_table_as_css 函数)
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Table</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-size: 16px;
                text-align: left;
            }
            th, td {
                border: 1px solid #dddddd;
                padding: 8px;
                text-align: center; /* Original script had center */
                word-wrap: break-word; /* Added from ocr.py for robustness */
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
    </head>
    <body>
        <table>
    """

    # 添加表格内容
    for row in table_data:
        html_content += "<tr>\n" # \n for readability, original script didn't have it here
        for cell_info in row: # cell_info is a dict or None
            if cell_info is None:
                continue  # 跳过合并的单元格
            text = cell_info.get("text", "") # Safely get text
            rowspan = cell_info.get("rowspan", 1)
            colspan = cell_info.get("colspan", 1)
            rowspan_attr = f" rowspan='{rowspan}'" if rowspan > 1 else ""
            colspan_attr = f" colspan='{colspan}'" if colspan > 1 else ""
            html_content += f"  <td{rowspan_attr}{colspan_attr}>{text}</td>\n" # \n for readability
        html_content += "</tr>\n" # \n for readability

    # 关闭 HTML 标签
    html_content += """
        </table>
    </body>
    </html>
    """

    # 保存到文件
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html_content)
        print(f"表格已保存为 HTML 文件：{output_path}")
    except IOError as e:
        print(f"IOError saving HTML file {output_path}: {e}")
    except Exception as e:
        print(f"Unexpected error saving HTML file {output_path}: {e}")

# --- 新的主要处理逻辑 ---

CATEGORY_MAP = {
    "text": [0, 1, 2, 8, 9, 10, 11],
    "image": [4, 5, 6],
    "equation": [3],
    "table": [7]
}

def get_category_type(category_id):
    for cat_type, ids in CATEGORY_MAP.items():
        if category_id in ids:
            return cat_type
    return "unknown" # 或 "text" 作为默认

def process_text_or_equation_patch(image_pil, recognition_pred, detection_pred, output_txt_path):
    """处理单个文本或公式图像块并保存结果。"""
    predictions = recognition_pred([image_pil], det_predictor=detection_pred)
    ocr_text_content = []
    if predictions and predictions[0] and hasattr(predictions[0], 'text_lines') and predictions[0].text_lines:
        for line in predictions[0].text_lines:
            ocr_text_content.append(line.text)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ocr_text_content))
    print(f"OCR 结果已保存到: {output_txt_path}")

def process_table_patch(image_pil, recognition_pred, detection_pred, table_rec_pred, output_html_path):
    """
    处理单个表格图像块并保存为HTML。
    (此函数现在采用用户原始 table.py 的核心逻辑)
    """
    # 1. 获取表格预测结果
    table_predictions_list = table_rec_pred([image_pil]) # List of TableResult objects
        
    table_object = table_predictions_list[0] # Get the first Table object

    # 2. 获取文字识别结果 (OCR)
    ocr_predictions_list = recognition_pred([image_pil], det_predictor=detection_pred)
    all_text_lines = []
    if ocr_predictions_list and ocr_predictions_list[0] and hasattr(ocr_predictions_list[0], 'text_lines'):
        all_text_lines = ocr_predictions_list[0].text_lines
    else:
        print(f"表格单元格填充的OCR失败: {os.path.basename(output_html_path)}")
        # Decide if to proceed with empty cells or return; original script proceeds.

    # 3. 提取表格基本信息
    # surya's Table object has `rows` and `cols` attributes which are lists of Row and Column objects
    # The number of rows/columns is len(table_object.rows) and len(table_object.cols)
    num_rows = len(table_object.rows)
    num_cols = len(table_object.cols)
    cells = table_object.cells # List of Cell objects

    if not cells:
        print(f"表格块 {os.path.basename(output_html_path)} 没有单元格数据。")
        with open(output_html_path.replace(".html", "_nocells.txt"), "w") as f:
            f.write("No cells found in the table structure.")
        return

    # 4. 将文字填入表格单元格 (使用更新后的 fill_table_with_text)
    fill_table_with_text(cells, all_text_lines) # This will populate cell.text_lines

    # 5. 构建二维数组表示表格，处理 rowspan 和 colspan
    table_representation = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for cell in cells: # cell is a surya.table_rec.Cell object
        row_id = cell.row_id
        col_id = cell.col_id
        # Use getattr for robustness, though original script accessed directly
        rowspan = getattr(cell, 'rowspan', 1)
        colspan = getattr(cell, 'colspan', 1)
        
        text_content = ""
        if hasattr(cell, 'text_lines') and cell.text_lines: # Check attribute from fill_table_with_text
            text_content = " ".join(cell.text_lines)

        # 填充单元格内容，处理 rowspan 和 colspan
        # Original script's loop structure:
        for r in range(row_id, row_id + rowspan):
            for c in range(col_id, col_id + colspan):
                # Crucial boundary check
                if 0 <= r < num_rows and 0 <= c < num_cols:
                    if r == row_id and c == col_id: # 主单元格
                        # Store raw text here; math processing later
                        table_representation[r][c] = {"text": text_content, "rowspan": rowspan, "colspan": colspan}
                    else: # 合并的单元格标记为 None
                        table_representation[r][c] = None
                # else:
                #     print(f"Warning: Cell span for cell at ({row_id},{col_id}) goes out of bounds at ({r},{c}). Table size: ({num_rows},{num_cols})")


    # 6. 处理表格内容中的 <math> 标签 (as per original script)
    for r_idx in range(len(table_representation)):
        for c_idx in range(len(table_representation[r_idx])):
            cell_data = table_representation[r_idx][c_idx]
            if cell_data is not None and "text" in cell_data:
                cell_data["text"] = process_math_tags(cell_data["text"])
    
    # 7. 保存表格为 HTML 文件 (使用更新后的 save_table_as_html)
    # print(f"Debug: table_representation for {os.path.basename(output_html_path)}:") # Optional debug
    # for r_idx, rep_row in enumerate(table_representation):
    #     print(f"  Row {r_idx}: {rep_row}")
    save_table_as_html(table_representation, output_html_path)

def main():
    parser = argparse.ArgumentParser(description="对指定目录结构中的图像块进行分类OCR处理。")
    parser.add_argument("input_dir", help="包含原始图像标号子目录的顶层输入目录 (例如 'output' 文件夹)。")
    parser.add_argument("output_dir", help="保存OCR结果的总输出目录。")
    
    args = parser.parse_args()

    # 初始化预测器
    print("正在初始化预测器...")
    try:
        table_rec_pred = TableRecPredictor()
        recognition_pred = RecognitionPredictor()
        detection_pred = DetectionPredictor()
        print("预测器初始化完成。")
    except Exception as e:
        print(f"初始化预测器时出错: {e}")
        return

    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在。")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历 input_dir 下的每个子目录 (如 "00140", "00141" 等)
    for image_id_folder_name in os.listdir(args.input_dir):
        image_id_folder_path = os.path.join(args.input_dir, image_id_folder_name)
        if not os.path.isdir(image_id_folder_path):
            continue

        print(f"\n正在处理图像ID: {image_id_folder_name}")
        
        # 查找 original_warped 文件夹
        original_warped_path = os.path.join(image_id_folder_path, "original_warped")
        if not os.path.isdir(original_warped_path):
            print(f"  未找到 'original_warped' 文件夹于 {image_id_folder_path}，跳过。")
            continue

        # 创建该图像ID对应的输出子目录
        current_output_base = os.path.join(args.output_dir, image_id_folder_name)
        os.makedirs(current_output_base, exist_ok=True)
        
        output_dirs = {
            "text": os.path.join(current_output_base, "text_results"),
            "image": os.path.join(current_output_base, "image_results"),
            "equation": os.path.join(current_output_base, "equation_results"),
            "table": os.path.join(current_output_base, "table_results"),
            "unknown": os.path.join(current_output_base, "unknown_results")
        }
        for _, path in output_dirs.items():
            os.makedirs(path, exist_ok=True)

        # 遍历 original_warped 文件夹中的每个图像块
        for patch_filename in os.listdir(original_warped_path):
            patch_filepath = os.path.join(original_warped_path, patch_filename)
            if not (patch_filename.lower().endswith(".png") or patch_filename.lower().endswith(".jpg")):
                continue

            print(f"  处理图像块: {patch_filename}")
            
            # 从文件名提取类别ID
            match = re.search(r"cat(\d+)", patch_filename)
            if not match:
                print(f"    无法从文件名 {patch_filename} 提取类别ID，跳过。")
                # 可以选择将其复制到 unknown_results
                try:
                    shutil.copy(patch_filepath, os.path.join(output_dirs["unknown"], patch_filename))
                except Exception as e:
                    print(f"    复制到unknown时出错: {e}")
                continue
            
            category_id = int(match.group(1))
            category_type = get_category_type(category_id)
            
            print(f"    类别ID: {category_id}, 类型: {category_type}")

            try:
                image_pil = Image.open(patch_filepath).convert("RGB")
            except FileNotFoundError:
                print(f"    错误: 图像块文件未找到 {patch_filepath}")
                continue
            except UnidentifiedImageError:
                print(f"    错误: 无法识别的图像文件 {patch_filepath}")
                continue
            except Exception as e:
                print(f"    加载图像块 {patch_filepath} 时出错: {e}")
                continue

            base_patch_name = os.path.splitext(patch_filename)[0]

            if category_type == "text":
                output_txt_path = os.path.join(output_dirs["text"], f"{base_patch_name}_ocr.txt")
                process_text_or_equation_patch(image_pil, recognition_pred, detection_pred, output_txt_path)
            
            elif category_type == "image":
                output_image_path = os.path.join(output_dirs["image"], patch_filename) # Destination filename is always patch_filename

                if category_id in [5, 6]: # Specific logic for cat 5 and 6 (assumed binarized)
                    binarized_dir_name = "processed_warped"  # Assumed directory for binarized images
                    binarized_source_filename = ""
                    
                    if patch_filename.startswith("original_"):
                        binarized_source_filename = "processed_" + patch_filename[len("original_"):]
                    
                    binarized_source_path = ""
                    if binarized_source_filename:
                        # image_id_folder_path is the parent of original_warped_path, e.g., /path/to/00140
                        binarized_source_path = os.path.join(image_id_folder_path, binarized_dir_name, binarized_source_filename)

                    copied_successfully = False
                    if binarized_source_path and os.path.exists(binarized_source_path):
                        try:
                            shutil.copy(binarized_source_path, output_image_path)
                            print(f"    处理后图像 (cat {category_id}) 已从 {binarized_source_path} 复制到: {output_image_path}")
                            copied_successfully = True
                        except Exception as e:
                            print(f"    复制处理后图像 {binarized_source_path} 到 {output_image_path} 时出错: {e}")
                    else:
                        if not binarized_source_filename:
                             print(f"    警告: 文件名 {patch_filename} (cat {category_id}) 不是以 'original_' 开头。无法确定处理后图像的文件名。")
                        elif binarized_source_path: # Only print if path was formed but file not found
                             print(f"    警告: 处理后图像 {binarized_source_path} (cat {category_id}) 未找到。")
                    
                    if not copied_successfully:
                        print(f"    回退: 正在从 {patch_filepath} 复制原始图像。")
                        try:
                            shutil.copy(patch_filepath, output_image_path) # patch_filepath is the original image
                            print(f"    原始图像已复制到: {output_image_path}")
                        except Exception as e:
                            print(f"    复制原始图像 (回退) {patch_filepath} 到 {output_image_path} 时出错: {e}")
                else: # For other image categories (e.g., cat 4)
                    try:
                        shutil.copy(patch_filepath, output_image_path) # patch_filepath is from original_warped
                        print(f"    图像 (cat {category_id}) 已从 {patch_filepath} 复制到: {output_image_path}")
                    except Exception as e:
                        print(f"    复制图像 {patch_filepath} 到 {output_image_path} 时出错: {e}")
            
            elif category_type == "equation":
                output_txt_path = os.path.join(output_dirs["equation"], f"{base_patch_name}_ocr.txt")
                process_text_or_equation_patch(image_pil, recognition_pred, detection_pred, output_txt_path)

            elif category_type == "table":
                output_html_path = os.path.join(output_dirs["table"], f"{base_patch_name}_table.html")
                process_table_patch(image_pil, recognition_pred, detection_pred, table_rec_pred, output_html_path)
            
            else: # unknown
                print(f"    未知类别类型 {category_type} (来自ID {category_id})，复制原始文件。")
                output_unknown_path = os.path.join(output_dirs["unknown"], patch_filename)
                try:
                    shutil.copy(patch_filepath, output_unknown_path)
                except Exception as e:
                    print(f"    复制到unknown时出错: {e}")
    
    print("\n所有图像ID处理完成。")

if __name__ == "__main__":
    main()