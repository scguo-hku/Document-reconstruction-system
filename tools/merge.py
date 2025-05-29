import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
import os
import argparse
import pickle # For potential pkl loading if ever needed, though direct inference is primary now
import glob

# MMDetection specific imports
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
# from mmdet.structures.mask import PolygonMasks # If your model might output PolygonMasks

# --- Helper Functions (from original processer.py) ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def remove_shadows_basic(image_cv, max_filter_ksize=51):
    """
    通过邻域最大值估计背景光并进行补偿，使图像光照趋于统一。
    Args:
        image_cv: 输入的BGR图像。
        max_filter_ksize: 用于最大值滤波的邻域核大小 (应为奇数)。
    Returns:
        处理后的BGR图像，如果输入为None或处理失败则返回原始图像。
    """
    if image_cv is None:
        return None
    try:
        # 确保 max_filter_ksize 是奇数
        if max_filter_ksize % 2 == 0:
            max_filter_ksize += 1
            # print(f"    Info: max_filter_ksize for shadow removal was even, adjusted to {max_filter_ksize}")

        img_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # 1. 逐像素计算其邻域的最大值，作为背景值
        #    可以考虑适当增大 max_filter_ksize 的默认值或通过参数调整
        #    例如，如果文本区域较大，max_filter_ksize 可以是 35, 51 等
        kernel = np.ones((max_filter_ksize, max_filter_ksize), np.uint8)
        l_background = cv2.dilate(l_channel, kernel) # 最大值滤波

        # 2. 根据背景光补偿图对原图光照进行补偿
        #    动态调整 target_intensity
        #    使用原始L通道的较高百分位点作为目标亮度，有助于将背景提亮
        #    或者使用均值加上一个偏移
        # target_intensity_mean = np.mean(l_channel) # This line is overridden by the next percentile line
        # target_intensity = min(255, target_intensity_mean + 30) # This line is overridden by the next percentile line
        
        # 使用较高百分位点作为目标亮度，更能抵抗暗区的影响
        target_intensity = np.percentile(l_channel, 90) # 尝试80-90之间的百分位点
        # 如果图像整体非常暗，百分位点可能仍然很低，可以加一个保底值
        target_intensity = max(target_intensity, 160.0) # 确保目标亮度不低于150


        # 计算补偿后的L通道
        l_compensated_float = l_channel.astype(np.float32) - l_background.astype(np.float32) + target_intensity
        
        # 将值裁剪到 [0, 255]
        l_compensated_clipped = np.clip(l_compensated_float, 0, 255)

        # 可选：对补偿后的L通道进行轻微的对比度拉伸
        # 将l_compensated_clipped的最小最大值映射到0-255
        # 这有助于进一步区分前景和可能残留的非常浅的背景不均
        if np.min(l_compensated_clipped) < np.max(l_compensated_clipped): # 避免除以零
            l_compensated_normalized = cv2.normalize(l_compensated_clipped, None, 0, 255, cv2.NORM_MINMAX)
            l_compensated = l_compensated_normalized.astype(np.uint8)
        else:
            l_compensated = l_compensated_clipped.astype(np.uint8)


        merged_channels = cv2.merge([l_compensated, a_channel, b_channel])
        final_img = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
        return final_img
    except cv2.error as e:
        print(f"    Warning: OpenCV error during shadow removal (max_filter method): {e}")
        return image_cv # 返回原始图像以防处理失败
    except Exception as e:
        print(f"    Warning: Unexpected error during shadow removal (max_filter method): {e}")
        return image_cv

def binarize_image(image_cv):
    """
    将经过光照补偿的图像转换为灰度图，然后使用Otsu方法进行二值化。
    Args:
        image_cv: 输入的BGR图像 (期望是已经过光照补偿的)。
    Returns:
        处理后的3通道BGR二值图像（白色背景，黑色前景），如果输入为None则返回None。
    """
    if image_cv is None:
        return None
    try:
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        elif len(image_cv.shape) == 2: # 已经是灰度图
            gray_image = image_cv
        else:
            print(f"    Warning: binarize_image 接收到意外形状的图像: {image_cv.shape}")
            return image_cv # 返回原始图像

        # 3. 对图像进行otsu二值化
        # Otsu's method finds an optimal global threshold.
        # THRESH_BINARY: if pixel > thresh, it is set to maxval (255), else 0.
        # We want white background (255) and black text (0).
        # This implies that after shadow removal, text should be darker than background.
        _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # If Otsu results in black background and white text (meaning text was lighter than bg),
        # and you need white background, black text, you might need to invert.
        # However, the goal of remove_shadows_basic is to make background lighter.
        # For now, assume THRESH_BINARY with Otsu gives the desired (or near desired) output.
        # If text becomes white and background black, uncomment the next line:
        # binary_mask = cv2.bitwise_not(binary_mask)


        # 将单通道二值掩码转换回3通道BGR图像，以便后续一致处理
        binary_image_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        return binary_image_bgr
        
    except cv2.error as e:
        print(f"    Warning: OpenCV error during binarization (Otsu): {e}")
        return image_cv # 返回原始图像以防处理失败
    except Exception as e:
        print(f"    Warning: Unexpected error during binarization (Otsu): {e}")
        return image_cv # 返回原始图像以防处理失败

def calculate_mask_iou(mask1, mask2):
    if mask1 is None or mask2 is None or mask1.shape != mask2.shape:
        return 0.0
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def is_contained(mask_child, mask_parent):
    if mask_child is None or mask_parent is None or mask_child.shape != mask_parent.shape:
        return False
    return np.array_equal(np.logical_and(mask_child, mask_parent), mask_child)

# --- Core Processing Functions ---
def convert_datasample_to_rles(prediction_datasample: DetDataSample, image_filename: str = "unknown_image"):
    """
    Converts detection results from a DetDataSample object to a list of RLE entries.
    """
    all_rles = []
    pred_instances = prediction_datasample.pred_instances

    if not hasattr(pred_instances, 'masks') or pred_instances.masks is None:
        print(f"Warning: No 'masks' found in DetDataSample for {image_filename}.")
        return all_rles

    if isinstance(pred_instances.masks, BitmapMasks):
        instance_masks_np = pred_instances.masks.masks
    elif isinstance(pred_instances.masks, torch.Tensor):
        instance_masks_np = pred_instances.masks.cpu().numpy()
    elif isinstance(pred_instances.masks, np.ndarray):
        instance_masks_np = pred_instances.masks
    else:
        print(f"Warning: Unrecognized mask type {type(pred_instances.masks)} for {image_filename}.")
        return all_rles

    if instance_masks_np.ndim == 2: # Single mask, add batch dim
        instance_masks_np = instance_masks_np[None, ...]
    
    if len(instance_masks_np) == 0:
        return all_rles

    labels = pred_instances.labels.cpu().numpy() if hasattr(pred_instances, 'labels') else [0] * len(instance_masks_np)
    scores = pred_instances.scores.cpu().numpy() if hasattr(pred_instances, 'scores') else [0.0] * len(instance_masks_np)
    bboxes = pred_instances.bboxes.cpu().numpy() if hasattr(pred_instances, 'bboxes') else [None] * len(instance_masks_np)

    img_id_attr = getattr(prediction_datasample, 'img_id', image_filename)
    if img_id_attr is None and hasattr(prediction_datasample, 'metainfo') and 'img_id' in prediction_datasample.metainfo:
        img_id_attr = prediction_datasample.metainfo['img_id']

    for i in range(len(instance_masks_np)):
        binary_mask_bool = instance_masks_np[i]
        if binary_mask_bool.dtype != bool:
            binary_mask_bool = binary_mask_bool > 0 

        if np.sum(binary_mask_bool) == 0: 
            continue

        binary_mask_uint8 = binary_mask_bool.astype(np.uint8)
        fortran_mask = np.asfortranarray(binary_mask_uint8)
        rle = mask_util.encode(fortran_mask)

        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        
        rle_entry = {
            'image_id': img_id_attr,
            'category_id': int(labels[i]),
            'segmentation': rle,
            'score': float(scores[i]),
            'original_mask_shape': binary_mask_uint8.shape
        }

        if bboxes[i] is not None and len(bboxes[i]) == 4:
            x1, y1, x2, y2 = bboxes[i]
            rle_entry['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        
        all_rles.append(rle_entry)
    return all_rles


def warp_perspective_from_rle_entry(rle_entry_dict, original_image_cv, padding_pixels=0):
    """
    Processes a single RLE entry dictionary.
    Outputs a straightened rectangular image patch of the instance.
    """
    segmentation_rle_obj = rle_entry_dict['segmentation']
    current_rle_obj_copy = segmentation_rle_obj.copy()
    if isinstance(current_rle_obj_copy['counts'], str):
        current_rle_obj_copy['counts'] = current_rle_obj_copy['counts'].encode('utf-8')

    binary_mask = mask_util.decode(current_rle_obj_copy)
    if binary_mask is None or np.sum(binary_mask) == 0:
        return None

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    center, (width, height), angle = cv2.minAreaRect(contour)

    expanded_width = max(0, width + 2 * padding_pixels) 
    expanded_height = max(0, height + 2 * padding_pixels)
    
    if expanded_width == 0 or expanded_height == 0:
        return None
        
    expanded_rect_min_area = (center, (expanded_width, expanded_height), angle)
    
    box_points = cv2.boxPoints(expanded_rect_min_area)
    src_pts = order_points(box_points)
    
    (tl, tr, br, bl) = src_pts
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        return None

    dst_pts = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")
    
    try:
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    except cv2.error: 
        return None
        
    warped_image = cv2.warpPerspective(original_image_cv, perspective_matrix, (max_width, max_height))
    return warped_image

def reconstruct_visualization(original_image_dims, kept_instances_details, padding_value, save_path):
    """
    Pastes the straightened warped instances onto a new white canvas,
    centered at their original positions, and saves their location info.
    Args:
        original_image_dims (tuple): Dimensions (height, width) of the original image.
        kept_instances_details (list): List of dicts, each with 'rle_entry_dict' and 'warped_image_path'.
        padding_value (int): The padding that was applied during warping.
        save_path (str): Path to save the reconstructed image.
    """
    original_h, original_w = original_image_dims
    canvas_image = np.ones((original_h, original_w, 3), dtype=np.uint8) * 255
    
    pasted_instances_info = [] 

    print(f"  Reconstructing straightened visualization on white canvas with {len(kept_instances_details)} instances for {os.path.basename(save_path)}...")

    for instance_detail in kept_instances_details:
        rle_entry_dict = instance_detail['rle_entry_dict']
        # warped_image_path = instance_detail['warped_image_path'] # Old key, now using specific keys below
        image_to_paste_path = instance_detail['image_to_paste_path'] # Path to the image that will be loaded and pasted
        json_image_path = instance_detail['json_image_path'] # Path to be written into the JSON file

        if not os.path.exists(image_to_paste_path): # Check existence of the image to be pasted
            print(f"    Warning: Warped image not found: {image_to_paste_path}. Skipping.")
            continue
        
        straightened_instance_img = cv2.imread(image_to_paste_path) # Load the (potentially processed) image for pasting
        if straightened_instance_img is None:
            print(f"    Warning: Could not load warped image: {image_to_paste_path}. Skipping.")
            continue

        img_h, img_w = straightened_instance_img.shape[:2]
        if img_h == 0 or img_w == 0:
            print(f"    Warning: Warped image has zero dimension: {image_to_paste_path}. Skipping.")
            continue

        segmentation_rle_obj = rle_entry_dict['segmentation']
        current_rle_obj_copy = segmentation_rle_obj.copy()
        if isinstance(current_rle_obj_copy['counts'], str):
            current_rle_obj_copy['counts'] = current_rle_obj_copy['counts'].encode('utf-8')
        
        binary_mask_orig = mask_util.decode(current_rle_obj_copy)
        if binary_mask_orig is None or np.sum(binary_mask_orig) == 0:
            print(f"    Warning: Could not decode RLE for center calculation for {warped_image_path}. Skipping.")
            continue 
        
        contours_orig, _ = cv2.findContours(binary_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_orig:
            print(f"    Warning: No contours found for center calculation for {warped_image_path}. Skipping.")
            continue
        
        contour_orig = max(contours_orig, key=cv2.contourArea)
        center_orig_float, (orig_w_float, orig_h_float), orig_angle_float = cv2.minAreaRect(contour_orig) 

        paste_dst_x1 = int(round(center_orig_float[0] - img_w / 2.0))
        paste_dst_y1 = int(round(center_orig_float[1] - img_h / 2.0))
        
        src_x1 = 0
        src_y1 = 0
        src_x2 = img_w 
        src_y2 = img_h 

        if paste_dst_x1 < 0:
            src_x1 = -paste_dst_x1
            paste_dst_x1 = 0
        if paste_dst_y1 < 0:
            src_y1 = -paste_dst_y1
            paste_dst_y1 = 0
        
        eff_width = min(paste_dst_x1 + (src_x2 - src_x1), original_w) - paste_dst_x1
        eff_height = min(paste_dst_y1 + (src_y2 - src_y1), original_h) - paste_dst_y1

        if eff_width > 0 and eff_height > 0:
            final_src_x2 = src_x1 + eff_width
            final_src_y2 = src_y1 + eff_height

            final_dst_x1 = paste_dst_x1
            final_dst_y1 = paste_dst_y1
            final_dst_x2 = paste_dst_x1 + eff_width
            final_dst_y2 = paste_dst_y1 + eff_height
            
            try:
                canvas_image[final_dst_y1:final_dst_y2, final_dst_x1:final_dst_x2] = \
                    straightened_instance_img[src_y1:final_src_y2, src_x1:final_src_x2]
                
                instance_pos_info = {
                    'warped_image_path': json_image_path, # Use the path of the original warped image for the JSON
                    'original_center_xy': (float(center_orig_float[0]), float(center_orig_float[1])),
                    'straightened_image_dims_wh': (img_w, img_h),
                    'canvas_paste_rect_xyxy': (final_dst_x1, final_dst_y1, final_dst_x2, final_dst_y2),
                    'source_crop_rect_xyxy': (src_x1, src_y1, final_src_x2, final_src_y2),
                    'rle_segmentation': rle_entry_dict['segmentation'] 
                }
                pasted_instances_info.append(instance_pos_info)

            except Exception as e:
                print(f"    Error during pasting instance from {warped_image_path} to canvas: {e}")
                print(f"    Canvas slice: [{final_dst_y1}:{final_dst_y2}, {final_dst_x1}:{final_dst_x2}] (Shape: {canvas_image[final_dst_y1:final_dst_y2, final_dst_x1:final_dst_x2].shape})")
                print(f"    Source slice: [{src_y1}:{final_src_y2}, {src_x1}:{final_src_x2}] (Shape: {straightened_instance_img[src_y1:final_src_y2, src_x1:final_src_x2].shape})")

    try:
        cv2.imwrite(save_path, canvas_image)
        print(f"  Reconstructed straightened visualization on white canvas saved to: {save_path}")
        
        json_save_path = os.path.splitext(save_path)[0] + "_info.json"
        with open(json_save_path, 'w') as f_json:
            json.dump(pasted_instances_info, f_json, indent=4)
        print(f"  Pasted instances positional information saved to: {json_save_path}")

    except Exception as e:
        print(f"  Error saving reconstructed image or info {save_path}: {e}")


def process_single_image(image_path, model, args):
    """
    Full processing pipeline for a single image.
    """
    print(f"\nProcessing image: {image_path}")
    original_image_cv = cv2.imread(image_path)
    if original_image_cv is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return
    
    original_image_dims = (original_image_cv.shape[0], original_image_cv.shape[1]) 
    image_filename = os.path.basename(image_path)

    print(f"  Running inference on {image_filename}...")
    result_datasample = inference_detector(model, image_path) 

    if isinstance(result_datasample, list):
        if not result_datasample:
            print(f"  Inference returned empty list for {image_filename}. Skipping.")
            return
        result_datasample = result_datasample[0] 

    if not isinstance(result_datasample, DetDataSample):
        print(f"  Inference did not return a DetDataSample for {image_filename}. Type: {type(result_datasample)}. Skipping.")
        return

    rle_data_list = convert_datasample_to_rles(result_datasample, image_filename)

    if not rle_data_list:
        print(f"  No RLEs generated for {image_filename}. Skipping further processing.")
        return
    print(f"  Generated {len(rle_data_list)} RLE entries for {image_filename}.")

    candidate_instances = []
    for i, rle_entry in enumerate(rle_data_list):
        current_rle_obj = rle_entry['segmentation'].copy()
        if isinstance(current_rle_obj['counts'], str):
            current_rle_obj['counts'] = current_rle_obj['counts'].encode('utf-8')
        binary_mask = mask_util.decode(current_rle_obj)
        if binary_mask is None or np.sum(binary_mask) == 0:
            continue
        candidate_instances.append({
            'rle_entry_dict': rle_entry,
            'binary_mask': binary_mask,
            'score': rle_entry.get('score', 0.0),
            'original_rle_index': i, 
            'area': np.sum(binary_mask)
        })

    if not candidate_instances:
        print(f"  No valid candidate instances after RLE decoding for {image_filename}.")
        return

    num_candidates = len(candidate_instances)
    keep_flags = [True] * num_candidates
    print(f"  Starting filtering for {num_candidates} candidates from {image_filename}...")
    # --- Filtering logic (assumed to be correct and complete) ---
    for i in range(num_candidates):
        if not keep_flags[i]: continue
        inst_i = candidate_instances[i]
        mask_i, score_i, original_idx_i, area_i = inst_i['binary_mask'], inst_i['score'], inst_i['original_rle_index'], inst_i['area']

        for j in range(i + 1, num_candidates):
            if not keep_flags[j]: continue
            inst_j = candidate_instances[j]
            mask_j, score_j, original_idx_j, area_j = inst_j['binary_mask'], inst_j['score'], inst_j['original_rle_index'], inst_j['area']
            
            handled_by_rule = False
            i_in_j = is_contained(mask_i, mask_j)
            j_in_i = is_contained(mask_j, mask_i)

            if i_in_j and j_in_i: # Identical
                if score_i < score_j: keep_flags[i] = False
                elif score_j < score_i: keep_flags[j] = False
                else: 
                    if original_idx_i < original_idx_j: keep_flags[j] = False
                    else: keep_flags[i] = False
                handled_by_rule = True
            elif i_in_j: # i strictly in j
                if score_i <= score_j: keep_flags[i] = False 
                else: keep_flags[j] = False 
                handled_by_rule = True
            elif j_in_i: # j strictly in i
                if score_j <= score_i: keep_flags[j] = False 
                else: keep_flags[i] = False 
                handled_by_rule = True
            
            if handled_by_rule:
                if not keep_flags[i]: break 
                if not keep_flags[j]: continue 
                continue
            
            intersection = np.sum(np.logical_and(mask_i, mask_j))
            if intersection == 0: continue 

            ios_val = 0.0
            smaller_is_i = (area_i <= area_j)
            if smaller_is_i:
                ios_val = intersection / area_i if area_i > 0 else 0
            else:
                ios_val = intersection / area_j if area_j > 0 else 0
            
            if ios_val > args.ios_threshold:
                if smaller_is_i: 
                    if score_i <= score_j: 
                        keep_flags[i] = False; handled_by_rule = True
                else: 
                    if score_j <= score_i: 
                        keep_flags[j] = False; handled_by_rule = True
                if handled_by_rule:
                    if not keep_flags[i]: break 
                    if not keep_flags[j]: continue
                    continue 
            
            if not handled_by_rule: 
                union = area_i + area_j - intersection
                iou = intersection / union if union > 0 else 0
                if iou > args.iou_threshold:
                    if score_i < score_j:
                        keep_flags[i] = False
                    elif score_j < score_i:
                        keep_flags[j] = False
                    else: 
                        if original_idx_i < original_idx_j: keep_flags[j] = False
                        else: keep_flags[i] = False
                    if not keep_flags[i]: break
    # --- End Filtering logic ---
    
    image_specific_output_dir = os.path.join(args.output_dir, os.path.splitext(image_filename)[0])
    if not os.path.exists(image_specific_output_dir):
        os.makedirs(image_specific_output_dir)

    original_warped_dir = os.path.join(image_specific_output_dir, "original_warped")
    processed_warped_dir = os.path.join(image_specific_output_dir, "processed_warped")
    debug_shadow_dir = os.path.join(image_specific_output_dir, "debug_shadow_removal")

    for d in [original_warped_dir, processed_warped_dir, debug_shadow_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    print(f"  Saving instances for {image_filename} to respective subdirectories in {image_specific_output_dir}...")
    saved_count_processed = 0
    saved_count_original = 0
    kept_instances_for_reconstruction = []

    FIGURE_CATEGORY_ID = args.figure_category_id 

    for k_idx in range(num_candidates):
        if keep_flags[k_idx]:
            instance_info = candidate_instances[k_idx]
            rle_entry_content = instance_info['rle_entry_dict']
            category_id = rle_entry_content.get('category_id', -1)
            score = instance_info['score']
            original_rle_idx = instance_info['original_rle_index']

            original_warped_region = warp_perspective_from_rle_entry(rle_entry_content, original_image_cv, padding_pixels=args.padding)
            
            if original_warped_region is None:
                print(f"    Warning: Warping failed for instance {original_rle_idx}. Skipping.")
                continue

            original_filename_base = f"original_cat{category_id}_inst{original_rle_idx}_score{score:.2f}_pad{args.padding}.png"
            original_output_path = os.path.join(original_warped_dir, original_filename_base)
            try:
                cv2.imwrite(original_output_path, original_warped_region)
                saved_count_original += 1
            except Exception as e:
                print(f"    Error saving original warped image {original_output_path}: {e}")
            
            current_processed_region = original_warped_region.copy()
            # Apply shadow removal using the passed max_filter_ksize from args
            shadow_removed_region = remove_shadows_basic(current_processed_region, max_filter_ksize=args.max_filter_ksize)
            
            if shadow_removed_region is None:
                print(f"    Warning: Shadow removal returned None for instance {original_rle_idx}. Using original warped.")
                shadow_removed_region = current_processed_region # Fallback
            
            if args.save_debug_images and shadow_removed_region is not None:
                debug_filename_base = f"cat{category_id}_inst{original_rle_idx}_score{score:.2f}"
                debug_color_path = os.path.join(debug_shadow_dir, f"{debug_filename_base}_shadow_removed_color.png")
                cv2.imwrite(debug_color_path, shadow_removed_region)
                gray_processed_region = cv2.cvtColor(shadow_removed_region, cv2.COLOR_BGR2GRAY)
                debug_gray_path = os.path.join(debug_shadow_dir, f"{debug_filename_base}_shadow_removed_gray.png")
                cv2.imwrite(debug_gray_path, gray_processed_region)

            final_region_to_save = shadow_removed_region 

            is_figure = (category_id == FIGURE_CATEGORY_ID)
            
            if is_figure:
                print(f"    Instance {original_rle_idx} (Category ID: {category_id}) is a figure. Skipping binarization. Using shadow-removed.")
                # final_region_to_save is already shadow_removed_region
            else: 
                if shadow_removed_region is not None:
                    # Using Otsu binarization as per user's selection
                    binarized_region = binarize_image(shadow_removed_region)
                    if binarized_region is None:
                        print(f"    Warning: Binarization (Otsu) returned None for instance {original_rle_idx}. Using shadow-removed region.")
                    else:
                        final_region_to_save = binarized_region 
                else:
                    print(f"    Skipping binarization for instance {original_rle_idx} as input (shadow_removed_region) is None.")

            processed_filename_base = f"processed_cat{category_id}_inst{original_rle_idx}_score{score:.2f}_pad{args.padding}.png"
            processed_output_path = os.path.join(processed_warped_dir, processed_filename_base)
            try:
                if final_region_to_save is not None:
                    cv2.imwrite(processed_output_path, final_region_to_save)
                    saved_count_processed += 1
                    kept_instances_for_reconstruction.append({
                        'rle_entry_dict': rle_entry_content,
                        'image_to_paste_path': processed_output_path, # Path of the image actually being pasted
                        'json_image_path': original_output_path     # Path to record in the JSON (original warped path)
                    })
                else:
                    print(f"    Error: Final region to save is None for instance {original_rle_idx}.")
            except Exception as e:
                print(f"    Error saving processed image {processed_output_path}: {e}")
    
    print(f"  Finished processing instances for {image_filename}.")
    print(f"    Saved {saved_count_original} original warped instances to {original_warped_dir}.")
    print(f"    Saved {saved_count_processed} processed (shadow-removed/binarized) instances to {processed_warped_dir}.")

    if kept_instances_for_reconstruction:
        reconstruction_save_path = os.path.join(args.output_dir, f"{os.path.splitext(image_filename)[0]}_reconstructed_straight.png")
        reconstruct_visualization(
            original_image_dims, 
            kept_instances_for_reconstruction,
            args.padding, 
            reconstruction_save_path
        )

def main():
    parser = argparse.ArgumentParser(description="Batch inference, RLE conversion, filtering, and warping of segmented regions.")
    parser.add_argument("config_file", help="Path to the MMDetection config file.")
    parser.add_argument("checkpoint_file", help="Path to the MMDetection checkpoint file.")
    parser.add_argument("output_dir", help="Main directory to save all processed outputs.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", help="Directory containing images to process.")
    input_group.add_argument("--image_path", help="Path to a single image file to process.")

    parser.add_argument("--device", default='cuda:0', help="Device for inference (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--padding", type=int, default=10, help="Pixels to expand bounding box (default: 10).")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for overlap filtering (default: 0.5).")
    parser.add_argument("--ios_threshold", type=float, default=0.8, help="IoS threshold for covered smaller instances (default: 0.8).")
    parser.add_argument('--image_extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
                        help="Image extensions for --image_dir.")
    
    # Argument for remove_shadows_basic
    parser.add_argument("--max_filter_ksize", type=int, default=51, help="Kernel size for max filter in shadow removal (must be odd).")
    
    # Argument for figure category ID
    parser.add_argument("--figure_category_id", type=int, default=4, help="Category ID that represents 'figure' instances (to skip binarization). Default is 4.")
    parser.add_argument("--save_debug_images", action='store_true', help="Save intermediate debug images (e.g., after shadow removal).")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Main output directory: {args.output_dir}")

    # Validate ksize
    if args.max_filter_ksize <= 1: 
        args.max_filter_ksize = 3
        print(f"Adjusted max_filter_ksize to be 3 (minimum odd).")
    elif args.max_filter_ksize % 2 == 0: 
        args.max_filter_ksize +=1
        print(f"Adjusted max_filter_ksize to be odd: {args.max_filter_ksize}")
    
    print("Initializing MMDetection model...")
    try:
        model = init_detector(args.config_file, args.checkpoint_file, device=args.device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing MMDetection model: {e}")
        return

    image_paths_to_process = []
    if args.image_path:
        if os.path.isfile(args.image_path): image_paths_to_process.append(args.image_path)
        else: print(f"Error: Image path '{args.image_path}' not found."); return
    elif args.image_dir:
        if os.path.isdir(args.image_dir):
            for ext in args.image_extensions:
                image_paths_to_process.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
            if not image_paths_to_process: print(f"No images found in '{args.image_dir}' with specified extensions."); return
            print(f"Found {len(image_paths_to_process)} images in '{args.image_dir}'.")
        else: print(f"Error: Image directory '{args.image_dir}' not found."); return
    
    for img_path in image_paths_to_process:
        process_single_image(img_path, model, args)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()
