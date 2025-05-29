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

def calculate_mask_iou(mask1, mask2):
    if mask1 is None or mask2 is None or mask1.shape != mask2.shape:
        return 0.0
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def is_contained(mask_child, mask_parent):
    return np.array_equal(np.logical_and(mask_child, mask_parent), mask_child)

# --- Core Processing Functions ---
def convert_datasample_to_rles(prediction_datasample: DetDataSample, image_filename: str = "unknown_image"):
    """
    Converts detection results from a DetDataSample object to a list of RLE entries.
    Adapted from the original pkl_to_rle function.
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
        # print(f"Info: No instances found in masks for {image_filename}.")
        return all_rles

    labels = pred_instances.labels.cpu().numpy() if hasattr(pred_instances, 'labels') else [0] * len(instance_masks_np)
    scores = pred_instances.scores.cpu().numpy() if hasattr(pred_instances, 'scores') else [0.0] * len(instance_masks_np)
    bboxes = pred_instances.bboxes.cpu().numpy() if hasattr(pred_instances, 'bboxes') else [None] * len(instance_masks_np)

    img_id_attr = getattr(prediction_datasample, 'img_id', image_filename) # Use filename as img_id if not present
    if img_id_attr is None and hasattr(prediction_datasample, 'metainfo') and 'img_id' in prediction_datasample.metainfo:
        img_id_attr = prediction_datasample.metainfo['img_id']


    for i in range(len(instance_masks_np)):
        binary_mask_bool = instance_masks_np[i]
        if binary_mask_bool.dtype != bool:
            binary_mask_bool = binary_mask_bool > 0 # Ensure boolean

        if np.sum(binary_mask_bool) == 0: # Skip empty masks
            continue

        binary_mask_uint8 = binary_mask_bool.astype(np.uint8)
        fortran_mask = np.asfortranarray(binary_mask_uint8)
        rle = mask_util.encode(fortran_mask)

        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        
        rle_entry = {
            'image_id': img_id_attr, # This will be the original image filename
            'category_id': int(labels[i]),
            'segmentation': rle,
            'score': float(scores[i]),
            'original_mask_shape': binary_mask_uint8.shape # Store original mask shape for context
        }

        if bboxes[i] is not None and len(bboxes[i]) == 4:
            x1, y1, x2, y2 = bboxes[i]
            rle_entry['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        
        all_rles.append(rle_entry)
    return all_rles


def warp_perspective_from_rle_entry(rle_entry_dict, original_image_cv, padding_pixels=0):
    """
    Processes a single RLE entry dictionary (not the 'segmentation' sub-dict directly).
    """
    segmentation_rle_obj = rle_entry_dict['segmentation'] # Get the RLE object
    
    current_rle_obj_copy = segmentation_rle_obj.copy() # Work on a copy
    if isinstance(current_rle_obj_copy['counts'], str):
        current_rle_obj_copy['counts'] = current_rle_obj_copy['counts'].encode('utf-8')

    binary_mask = mask_util.decode(current_rle_obj_copy)
    if binary_mask is None or np.sum(binary_mask) == 0:
        # print(f"Warning: Empty or invalid mask for RLE entry during warping.")
        return None

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # print(f"Warning: No contours found for RLE entry during warping.")
        return None

    contour = max(contours, key=cv2.contourArea)
    center, (width, height), angle = cv2.minAreaRect(contour)

    expanded_width = width + 2 * padding_pixels
    expanded_height = height + 2 * padding_pixels
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
        # print(f"Warning: Invalid dimensions ({max_width}x{max_height}) for warped image after padding.")
        return None

    dst_pts = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(original_image_cv, perspective_matrix, (max_width, max_height))
    return warped_image


def process_single_image(image_path, model, args):
    """
    Full processing pipeline for a single image:
    Inference -> RLE conversion -> Filtering -> Warping & Saving.
    """
    print(f"\nProcessing image: {image_path}")
    original_image_cv = cv2.imread(image_path)
    if original_image_cv is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return

    # 1. Inference
    print(f"  Running inference on {os.path.basename(image_path)}...")
    result_datasample = inference_detector(model, image_path) # Returns a DetDataSample or list of them

    # Handle if inference_detector returns a list (e.g. for video frames, though not typical for single image)
    if isinstance(result_datasample, list):
        if not result_datasample:
            print(f"  Inference returned empty list for {os.path.basename(image_path)}. Skipping.")
            return
        result_datasample = result_datasample[0] # Take the first for single image processing

    if not isinstance(result_datasample, DetDataSample):
        print(f"  Inference did not return a DetDataSample for {os.path.basename(image_path)}. Type: {type(result_datasample)}. Skipping.")
        return

    # 2. Convert DetDataSample to RLE list
    image_filename = os.path.basename(image_path)
    rle_data_list = convert_datasample_to_rles(result_datasample, image_filename)

    if not rle_data_list:
        print(f"  No RLEs generated from inference results for {image_filename}. Skipping filtering and warping.")
        return
    print(f"  Generated {len(rle_data_list)} RLE entries for {image_filename}.")

    # 3. Filter RLEs (Containment, IoS, IoU)
    candidate_instances = []
    for i, rle_entry in enumerate(rle_data_list):
        # The RLE object is rle_entry['segmentation']
        current_rle_obj = rle_entry['segmentation'].copy()
        if isinstance(current_rle_obj['counts'], str):
            current_rle_obj['counts'] = current_rle_obj['counts'].encode('utf-8')
        
        binary_mask = mask_util.decode(current_rle_obj)
        if binary_mask is None or np.sum(binary_mask) == 0:
            # print(f"  Skipping RLE entry original_index={i} for {image_filename} due to empty/invalid mask post-RLE conversion.")
            continue
        
        candidate_instances.append({
            'rle_entry_dict': rle_entry, # Store the full RLE entry dictionary
            'binary_mask': binary_mask,
            'score': rle_entry.get('score', 0.0),
            'original_rle_index': i, # Index within this image's RLE list
            'area': np.sum(binary_mask)
        })

    if not candidate_instances:
        print(f"  No valid candidate instances after decoding RLEs for {image_filename}. Skipping further processing.")
        return

    num_candidates = len(candidate_instances)
    keep_flags = [True] * num_candidates
    print(f"  Starting filtering for {num_candidates} candidates from {image_filename}...")

    for i in range(num_candidates):
        if not keep_flags[i]: continue
        inst_i = candidate_instances[i]
        mask_i, score_i, original_idx_i, area_i = inst_i['binary_mask'], inst_i['score'], inst_i['original_rle_index'], inst_i['area']

        for j in range(i + 1, num_candidates):
            if not keep_flags[j]: continue
            inst_j = candidate_instances[j]
            mask_j, score_j, original_idx_j, area_j = inst_j['binary_mask'], inst_j['score'], inst_j['original_rle_index'], inst_j['area']
            
            handled_by_rule = False
            # Rule 1: Containment
            i_in_j = is_contained(mask_i, mask_j)
            j_in_i = is_contained(mask_j, mask_i)

            if i_in_j and j_in_i: # Identical
                if score_i < score_j: keep_flags[i] = False
                elif score_j < score_i: keep_flags[j] = False
                else: keep_flags[j] = False # Tie-break by original_rle_index (keep smaller)
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

            # Rule 2: IoS
            ios_val = 0.0
            if area_i <= area_j: ios_val = intersection / area_i if area_i > 0 else 0
            else: ios_val = intersection / area_j if area_j > 0 else 0
            
            if ios_val > args.ios_threshold:
                if area_i <= area_j: # i is smaller or equal
                    if score_i <= score_j: keep_flags[i] = False; handled_by_rule = True
                else: # j is smaller
                    if score_j <= score_i: keep_flags[j] = False; handled_by_rule = True
                
                if handled_by_rule:
                    if not keep_flags[i]: break
                    if not keep_flags[j]: continue
                    continue
            
            # Rule 3: IoU
            if not handled_by_rule:
                iou = intersection / (area_i + area_j - intersection)
                if iou > args.iou_threshold:
                    if score_i < score_j: keep_flags[i] = False
                    elif score_j < score_i: keep_flags[j] = False
                    else: keep_flags[j] = False # Tie-break
                    if not keep_flags[i]: break
    
    # 4. Warp and Save filtered instances
    # Create a subdirectory for this image's warped outputs
    image_specific_output_dir = os.path.join(args.output_dir, os.path.splitext(image_filename)[0])
    if not os.path.exists(image_specific_output_dir):
        os.makedirs(image_specific_output_dir)
        print(f"  Created output subdirectory: {image_specific_output_dir}")

    print(f"  Processing and saving filtered instances for {image_filename} to {image_specific_output_dir}...")
    saved_count = 0
    for k_idx in range(num_candidates):
        if keep_flags[k_idx]:
            instance_info = candidate_instances[k_idx]
            # Pass the full RLE entry dictionary to warp_perspective_from_rle_entry
            warped_region = warp_perspective_from_rle_entry(instance_info['rle_entry_dict'], original_image_cv, padding_pixels=args.padding)
            
            if warped_region is not None:
                rle_entry_content = instance_info['rle_entry_dict']
                category_id = rle_entry_content.get('category_id', 'unknown')
                score = instance_info['score']
                original_rle_idx = instance_info['original_rle_index'] # Index within this image's RLEs
                
                output_filename = f"warped_img_{image_filename}_cat{category_id}_inst{original_rle_idx}_score{score:.2f}_pad{args.padding}.png"
                output_path = os.path.join(image_specific_output_dir, output_filename)
                try:
                    cv2.imwrite(output_path, warped_region)
                    # print(f"    Saved warped region to: {output_path}")
                    saved_count += 1
                except Exception as e:
                    print(f"    Error saving warped image {output_path}: {e}")
            # else:
                # print(f"    Skipped instance original_rle_index={instance_info['original_rle_index']} for {image_filename} due to error during warping.")
    print(f"  Finished processing {image_filename}. Saved {saved_count} warped instances.")


def main():
    parser = argparse.ArgumentParser(description="Batch inference, RLE conversion, filtering, and warping of segmented regions.")
    # Model and Paths
    parser.add_argument("config_file", help="Path to the MMDetection config file.")
    parser.add_argument("checkpoint_file", help="Path to the MMDetection checkpoint file.")
    parser.add_argument("output_dir", help="Main directory to save all processed outputs.")
    
    # Input: either a directory of images or a single image file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", help="Directory containing images to process.")
    input_group.add_argument("--image_path", help="Path to a single image file to process.")

    # Processing Parameters
    parser.add_argument("--device", default='cuda:0', help="Device to use for inference (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--padding", type=int, default=10, help="Pixels to expand bounding box (default: 0).")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for overlap filtering (default: 0.5).")
    parser.add_argument("--ios_threshold", type=float, default=0.8, help="IoS threshold for covered smaller instances (default: 0.8).")
    parser.add_argument('--image_extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
                        help="List of image extensions to process when using --image_dir (default: .jpg, .jpeg, .png, .bmp, .tif, .tiff).")

    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created main output directory: {args.output_dir}")

    # Initialize MMDetection model
    print("Initializing MMDetection model...")
    try:
        model = init_detector(args.config_file, args.checkpoint_file, device=args.device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing MMDetection model: {e}")
        return

    # Determine list of images to process
    image_paths_to_process = []
    if args.image_path:
        if os.path.isfile(args.image_path):
            image_paths_to_process.append(args.image_path)
        else:
            print(f"Error: Specified image_path '{args.image_path}' does not exist or is not a file.")
            return
    elif args.image_dir:
        if os.path.isdir(args.image_dir):
            for ext in args.image_extensions:
                image_paths_to_process.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
            if not image_paths_to_process:
                print(f"No images found in directory '{args.image_dir}' with specified extensions.")
                return
            print(f"Found {len(image_paths_to_process)} images to process in '{args.image_dir}'.")
        else:
            print(f"Error: Specified image_dir '{args.image_dir}' does not exist or is not a directory.")
            return
    
    # --- Batch Processing ---
    for img_path in image_paths_to_process:
        process_single_image(img_path, model, args)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()