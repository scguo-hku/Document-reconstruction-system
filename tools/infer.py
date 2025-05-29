import pickle
import numpy as np
import pycocotools.mask as mask_util
import json
# 确保从你的 MMDetection 版本正确导入 DetDataSample 和相关掩码类
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks 
# 如果你的模型可能输出 PolygonMasks 并且你想处理它们，也需要导入
# from mmdet.structures.mask import PolygonMasks
import torch # 用于 isinstance(..., torch.Tensor) 检查

def pkl_to_rle(pkl_file_path, rle_output_path):
    """
    从 pkl 文件加载分割结果 (DetDataSample)，并将掩码转换为 RLE 格式。

    Args:
        pkl_file_path (str): 输入的 pkl 文件路径。
        rle_output_path (str): 输出 RLE 结果的 JSON 文件路径。
    """
    with open(pkl_file_path, 'rb') as f:
        prediction_result = pickle.load(f)

    all_rles = []

    if isinstance(prediction_result, DetDataSample):
        pred_instances = prediction_result.pred_instances
        
        if not hasattr(pred_instances, 'masks') or pred_instances.masks is None:
            print("DetDataSample 中未找到 'masks'。")
            return

        # pred_instances.masks 通常是 BitmapMasks 对象或 torch.Tensor
        # BitmapMasks.masks 是一个 (num_instances, H, W) 的布尔 NumPy 数组
        if isinstance(pred_instances.masks, BitmapMasks):
            instance_masks_np = pred_instances.masks.masks
        elif isinstance(pred_instances.masks, torch.Tensor):
            instance_masks_np = pred_instances.masks.cpu().numpy()
        elif isinstance(pred_instances.masks, np.ndarray):
            instance_masks_np = pred_instances.masks
        else:
            print(f"无法识别的掩码类型: {type(pred_instances.masks)}")
            return

        # 如果由于某种原因 instance_masks_np 是单个掩码 (H,W)，则添加一个批次维度
        if instance_masks_np.ndim == 2:
            instance_masks_np = instance_masks_np[None, ...]

        labels = pred_instances.labels.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy() if hasattr(pred_instances, 'scores') else [None] * len(labels)
        bboxes = pred_instances.bboxes.cpu().numpy() if hasattr(pred_instances, 'bboxes') else [None] * len(labels)

        img_id_attr = getattr(prediction_result, 'img_id', None)
        if img_id_attr is None and hasattr(prediction_result, 'metainfo') and 'img_id' in prediction_result.metainfo:
            img_id_attr = prediction_result.metainfo['img_id']


        for i in range(len(instance_masks_np)):
            binary_mask_bool = instance_masks_np[i] # (H, W)
            
            # 确保是布尔掩码
            if binary_mask_bool.dtype != bool:
                binary_mask_bool = binary_mask_bool > 0 

            binary_mask_uint8 = binary_mask_bool.astype(np.uint8)
            # pycocotools.mask.encode 需要 Fortran-contiguous array
            fortran_mask = np.asfortranarray(binary_mask_uint8)
            rle = mask_util.encode(fortran_mask)
            
            # 将 RLE 中的 'counts' 字节串解码为字符串，以便 JSON 序列化
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            
            rle_entry = {
                'image_id': img_id_attr,
                'category_id': int(labels[i]),
                'segmentation': rle,
                'score': float(scores[i]) if scores[i] is not None else None,
            }

            # 将边界框转换为 COCO 格式 [x_min, y_min, width, height]
            if bboxes[i] is not None:
                x1, y1, x2, y2 = bboxes[i]
                rle_entry['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            all_rles.append(rle_entry)

    # 保留旧格式处理逻辑，以防万一，但对于新版 MMDetection 可能不太需要
    elif isinstance(prediction_result, tuple) and len(prediction_result) == 2:
        print("正在处理旧版元组格式 (bbox_results, mask_results)...")
        mask_results_legacy = prediction_result[1] 
        # 假设 mask_results_legacy 是 [类别数][实例数] 的掩码列表 (np.ndarray bool)
        for cls_id, cls_masks in enumerate(mask_results_legacy):
            if not cls_masks: continue
            for instance_mask_np_bool in cls_masks:
                if isinstance(instance_mask_np_bool, np.ndarray) and instance_mask_np_bool.dtype == bool:
                    binary_mask_uint8 = instance_mask_np_bool.astype(np.uint8)
                    fortran_mask = np.asfortranarray(binary_mask_uint8)
                    rle = mask_util.encode(fortran_mask)
                    if isinstance(rle['counts'], bytes):
                        rle['counts'] = rle['counts'].decode('utf-8')
                    all_rles.append({
                        'category_id': cls_id,
                        'segmentation': rle,
                    })
    elif isinstance(prediction_result, list) and \
         prediction_result and isinstance(prediction_result[0], list) and \
         isinstance(prediction_result[0][0], np.ndarray) and prediction_result[0][0].dtype == bool:
        print("正在处理旧版掩码列表的列表格式...")
        mask_results_legacy = prediction_result
        for cls_id, cls_masks in enumerate(mask_results_legacy):
            if not cls_masks: continue
            for instance_mask_np_bool in cls_masks:
                binary_mask_uint8 = instance_mask_np_bool.astype(np.uint8)
                fortran_mask = np.asfortranarray(binary_mask_uint8)
                rle = mask_util.encode(fortran_mask)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                all_rles.append({
                    'category_id': cls_id,
                    'segmentation': rle,
                })
    else:
        print(f"无法识别的 pkl 内容格式或内部结构: {type(prediction_result)}")
        return

    with open(rle_output_path, 'w') as f:
        json.dump(all_rles, f, indent=4)
    print(f"RLE 结果已保存到: {rle_output_path}")

if __name__ == '__main__':
    # 这些导入仅用于主脚本执行
    from mmdet.apis import init_detector, inference_detector

    # 配置文件和 checkpoint 路径
    config_file = '/userhome/cs2/scguo/mmdetection/configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py'
    checkpoint_file = '/userhome/cs2/scguo/mmdetection/output/epoch_36.pth'

    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图像路径
    img_path = '/userhome/cs2/scguo/mmdetection/small_datasets/00140.jpg'

    # 推理 (对于单张图像，返回单个 DetDataSample)
    result_datasample = inference_detector(model, img_path)

    # 保存 DetDataSample 结果为 pkl 文件
    output_pkl_path = '/userhome/cs2/scguo/mmdetection/small_datasets/result.pkl'
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(result_datasample, f)
    print(f"预测结果 (DetDataSample) 已经保存到 {output_pkl_path}")

    # 调用 pkl_to_rle 进行转换
    output_rle_json_path = '/userhome/cs2/scguo/mmdetection/small_datasets/result_rle.json'
    pkl_to_rle(output_pkl_path, output_rle_json_path)