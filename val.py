import warnings, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
from ultralytics.utils import LOGGER
RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, BOLD, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[96m", "\033[95m", "\033[1m", "\033[0m"

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = ''
    imgsz = 416

    model = YOLO(model_path) 
    result = model.val(data='',
                        split='val',
                        imgsz=imgsz,
                        batch=16,
                        # iou=0.7,
                        # rect=False,
                        save_json=True,
                        project='val',
                        name='yolov11-a2',
                        device=os.environ.get("CUDA_VISIBLE_DEVICES", 0),
                        )

    length = result.box.p.size
    model_names = list(result.names.values())
    preprocess_time_per_image = result.speed['preprocess']
    inference_time_per_image = result.speed['inference']
    postprocess_time_per_image = result.speed['postprocess']
    all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
    
    n_l, n_p, n_g, flops = model_info(model.model, imgsz=imgsz)

    model_info_table = PrettyTable()
    model_info_table.title = "Model Info"
    model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
    model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])

    for _ in range(5):
        LOGGER.info(f'{BOLD}{ORANGE}{"-"*20}论文上的数据以以下结果为准{"-"*20}{RESET}')
    
    print(model_info_table)

    model_metrice_table = PrettyTable()
    model_metrice_table.title = "Model Metrice"
    if model.task == 'detect' or model.task == 'obb':
        model_metrice_table.field_names = ["Class Name", "Box (Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95)"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx], 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
    elif model.task == 'segment':
        model_metrice_table.field_names = ["Class Name", "Precision(Box)", "Recall(Box)", "F1-Score(Box)", "mAP50(Box)", "mAP75(Box)", "mAP50-95(Box)", 
                                           "Precision(Seg)", "Recall(Seg)", "F1-Score(Seg)", "mAP50(Seg)", "mAP75(Seg)", "mAP50-95(Seg)"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx], 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}",
                                        f"{result.seg.p[idx]:.4f}", 
                                        f"{result.seg.r[idx]:.4f}", 
                                        f"{result.seg.f1[idx]:.4f}", 
                                        f"{result.seg.ap50[idx]:.4f}", 
                                        f"{result.seg.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.seg.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
                                    f"{result.results_dict['metrics/precision(M)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(M)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(M)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(M)']:.4f}"
                                ])
    elif model.task == 'pose':
        model_metrice_table.field_names = ["Class Name", "Precision(Box)", "Recall(Box)", "F1-Score(Box)", "mAP50(Box)", "mAP75(Box)", "mAP50-95(Box)", 
                                           "Precision(Pose)", "Recall(Pose)", "F1-Score(Pose)", "mAP50(Pose)", "mAP75(Pose)", "mAP50-95(Pose)"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx], 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}",
                                        f"{result.pose.p[idx]:.4f}", 
                                        f"{result.pose.r[idx]:.4f}", 
                                        f"{result.pose.f1[idx]:.4f}", 
                                        f"{result.pose.ap50[idx]:.4f}", 
                                        f"{result.pose.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.pose.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
                                    f"{result.results_dict['metrics/precision(P)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(P)']:.4f}", 
                                    f"{np.mean(result.box.f1[:length]):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(P)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(P)']:.4f}"
                                ])
        
    print(model_metrice_table)

    with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
        f.write(str(model_info_table))
        f.write('\n')
        f.write(str(model_metrice_table))
    
    for _ in range(5):
        LOGGER.info(f'{BOLD}{ORANGE}{"-"*20}结果已保存至 {result.save_dir}/paper_data.txt...{"-"*20}{RESET}')