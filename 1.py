# from mmdet.apis import init_detector, inference_detector

# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# device = 'cuda:0'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# inference_detector(model, 'demo/demo.jpg')


# # import json
# # import os

# # # 载入CityPersons的标签文件
# # citypersons_ann_file = '/workspace/data/labels/annotations/instances_test.json'
# # with open(citypersons_ann_file, 'r') as f:
# #     citypersons_ann = json.load(f)

# # # 创建一个COCO数据集的字典
# # coco_dataset = {
# #     "info": {},
# #     "licenses": [],
# #     "categories": [
# #         {
# #             "id": 1,
# #             "name": "pedestrian",
# #             "supercategory": "person"
# #         }
# #     ],
# #     "images": [],
# #     "annotations": []
# # }

# # # 将CityPersons中的图像信息转换为COCO数据集中的图像信息
# # for img in citypersons_ann['images']:
# #     coco_dataset['images'].append({
# #         'id': img['id'],
# #         'width': img['width'],
# #         'height': img['height'],
# #         'file_name': img['file_name'],
# #         'license': 0,
# #         'flickr_url': '',
# #         'coco_url': '',
# #         'date_captured': ''
# #     })

# # # 将CityPersons中的标注信息转换为COCO数据集中的标注信息
# # for ann in citypersons_ann['annotations']:
# #     coco_dataset['annotations'].append({
# #         'id': ann['id'],
# #         'image_id': ann['image_id'],
# #         'category_id': 1,
# #         'segmentation': [],
# #         'area': ann['bbox'][2] * ann['bbox'][3],
# #         'bbox': ann['bbox'],
# #         'iscrowd': 0
# #     })

# # # 保存COCO数据集的标签文件
# # coco_ann_file = '/workspace/data/labels/annotations/instances_test.json'
# # with open(coco_ann_file, 'w') as f:
# #     json.dump(coco_dataset, f)

# import os
# import shutil

# # 定义源文件夹和目标文件夹的路径
# source_folder = "/workspace/data/img/val"
# target_folder = "/workspace/data/ALL_img"

# # 确保目标文件夹存在，如果不存在就创建
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# # 遍历源文件夹下的所有文件和子文件夹
# for root, dirs, files in os.walk(source_folder):
#     # 遍历当前文件夹下的所有文件
#     for file in files:
#         # 如果文件是图片文件，则复制到目标文件夹
#         if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".bmp"):
#             src_path = os.path.join(root, file)  # 原文件路径
#             dst_path = os.path.join(target_folder, file)  # 目标文件路径
#             shutil.copyfile(src_path, dst_path)  # 复制文件


import os
 
from mmdet.apis import init_detector, inference_detector
 
config_file = '/workspace/work_dirs/fcos/fcos.py'
checkpoint_file = '/workspace/work_dirs/epoch_9.pth'
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
imgPath = '/workspace/data/img/test/toulouse/toulouse_00822.png'

model.show_result(
        imgPath,
        inference_detector(model, imgPath),
        bbox_color=(255, 0, 0),
        text_color=(200, 200, 200),out_file='/workspace/result/result10.jpg')





# faster RCNN, cascade RCNN, fcos, retinaNet, deformable detr

'''
/workspace/data/img/test/bologna/bologna_01431.png
/workspace/data/img/test/bologna/bologna_02034.png
/workspace/data/img/test/toulouse/toulouse_00811.png
/workspace/data/img/test/ulm/ulm_00739.png
/workspace/data/img/test/ulm/ulm_00742.png
/workspace/data/img/test/toulouse/toulouse_01142.png
/workspace/data/img/test/zagreb/zagreb_00873.png
/workspace/data/img/test/roma/roma_01346.png
/workspace/data/img/test/bologna/bologna_01433.png
/workspace/data/img/test/berlin/berlin_00473.png



python ./tools/test.py /workspace/work_dirs/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco.py /workspace/work_dirs/cascade_rcnn_r50_fpn_1x_coco/latest.pth --out=result.pkl

python tools/analysis_tools/confusion_matrix.py \
    /workspace/work_dirs/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco.py\
    /workspace/result.pkl \
    /workspace/work_dirs \
    --show
'''
