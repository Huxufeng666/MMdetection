import json
import os, tqdm
import random




attribute_list = [
              {"id": 1, "name": "pedestrian"},
              {"id": 2, "name": "bicycle-group"},
              {"id": 3, "name": "person-group-far-away"},
              {"id": 4, "name": "scooter-group"},
              {"id": 5, "name": "motorbike"},
              {"id": 6, "name": "bicycle"},
              {"id": 7, "name": "rider"},
              {"id": 8, "name": "motorbike-group"},
              {"id": 9, "name": "rider+vehicle-group-far-away"},
              {"id": 10, "name": "buggy-group"},
              {"id": 11, "name": "wheelchair-group"},
              {"id": 12, "name": "tricycle-group"}    

            
]





# 创建COCO数据集的JSON文件
coco_data = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [ 
              {"id": 1, "name": "pedestrian"},
              {"id": 2, "name": "bicycle-group"},
              {"id": 3, "name": "person-group-far-away"},
              {"id": 4, "name": "scooter-group"},
              {"id": 5, "name": "motorbike"},
              {"id": 6, "name": "bicycle"},
              {"id": 7, "name": "rider"},
              {"id": 8, "name": "motorbike-group"},
              {"id": 9, "name": "rider+vehicle-group-far-away"},
              {"id": 10, "name": "buggy-group"},
              {"id": 11, "name": "wheelchair-group"},
              {"id": 12, "name": "tricycle-group"}    

                    ]
                }


# Set image directory and JSON label directory
img_dir = "/workspace/data/img/val1/"
label_dir = "/workspace/data/label_val/"

k=1
for root, dirs , file in os.walk(label_dir):
    for filename in file:
        number = random.randint(1000, 9999)
        if filename.endswith(".json"):
            # 读取JSON标签文件
            with open(os.path.join(root,  filename), "r") as f:
                data = json.load(f)
            
            # 获取图像ID
            image_id = int((filename.split(".")[0]).split("_")[1])
            image_name = filename.split(".")[0]
            x = f"{number}{image_id}"
        
            # 添加图像信息
            image_info = {
                "id":  x,
                "file_name": f"{image_name}.png",
                "height": data["imageheight"],
                "width": data["imagewidth"]
            }
            coco_data["images"].append(image_info)
            
        
            # 遍历每个行人标注
            for i, bbox in enumerate(data["children"]):
                # 转换行人类别为COCO数据集的类别ID
                for a in attribute_list:
                    if a["name"]==bbox.get("identity"):
                        attribute_ids=(a["id"])
                
                # 添加行人标注信息
                annotation_info = {
                    "id": k,
                    "image_id": x,
                    "category_id": attribute_ids,
                    "area" :(bbox["x1"]-bbox["x0"])*(bbox["y1"]-bbox["y0"]),
                    "iscrowd": 0,
                    # "bbox": [bbox["x0"], bbox["y0"], bbox["x1"]-bbox["x0"], bbox["y1"]-bbox["y0"]],
                    "bbox": [bbox["x0"], bbox["y0"], bbox["x1"]-bbox["x0"], bbox["y1"]-bbox["y0"]],
                    "ignore": 0,
                    # "attributes": attribute_ids
                }
                coco_data["annotations"].append(annotation_info)

                k = k+1
            

with open("citypersons_val.json", "w") as f:
    json.dump(coco_data, f)
