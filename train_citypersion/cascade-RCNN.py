# faster RCNN, cascade RCNN, fcos, retinaNet, deformable detr)

# inherits a base config
_base_ = ["/workspace/configs/_base_/models/cascade_rcnn_r50_fpn.py"] 


#change the num_classes in head to match the dataset's annnotation

model = dict(
    roi_head=dict(
    bbox_head = dict(num_classes=12)
    )
)

#modeify dataset releated settings

dataset_type = 'CocoDataset'

classes = ('pedestrian', 'bicycle-group', 'person-group-far-away', 'scooter-group', 
            'motorbike', 'bicycle', 'rider', 'motorbike-group', 'rider+vehicle-group-far-away', 
                'buggy-group', 'wheelchair-group', 'tricycle-group' )

data = dict(
    train=dict(   
        img_prefix='/workspace/data/ALL_img/',
        classes=classes,
        ann_file='/workspace/data/labels/citypersons_train.json'),
    val=dict( 
        img_prefix='/workspace/data/ALL_img/',
        classes=classes,
        ann_file='/workspace/data/labels/citypersons_val.json'),
    test=dict(
        img_prefix='/workspace/data/ALL_img/',
        classes=classes,
        ann_file='/workspace/data/labels/citypersons_val.json')
        )

# use the [re-trained faster-rcnn model to obtain higher pwedormacne]

load_form ="/workspace/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"


# python tools/train.py /workspace/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --gpus 1

'''
python demo/image_demo.py /workspace/data/img/test/milano/milano_01527.png /workspace/work_dirs/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco.py  /workspace/work_dirs/cascade_rcnn_r50_fpn_1x_coco/epoch_6.pth 

'''