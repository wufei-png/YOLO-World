import onnxruntime as ort
import numpy as np
import cv2
from postprocess import pred_by_feat
import torch
session = ort.InferenceSession('../work_dir/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.onnx')
text='person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush'
texts = [[t.strip()] for t in text.split(',')] + [[' ']]
def read_image(image_path="demo.png"):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))  # Resize to the input dimension expected by the YOLO model
    image = image.astype(np.float32) / 255.0  # Normalize the image
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def save_image(outputs,batch_idx,image_path,output_path):
    output_image = cv2.imread(image_path)
    output_image = cv2.resize(output_image, (640, 640))  # Resize to the input dimension expected by the YOLO model
    # print("outputs[0]",outputs[0])
    # print("outputs[1]",outputs[1])
    # print("outputs[2]",outputs[2])
    # print("outputs[3]",outputs[3])
    cumulative_sum = torch.cumsum(outputs[0], dim=0)
    # print("cumulative_sum",cumulative_sum)
    cumulative_sum = torch.cat((torch.tensor([0]).to(cumulative_sum.device), cumulative_sum))
    start,end = cumulative_sum[batch_idx],cumulative_sum[batch_idx+1]
    bbox = outputs[1][start:end]
    scores = outputs[2][start:end]

    additional_info = outputs[3][start:end]
    score_threshold = 0.2

    for i, score in enumerate(scores):
        if (additional_info[i] != -1):
            x_min, y_min, x_max, y_max = bbox[i]
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))
            color = (0, 255, 0)
            cv2.rectangle(output_image, start_point, end_point, color, 1)
            label = f"Class: {texts[additional_info[i]]}, Score: {score:.2f}"
            cv2.putText(output_image, label, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    retval = cv2.imwrite(output_path, output_image)
    if not retval:
        raise Exception("Failed to save image")

image1=read_image("dog1_1024_683.jpg")
image2=read_image("dog.png")

image=np.vstack((image1,image2))
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
outputs = session.run(output_names, {input_name: image})
cls_scores=[torch.from_numpy(outputs[0][:,:80,:,:]).to('cuda:0'),torch.from_numpy(outputs[1][:,:80,:,:]).to('cuda:0'),torch.from_numpy(outputs[2][:,:80,:,:]).to('cuda:0')]
bbox_preds=[torch.from_numpy(outputs[0][:,80:,:,:]).to('cuda:0'),torch.from_numpy(outputs[1][:,80:,:,:]).to('cuda:0'),torch.from_numpy(outputs[2][:,80:,:,:]).to('cuda:0')]

nms_outputs=pred_by_feat(cls_scores=cls_scores,bbox_preds=bbox_preds)
save_image(nms_outputs,0,"dog1_1024_683.jpg","output_dog1_1024_683.png")
save_image(nms_outputs,1,"dog.png","output_dog.png")

