"""
image inference - run model inference on single frame and output results
"""
from models import yolo
from cv2 import imread, rectangle, putText, imwrite, FONT_HERSHEY_SIMPLEX
import argparse
from os.path import isdir, basename, exists
from os import scandir, mkdir
from time import time
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument("cfg_path", help=".cfg file for model", nargs = "?", default=  "network_parameters/yolov4-eggs.cfg")
parser.add_argument("weights_path", help=".weights file for model", nargs = "?", default = "network_parameters/yolov4-eggs_best.weights")
parser.add_argument("names_path", help=".names file for model", nargs = "?", default="network_parameters/eggs.names")
parser.add_argument("image_path", help = "image(s) to run inference on", nargs ="?", default= "images")

# "network_parameters/tiny/yolov4-tiny-eggs.cfg",
#                 "network_parameters/tiny/yolov4-tiny-eggs_best.weights",
#                 "network_parameters/eggs.names",
#                 "E:/young sporophytes/220519153005.mp4"
args = parser.parse_args()



names_path = args.names_path
weights_path = args.weights_path
cfg_path = args.cfg_path

image_path = args.image_path

with open(names_path) as f:
    classes = f.read().split("\n")
    #print(classes)

if not exists("predictions"):
    mkdir("predictions")

print("Classes: ", classes )
#build a list of paths to process
if isdir(image_path):
    image_paths = [path.path for path in scandir(image_path) if path.name.endswith(".jpg")]
else:
    image_paths = [image_path]

#store detections
detections = {"name":[], "x": [], "y":[], "w":[], "h":[], "confidence":[], "class":[], "classID":[]}
model = yolo(cfg_path, weights_path,classes)
#run inference on each path
for image_path in image_paths:
    #grab the name of the image before the extension
    name = basename(image_path).split(".")[0]
    img = imread(image_path)
    start = time()
    
    predictions = model.run_inference(img)
    end = time()

    print(predictions, '\n', 'total inference time: ', end - start)

    for (x,y,w,h, confidence, label, classID) in predictions:
        rectangle(img, (x, y), (x + w, y + h), 0, 2)
        text = "{}: {:.4f}".format(label, confidence)
        putText(img, text, (x, y - 5), FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        for (value,key) in zip([name, x,y,w,h, confidence, label, classID], detections.keys()):
            detections[key].append(value)
    imwrite("predictions/" + name + " prediction.jpg", img)
DataFrame(detections).to_csv("predictions/detections.csv")
