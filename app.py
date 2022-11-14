import gradio as gr
from models import yolo
from cv2 import  rectangle, putText, FONT_HERSHEY_SIMPLEX, applyColorMap, COLORMAP_HSV
import numpy as np




with open("eggs.names") as f:
    classes = f.read().split("\n")

num_classes = len(classes)
palette = np.arange(0, 255, dtype=np.uint8).reshape(1, 255, 1)
palette = applyColorMap(palette, COLORMAP_HSV).squeeze(0)
np.random.shuffle(palette)

classifier = yolo("yolov4-eggs.cfg", "yolov4-eggs_best.weights", classes)

def classify(img):
    predictions = classifier.run_inference(img)
    
    for (x,y,w,h, confidence, label, classID) in predictions:
        #print(label)
        rectangle(img, (x, y), (x + w, y + h), tuple(palette[classID].tolist()), 2)
        text = "{}: {:.2f}".format(label, confidence)
        putText(img, text, (x, y - 5), FONT_HERSHEY_SIMPLEX, 0.5, tuple(palette[classID].tolist()), 1)
        # for (value,key) in zip([name, x,y,w,h, confidence, label, classID], detections.keys()):
        #     detections[key].append(value)
    return img
        
def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(classify, gr.Image(shape=(896, 684)), "image", examples = ["ex1.jpg", "ex2.jpg", "ex3.jpg", "ex4.jpg"])
iface.launch()