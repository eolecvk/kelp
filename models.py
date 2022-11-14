import cv2 as cv
import numpy as np
from re import search
#import os

class yolo:
    def __init__(self, cfg_path, weights_path, classes, confidence_threshold = 0.25):
        #does this matter? maybe not
        #grab width and height from config
        with open(cfg_path) as f:
            cfg = f.read()
            self.height = int(search("height[/s]*=[/s]*([0-9]*)\n", cfg).group(1))
            self.width = int(search("width[/s]*=[/s]*([0-9]*)\n", cfg).group(1))
        self.net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.classes = classes
        self.confidence_threshold = confidence_threshold

    def run_inference(self, img):
        """forward pass on a list of images

            returns: list of predictions in the format
            [(-1, 311, 198, 59, 0.8714446425437927, 'Sporophyte'), (215, 47, 199, 258, 0.8504674434661865, 'Sporophyte')] 
        """




        ln = self.net.getLayerNames()
        #I don't understand what this is doing
        #fix from https://stackoverflow.com/questions/32978575/how-to-fix-indexerror-invalid-index-to-scalar-variable
        #the original code is from
        #https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # construct a blob from the image
        
        
        blob = cv.dnn.blobFromImage(img, 1/255.0, (self.width, self.height), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []

        #this assumes all images are the same size
        h, w = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                #todo: parameterize confidence threshold
                if confidence > self.confidence_threshold:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        outputs = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                label = self.classes[classIDs[i]]
                confidence = confidences[i]
                outputs.append((x,y,w,h,confidence,label, classIDs[i]))
        return outputs
        """
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        debug = 1
        cv.imwrite("outputs\\" + imgPath.replace('frames\\',''), img)
        """

#model = yolo("network_parameters\\yolov4-eggs.cfg", "network_parameters\\yolov4-eggs_final.weights",classes)

# files = os.listdir("./frames")
# for frame in files:
#     output = model.run_inference("frames\\" + frame)
"""

vidcap = cv.VideoCapture('C:\\Users\\Adam Honts\\Documents\\Kelp\\5-11\\videos\\220520152231.mp4')
success,image = vidcap.read()
count = 0
while success:
    cv.imwrite("frames\\frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
"""