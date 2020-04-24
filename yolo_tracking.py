import cv2
import numpy as np
import argparse
import time
import imutils
from sort import *

# Использовал SORT tracker и обученную на СОСО YOLOv3

# Проблемы
# Результат неплохой, но некоторых людей YOLO долго не видит или теряет на срок, достаточный, чтобы SORT присвоил ему новый id.
# Я попробовал поменять параметры самого SORT, и это в какой-то мере улучшило ситуацию. 
# При пересечении biunding boxes SORT присваивает неправильные id.
# Долгая обработка видео.

# Решение 
# Finetuning YOLO на датасетах с камер наблюдений, потому что у них очень специфический ракурс. 
# Также можно использовать Deep Sort алгоритм вместо SORT.  
# Для повышения производительности нужно будет перейти на cuda.

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', help="Path of video file",
					default="./TownCentreXVID.avi")
parser.add_argument('--save_path', help="Path of video file",
					default="./arseny_gorbunov_video.avi")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

def load_yolo():
	# download weights and model for COCO dataset 
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1]
					 for i in net.getUnconnectedOutLayers()]
	return net, classes, output_layers


def detect_objects(img, net, outputLayers):
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
		320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.1:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids


def draw_labels(boxes, confs, class_ids, classes, img, mot_tracker,video):
	boxes_, confs_ = [], []
	for i in range(len(class_ids)):
		if class_ids[i] == 0: 
			boxes_.append(boxes[i])
			confs_.append(confs[i])
	boxes, confs = boxes_, confs_ 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	detections = np.vstack([ boxes[i] + [confs[i]] for i in range(len(boxes)) if i in indexes ]) #[x, y, w, h]
	# [x, y, w, h] to [x1, y1, x2, y2]
	detections[:,2] = detections[:,0] + detections[:,2] # x2 = x + w
	detections[:,3] = detections[:,1] + detections[:,3] # y2 = y + h
	# add boxes to SORT
	labels = mot_tracker.update(detections)
	font = cv2.FONT_HERSHEY_PLAIN
	if len(boxes) == 0: # condition that was asked in sort paper
		mot_tracker.update() 
	for i in range(len(labels)):
		if i in indexes:
			x1, y1, x2, y2 = labels[i, :4].astype(int)
			x, y , w, h = x1, y1, x2 - x1,y2 - y1
			label = int(labels[i, 4])
			cv2.rectangle(img, (x,y), (x+w, y+h), 300, 2)
			cv2.putText(img, f'Person {label}', (x, y - 5), font, 2, 300, 2)
	# uncomment to see real time evaluation
	# cv2.imshow("Image", img)
	return img 

def start_video(video_path):
	# initialise SORT algorithm 
	mot_tracker = Sort()
	# load model 
	model, classes, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	# save video
	frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter(save_path, fourcc, 5, (frame_width, frame_height))
	# loop over frames from the video stream
	count = 0 # count number of frames 
	while True:
		ret, frame = cap.read()
		# I took every fifth frame to analyze
		if count%5 == 0: 
			height, width, channels = frame.shape
			blob, outputs = detect_objects(frame, model, output_layers)
			boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
			img = draw_labels(boxes, confs, class_ids, classes, frame, mot_tracker,video)
			video.write(cv2.resize(img, (frame_width, frame_height)))
			key = cv2.waitKey(1)
			if key == 27:
				break
		count +=1
		if count == 50:
			print(f'processed {count /int(cap.get(cv2.CAP_PROP_FRAME_COUNT))*100}% of frames')
	video.release()
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	video_path = args.video_path
	save_path = args.save_path
	if args.verbose:
		print('Opening '+video_path+" .... ")
	start_video(video_path)