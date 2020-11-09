import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from PIL import Image
import cv2
import numpy as np
import random
import colorsys

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label=True, allowed_classes=['Human_face']):
    classes = allowed_classes
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        if class_name not in allowed_classes:
            continue
        else:
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if info:
                print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))

            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

            if counted_classes != None:
                height_ratio = int(image_h / 25)
                offset = 15
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    offset += height_ratio
    return image


def crop_objects(img, data, path, allowed_classes,num_obj):
    boxes, scores, classes,max_objects = data
    num_obj = min(num_obj,max_objects)
    # class_names = read_class_names("./data/classes/custom.names") #TODO : see if you can make it an arg
    class_names = ["Human_face"]
    # create dictionary to hold count of objects for image name
    counts = dict()
    cropped_images = []
    for i in range(num_obj):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name)
            # save image
            cropped_img = cv2.resize(cropped_img, (96, 96))
            cv2.imwrite(img_path, cropped_img)
            cropped_images.append(cropped_img)
        else:
            continue
    return cropped_images



def detect_face_image (image,infer,image_name = 'detect',input_size = 512,crop = True,dont_show = False,IOU = 0.45 , SCORE_THRESHOLD = 0.5):

    output_path = 'C:/Users/Bashar Sader/PycharmProjects/Face_Recognition/detections/'
    image_data = cv2.resize(image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=IOU,
        score_threshold=SCORE_THRESHOLD
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = image.shape
    bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    # pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0]]
    num_objects = 1
    allowed_classes = ['Human_face']
    cropped_images = []
    # if crop flag is enabled, crop each detection and save it as new image
    if crop:
        crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        cropped_images = crop_objects(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes, num_objects)

    image = draw_bbox(image, pred_bbox, allowed_classes=allowed_classes,)
    image = Image.fromarray(image.astype(np.uint8))

    if not dont_show:
        image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path+ 'detection.png', image)
    return image, cropped_images






def get_face(img_path,infer):
    image_name = img_path.split('/')[-1]
    image_name = image_name.split('.')[0]
    img = cv2.imread(img_path,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## uncomment if the colors are loaded wrong
    result, cropped_res = detect_face_image(image=img, infer=infer, image_name=image_name, dont_show=True)
    for cropped in cropped_res:
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey(0)
        cv2.imwrite("images/"+image_name+".png",cropped)
