import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import app
from tensorflow.python.saved_model import tag_constants
import cv2
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from fr_utils import *
from utils import *
from FaceRecoModel import *
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')


def main(_argv):

    #DataBase #
    database = {}

    ## Variables ####
    # input_path = 'C:/Users/Bashar Sader/Desktop/rami2.jpg' ## uncomment this to enable webcam
    input_path = '0' ## uncomment this to enable webcam
    type = 'video'
    model_path = 'C:/Users/Bashar Sader/PycharmProjects/Face_Recognition/model'
    # load model
    output = 'C:/Users/Bashar Sader/Desktop/results.avi'

    # load YOLOv4 FaceDetection Model
    FDmodel = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    infer = FDmodel.signatures['serving_default']

    # load FaceNet Model
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer='adam', loss=triplet_loss)
    load_weights_from_FaceNet(FRmodel)

    #Load Images and store their encodings in DB
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        print(identity)
        image = cv2.imread(file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## uncomment if the colors are loaded wrong
        image = cv2.resize(image,(96,96))
        # cv2.imshow("ident",image)
        # cv2.waitKey(0)
        database[identity] = img_to_encoding(image, FRmodel)



    if type == 'image':
        img2 = cv2.imread(input_path, 1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        image_name = input_path.split('/')[-1]
        image_name = image_name.split('.')[0]

        result, cropped_res = detect_face_image(image=img2, infer=infer, image_name=image_name, dont_show=True)
        for cropped in cropped_res:
                # cv2.imshow("img1", cropped)
                # cv2.waitKey(0)
                # verify(cropped, image_name, database, FRmodel)
                recognise(cropped, database, FRmodel)

    elif type == 'video':
        try:
            vid = cv2.VideoCapture(int(input_path))
        except:
            vid = cv2.VideoCapture(input_path)

        if output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0

        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_num += 1
            else:
                print('Video has ended or failed, try a different video format!')
                break
            crop_rate = 30
            result , cropped_res= detect_face_image(image = frame ,image_name='frame_'+str(frame_num), infer=infer,crop=(frame_num % crop_rate == 0),dont_show=True)

            for cropped in cropped_res:
                if frame_num % crop_rate == 0:
                     print("Frame: %.2f" % frame_num)
                     recognise(cropped, database, FRmodel)

            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
        out = None

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
