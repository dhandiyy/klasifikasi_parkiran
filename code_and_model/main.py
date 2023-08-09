import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from skimage.transform import resize

#booking parkir (0: tidak dibooking dan 1: dibooking)
park1 = 0 #area parkir sebelah kiri
park2 = 0 #area parkir sebelah kanan

# Mendefinisikan model TensorFlow Hub MODEL1
model_path = './model_parkir_1.h5'
model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mendefinisikan model TensorFlow Hub MODEL2
model_path = './model_parkir_2.h5'
model2 = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Membaca mask image
mask_image_path = './mask1.png'
mask = cv2.imread(mask_image_path, 0)

# Membaca video
video_path = './Motor.mp4'
cap = cv2.VideoCapture(video_path)

def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize the frame to match model input size
    frame = img_to_array(frame)
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)  # Add a batch dimension
    return frame

def predict_image(image_path):
    img = preprocess_frame(image_path)
    predictions = model.predict(img)
    class_labels = ["Kosong", "Mobil", "Motor", "Orang"]
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_label, confidence

def model_parkir_2(image):
    img = preprocess_frame(image)
    predictions = model2.predict(img)
    class_labels = ["benar","tidak_benar"]
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_label, confidence

# Mendapatkan bounding box dari mask image
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# frame_nmr = 0
ret = True
# step = 30
# previous_frame = None
while ret:
    ret, frame = cap.read()
    if frame is not None:
        for spot in spots:
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            predicted_label, confidence = predict_image(spot_crop)

            #park1 kiri
            if spot[0] == 19:
                
                if predicted_label == "Motor" and confidence >= 0.95:
                    frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (0,0,255),2) #red

                elif predicted_label == "Mobil" and confidence >= 0.95:

                    #booking parkir
                    if park1 == 1:
                        frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (0,0,255),2) #red
                        cv2.putText(frame, 'Lahan parkir sudah dipesan', (100, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (0,255,0),2) #green
                        
                        labelnya, kemungkinan = model_parkir_2(spot_crop)

                        if kemungkinan > 0.020:
                            cv2.putText(frame, 'Parkir Tidak Benar(L)', (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, 'Parkir Benar(L)', (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                else:
                    frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (255,0,0),2) #blue

            #parkir kanan
            elif spot[0] == 665:
                
                if predicted_label == "Motor" and confidence >= 0.95:
                    frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (0,0,255),2) #red

                elif predicted_label == "Mobil" and confidence >= 0.95:
                    frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (0,255,0),2) #green

                    #booking parkir
                    if park2 == 1:
                        frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (0,0,255),2) #red
                        cv2.putText(frame, 'Lahan parkir sudah dipesan', (800, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:

                        labelnya, kemungkinan = model_parkir_2(spot_crop)

                        if kemungkinan > 0.020:
                            cv2.putText(frame, 'Parkir Tidak Benar(R)', (900, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, 'Parkir Benar(R)', (900, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1 + h), (255,0,0),2) #blue
            else:
                break

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  

cap.release()
cv2.destroyAllWindows()
