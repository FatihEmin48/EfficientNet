# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:56:44 2024

@author: fatih
"""

import cv2
import numpy as np
from keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

model = EfficientNetB0(weights = "imagenet")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    x = cv2.resize(frame, (224, 224))
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    
    predictions = model.predict(x)
    
    label = decode_predictions(predictions, top = 1)[0][0]
    
    label_name, label_cinfidence = label[1], label[2]
    
    cv2.putText(frame, f'{label_name}({label_cinfidence*100:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("EfficientNet", frame)
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()