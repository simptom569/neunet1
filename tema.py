import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

import utils
import sys

model = ResNet50(weights='imagenet')
for video_path in sys.argv[1:]: 
    video = cv2.VideoCapture(video_path)

    frames = utils.get_total_frames(video_path)
    frames_point = 100
    common_top = {}
        
    print(int(frames/frames_point))
        
    for frame_number in range(0, frames, int(frames/frames_point)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        resized_frame = cv2.resize(frame, (224, 224))

        preprocessed_frame = preprocess_input(np.expand_dims(resized_frame, axis=0))

        predictions = model.predict(preprocessed_frame)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        for _, label, probability in decoded_predictions:
            # print(f"{label} - {probability}")
            if label in common_top:
                # If the label already exists in the dictionary, add the probability to the existing value
                common_top[label] += probability
            else:
                # If the label doesn't exist in the dictionary, create a new key-value pair
                common_top[label] = probability

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Сортировка словаря по значениям в порядке убывания
    sorted_common_top = sorted(common_top.items(), key=lambda x: x[1], reverse=True)
    all_weight = 0
    for label, probability in sorted_common_top[:5]:
        all_weight += probability
        # print(f"{label} - {probability}")
    
    prompt = ""
    for label, probability in sorted_common_top[:5]:
        if (probability/all_weight)*100 > 20:
            prompt += f"{label}, "
    
    print(prompt)

    video.release()

cv2.destroyAllWindows()