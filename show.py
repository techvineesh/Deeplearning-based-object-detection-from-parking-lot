import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from detecto.utils import reverse_normalize, normalize_transform, _is_iterable
from torchvision import transforms
import numpy as np

def detect(model, input_file, output_file, fps=30, score_filter=0.7):
    video = cv2.VideoCapture(input_file)
    print('I m work')
    # Video frame dimensions
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(frame_width)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_height)
    # Scale down frames when passing into model for faster speeds
    scaled_size = 1600
    scale_down_factor = min(frame_height, frame_width) / scaled_size

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

    # Transform to apply on individual frames of the video
    transform_frame = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        normalize_transform(),
    ])

    # Loop through every frame of the video
    while True:
        ret, frame = video.read()
        # Stop the loop when we're done with the video
        if not ret:
            break

        transformed_frame = frame  # TODO: Issue #16
        predictions = model.predict(transformed_frame)
        # print(predictions)
        # Add the top prediction of each class to the frame
        for label, box, score in zip(*predictions):
            if score < score_filter:
                continue

            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            if label == 'occupied':
                cv2.rectangle(frame, c1, c2, (255, 0, 0), 3)
            else:
                cv2.rectangle(frame, c1, c2, (0, 255, 0), 3)


        # Write this frame to our video file
        out.write(frame)

        # If the 'q' key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # When finished, release the video capture and writer objects
    video.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()

#
# from detecto.core import Model
# model = Model.load('Objmodel1.h5', ['occupied', 'unoccupied'])
# detect(model,'video.mp4', 'output.avi')

# print('completed')