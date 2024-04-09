import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from detecto.utils import reverse_normalize, normalize_transform, _is_iterable
from torchvision import transforms
import numpy as np

def detect(model, label, boxes, scores, input_file, output_file, fps=30, score_filter=0.7):
    video = cv2.VideoCapture(input_file)

    # Video frame dimensions
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(frame_width)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_height)
    # Scale down frames when passing into model for faster speeds
    scaled_size = 1600
    scale_down_factor = min(frame_height, frame_width) / scaled_size

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

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

        # The transformed frame is what we'll feed into our model
        # transformed_frame = transform_frame(frame)
        transformed_frame = frame  # TODO: Issue #16
        predictions = model.predict(transformed_frame)
        print(predictions)
        # Add the top prediction of each class to the frame
        for label, box, score in zip(*predictions):
            if score < score_filter:
                continue

            # Since the predictions are for scaled down frames,
            # we need to increase the box dimensions
            # box *= scale_down_factor  # TODO Issue #16

            # Create the box around each object detected
            # Parameters: frame, (start_x, start_y), (end_x, end_y), (r, g, b), thickness
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            if label == 'occupied':
                cv2.rectangle(frame, c1, c2, (255, 0, 0), 3)
            else:
                cv2.rectangle(frame, c1, c2, (0, 255, 0), 3)

            # Write the label and score for the boxes
            # Parameters: frame, text, (start_x, start_y), font, font scale, (r, g, b), thickness
            # cv2.putText(frame, '{}: {}'.format(label, round(score.item(), 2)), (box[0], box[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

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


from detecto.core import Model
from detecto import utils, visualize

model = Model.load('Objmodel1.h5', ['occupied', 'unoccupied'])

# visualize.detect_video(model, 'video.mp4', 'output.avi')

from detecto import utils, visualize

# model = Model()

image = utils.read_image('image1.jpg')  # Helper function to read in images

labels, boxes, scores = model.predict(image)  # Get all predictions on an image
predictions = model.predict_top(image)  # Same as above, but returns only the top predictions

print(labels, boxes, scores)
print(predictions)

detect(model, labels, boxes, scores, 'video.mp4', 'output.mp4')

print('completed')