'''
TODO:
    1. Seperate thread for input
    2. Seperate thread for final processing and output display
'''


# Dependencies
import cv2
import torch

print(torch.cuda.get_device_name(0))

import matplotlib.pyplot as plt

# Download the midas model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda:0')
midas.eval()

transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transform.small_transform

# OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda:0')

    # prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        print(prediction[0])
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        # print(prediction.shape)

        output = prediction.cpu().numpy()
        # print(output)
        print(output.shape)
        
    plt.imshow(output)
    cv2.imshow('frame', frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        cap.release()
        cv2.destroyAllWindows()

plt.show()