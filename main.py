import cv2
import torch
import utils.crnn as crnn
import utils.character_set as character_set
import utils.string_utils as string_utils
import json
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def HWR_Plot(preds,image,gt, char_to_idx,idx_to_char,ground_truth, pred, cer, name="test"):
    # matplotlib.rcParams['font.sans-serif'] = "Abyssinica SIL"
    # matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['font.size'] = 25
    image = image.cpu().numpy()
    image = np.transpose(image,(1,2,0))
    print(image.shape[0])
    print(preds.shape[0])
    preds = np.exp(preds)
    gt_indicies = {0}
    for x in gt:
        try:
            gt_indicies.add(char_to_idx[x])
        except KeyError as e:
            return 7
    gt_indicies = list(gt_indicies)
    test = preds[:,gt_indicies]
    print(len(gt_indicies))
    print(test.shape)
    print(gt_indicies)
    x = np.arange(test.shape[0])  # the label locations
    fig,ax = plt.subplots(1)
    # ax[0].title.set_text('Character Error Rate: '+"{:.2f}".format(cer*10))
    print('Character Error Rate: ' + "{:.2f}".format(cer * 10))
    # ax[1].title.set_text('Character Probabilities')
    # ax[2].title.set_text('Character Probabilities w/o Blanks')
    print("GT: " + ground_truth + " Prediction: " + pred)
    # rects = []
    # ax[0].imshow(image)
    cycle = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#39FF14','#000000']

    for i in range(len(gt_indicies)):
        if i == 0:
            markerline, stemlines, baseline = ax.stem(range(0, test.shape[0] * 4, 4), test[:, i],
                                                         linefmt="C" + str(i) + "-", markerfmt="C" + str(i) + "o",
                                                         basefmt=' ', label="Blank")
            markerline.set_c(cycle[i])
            for line in stemlines:
                line.set_color(cycle[i])
        else:
            markerline, stemlines, baseline = ax.stem(range(0,test.shape[0]*4,4),test[:,i], linefmt="C0-", markerfmt="C0o", basefmt=' ', label=idx_to_char[gt_indicies[i]])
            markerline.set_c(cycle[i])
            for line in stemlines:
                line.set_color(cycle[i])
        # else:
        #     rects.append(ax[1].stem(range(0,test.shape[0]*4,4),test[:,i], linefmt="C"+str(i%10)+"--", markerfmt="C"+str(i%10)+"*", basefmt=' ', label=idx_to_char[gt_indicies[i]]))


idx_to_char, char_to_idx = character_set.load_char_set('char_set.json')




hw = crnn.create_model({
        'input_height': 64,
        'cnn_out_size': 512,
        'num_of_channels': 3,
        'num_of_outputs': len(idx_to_char) + 1
    })
state_dict=torch.load("crnn_p_3.pt",map_location=torch.device('cpu') )
hw.load_state_dict(state_dict)
dtype = torch.FloatTensor

cap = cv2.VideoCapture()

cap.open(0, cv2.CAP_DSHOW)
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
r = cv2.selectROI("select the area", frame)
cv2.destroyAllWindows()
# Crop image
cropped_image = frame[int(r[1]):int(r[1] + r[3]),
                int(r[0]):int(r[0] + r[2])]

img_shape = cropped_image.shape
percent = 64 / img_shape[0]

# img = cv2.resize(cropped_image,(int(cropped_image.shape[1]*percent),64), interpolation = cv2.INTER_AREA)

img = np.asarray(Image.fromarray(cropped_image).resize((int(cropped_image.shape[1]*percent),64), Image.ANTIALIAS))

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
blur = cv2.GaussianBlur(img2,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
img[th2==255] = 255
# cv2.imshow('test',img)
# cv2.waitKey(0)
img = img.transpose(2,0,1)
print(img.shape)
img = (img / 128.0) - 1.0
img = torch.unsqueeze(torch.tensor(img),dim=0).float()
preds = hw(img)
output_batch = preds.permute(1, 0, 2)
out = output_batch.data.cpu().numpy()
logits = out[0, ...]
pred, raw_pred = string_utils.naive_decode(logits)
pred_str = string_utils.label2str(pred, idx_to_char, False)


print(pred_str)


print(cropped_image.shape)







# Display cropped image
# cv2.imshow("Cropped image", th3)
# cv2.waitKey(0)

cv2.destroyAllWindows()