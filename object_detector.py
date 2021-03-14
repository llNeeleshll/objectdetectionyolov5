from yolov5processor.infer import ExecuteInference 
import cv2

def from_yolo_to_opencv(x,y,w,h):

    nx = int(x)
    ny = int(y)
    nw = int(w)
    nh = int(h)

    return nx,ny,nw,nh

def mark_prediction(pred, image, model):

    for i, det in enumerate(pred):

        print(f'Detection: {i}, {det}')   

        xmin, ymin, xmax, ymax = from_yolo_to_opencv(det['points'][0].item(),det['points'][1].item(),det['points'][2].item(),det['points'][3].item())
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255))

        text_class = model.names[int(det['class'].item())]

        cv2.putText(image, text_class, (xmin,ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    return image

def run_for_video(model):

    vid = cv2.VideoCapture(0) 

    while(True):         
        # Capture the video frame 
        # by frame 
        ret, image = vid.read() 
    
        # Display the resulting frame 
        dh, dw, _ = image.shape
        pred = model.predict(image)

        image = mark_prediction(pred, image, model)

        cv2.imshow('cvwindow', image)
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 

def run_for_image(model):

    image = cv2.imread("zidane.jpg")
    dh, dw, _ = image.shape
    pred = model.predict(image)

    image = mark_prediction(pred, image, model)

    cv2.imshow('cvwindow', image)

    if cv2.waitKey(0) == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()


img_size = 640
weights = "yolov5s.pt"

model = ExecuteInference(weight=weights, confidence=0.25, \
            img_size=img_size, agnostic_nms=False, gpu=False, iou=0.5)

run_for_image(model)




