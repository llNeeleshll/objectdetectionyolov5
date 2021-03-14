
## Train the model with new object to recognize

- Label your data using https://makesence.ai.
- Create a custom YAML file and place your data location in the YAML file.
- Run the below command for atleast 500 epoch to train the algo.
- You can find your trained weights at /runs/train/exp(1-n)/weights/last.pt

```bash
!python train.py --img 640 --batch 16 --epochs 3 --data custom_invoice.yaml --weights yolov5s.pt --nosave --cache
```

Once the detector is trained. Open object_detector.py and change the weights file with the newly created one. Place an image in the "imread" and run the code.