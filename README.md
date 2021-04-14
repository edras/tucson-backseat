# tucson-backseat
Technical Challenge for DL on dataset from https://sviro.kl.dfki.de/data/

you will need to download and extract these datasets: 
- https://sviro.kl.dfki.de/download/hyundai-tucson-2/ 
- https://sviro.kl.dfki.de/download/hyundai-tucson-4/

you also need environment configured with: 
- tensorflow
- keras
- opencv
- scikit-learn
- numpy
- imutils

to train the network, execute command:
```
python train_model.py --dataset tucson_RGB/tucson/train/RGB --output output
```

to test the trained model, execute command:
```
python test_model.py --input tucson_RGB_wholeImage/tucson/test/RGB_wholeImage --model output/weights-028-0.0554.hd
```

To see the results and some description, check 'Challenge.pdf'.
