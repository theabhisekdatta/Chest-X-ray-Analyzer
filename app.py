
from fastai.vision.all import *

learn = load_learner('image_classification_model_vgg16.pkl')


pedict = learn.predict("val/NORMAL/NORMAL2-IM-1442-0001.jpeg")

print(pedict)
