'''
from google.colab import drive
drive.mount('/content/gdrive/')
root_path = 'gdrive/My Drive/Colab Notebooks/food'
'''


from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
def get_food_name(img):
    model = load_learner(root_path, 'model_food.pkl')
    pred_class,pred_idx,outputs = model.predict(img)
    return pred_class
img = open_image('gdrive/My Drive/Colab Notebooks/food/pizza/00000098.png')
get_food_name(img)
