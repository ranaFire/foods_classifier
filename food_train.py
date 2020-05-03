!pip uninstall torch torchvision -y
!pip install torch==1.4.0 torchvision==0.5.0
from google.colab import drive
drive.mount('/content/gdrive/')
root_path = 'gdrive/My Drive/Colab Notebooks/'

%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt

classes = ['cake','hotdog','pizza'] # Folder Name

for cl in classes:
    folder = cl
    file = "urls_"+cl+".txt"
    path = Path(root_path+'/food')
    dest = path/folder
    dest.mkdir(parents=True, exist_ok=True)
    download_images(path/file, dest, max_pics=200)

for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_workers=8)
    
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,8))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(10)
learn.save('model',return_path=True)
learn.export('model_food.pkl')
