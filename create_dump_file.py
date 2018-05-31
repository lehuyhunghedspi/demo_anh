
import os
import numpy as np
from scipy import ndimage
import _pickle as pickle
import pandas as pd


image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder,df_goc, min_num_images):
    df=df_goc.reset_index(drop=True)
    """Load the data for a single letter label."""


    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(df.shape[0], image_size, image_size,3),
                         dtype=np.float32)
    print(folder)
    #for image_index, image in enumerate(image_files):
    for index,row in df.iterrows():
        if(index%1000==0):
            print(index)
        image_file = os.path.join(folder, row['resizedfilename'])
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            #print(image_data.shape)
            if image_data.shape != (image_size, image_size,3):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[index, :, :,:] = image_data
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = index + 1
    dataset = dataset[0:num_images, :, : , :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

train_filename='train_dump_img'
train_groundtruth='train_dump_groundtruth'
folder='/home/hung/Desktop/model to compare/Recursive-CNNs-server_branch/resized/'
val_filename='val_dump_img'
val_groundtruth='val_dump_groundtruth'

df=pd.read_csv('/home/hung/Desktop/model to compare/Recursive-CNNs-server_branch/resized_with_resizefilename.csv')
train = df.sample(frac=0.8, random_state=200)
train_label=train[['x1_new','y1_new','x2_new','y2_new','x3_new','y3_new','x4_new','y4_new']].values

np.save(train_groundtruth,train_label)
test = df.drop(train.index)
test_label=test[['x1_new','y1_new','x2_new','y2_new','x3_new','y3_new','x4_new','y4_new']].values
np.save(val_groundtruth,test_label)

min_num_images_per_class=2
dataset = load_letter(folder, train,min_num_images_per_class)
np.save(train_filename, dataset)

min_num_images_per_class=2
dataset = load_letter(folder, test,min_num_images_per_class)
np.save(val_filename, dataset)