import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from clearml import Task, Logger

tf.enable_v2_behavior()
task = Task.init(project_name='MNIST',
       task_name='Tensorflow Local')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir='data/'
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()


model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_test,
)

# saving model
model.save('model.savedmodel')

from PIL import Image

# pre-processing
def preprocessing():
    INPUT_SHAPE = (28, 28)
    '''
    Return (1, 28, 28, 1) with FP32 input from image
    '''
    img = Image.open('/content/7.png').convert('L')
    img = img.resize(INPUT_SHAPE)
    imgArr = np.asarray(img) / 255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(np.float32)
    print(imgArr.shape)
    # print(imgArr)
    return imgArr




