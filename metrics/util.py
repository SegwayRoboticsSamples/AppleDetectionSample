from tensorflow import keras
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visiualize_annotations(image_data, boxes):
    img_height, img_width, _ = image_data.shape
    image = Image.fromarray((image_data*255).astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf', 34)
    for box in boxes:
        if box[0] < 0:
            continue
        
        ymin = int(box[0]  * img_height) #y
        xmin = int(box[1] * img_width) #x
        ymax = int(box[2] * img_height)  #y+h
        xmax = int(box[3] * img_width) #x+w
        
        draw.line((xmin, ymin, xmax, ymin), fill=128, width=6)
        draw.line((xmin, ymax, xmax, ymax), fill=128, width=6)
        draw.line((xmin, ymin, xmin, ymax), fill=128, width=6)
        draw.line((xmax, ymin, xmax, ymax), fill=128, width=6)
    return np.array(image).astype(np.uint8)


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)
