
!pip install -q efficientnet
import efficientnet.tfkeras as efn

LABEL_SMOOTHING = 0.05
def build_model_1(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS):
    model = tf.keras.Sequential([
        efn.EfficientNetB7(input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS),weights='imagenet',pooling='avg',include_top=False),
        Dense(2, activation='softmax')
        ])
    
    model.compile(optimizer='adam',
                  loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = LABEL_SMOOTHING),
                  #loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
     
    #model.add(Dense(2, activation='softmax'))  

   # model.compile(loss='categorical_crossentropy',
                  #optimizer='rmsprop',
                  #metrics=[tf.keras.metrics.AUC(name='auc')])
    return model 
model_1=build_model_1(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS)
model_1.summary()

from keras.utils import plot_model
plot_model(model_1, to_file='model.png')
