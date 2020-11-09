total_train =train_df.shape[0]
print(total_train)
total_validate =validate_df.shape[0]
print(total_validate)

batch_size=5
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
     
     directory="/content/drive/My Drive/deepfake_data/train_real_fake_1/train_real_fake_img/",
     dataframe= train_df,
     #label_mode="int",
     x_col='ori_id',
     y_col='label',
     target_size=IMAGE_SIZE,
     class_mode='categorical',
     #class_mode='binary',
     batch_size=batch_size,
     shuffle=True
)
#print(train_generator)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
     
    directory="/content/drive/My Drive/deepfake_data/train_real_fake_1/train_real_fake_img/", 
    dataframe=validate_df,
     
    x_col='ori_id',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    #class_mode='binary',
    batch_size=batch_size,
    shuffle=True
)

example_df = train_ori_label.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/content/drive/My Drive/deepfake_data/train_real_fake_1/train_real_fake_img/", 
    x_col='ori_id',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
    #class_mode='binary'
)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

WARMUP = 5
epochs=15
LR = 0.00004


def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=epochs)
