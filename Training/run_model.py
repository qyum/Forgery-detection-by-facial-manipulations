
FAST_RUN = False
#epochs=3 if FAST_RUN else 10
#epochs=5
history_1= model_1.fit(
    train_generator,
    epochs=epochs,
    callbacks=[lr_schedule],
    steps_per_epoch=total_train//batch_size,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    #callbacks=callbacks
    #use_multiprocessing=True,
)

# evaluate the model
_, train_acc = model_1.evaluate(train_generator, verbose=0)
_, test_acc = model_1.evaluate(validation_generator, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#test image.......................

import cv2
img = cv2.imread('/content/drive/My Drive/deepfake_data/Capture_3.JPG')
img = cv2.resize(img,(380,380))
img = np.reshape(img,[1,380,380,3])

classes =model_1.predict_classes(img)
print(classes)
