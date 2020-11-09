
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
