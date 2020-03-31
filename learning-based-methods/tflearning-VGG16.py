from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

# ************************* # Load the pre-trained model # ************************* #
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# ************************* # Extract Features # ************************* #
train_dir = './clean-dataset/train'
validation_dir = './clean-dataset/validation'

nTrain = 600
nVal = 150

# Generate batches of images and labels
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain, 3))

train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size,
                                              class_mode='categorical', shuffle=shuffle)

# Pass image through the network and give us a 7 x 7 x 512 dimensional tensor
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

# ************************* # Create your own model # ************************* #
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# ************************* # Train the model # ************************* #
model.compile(optimizer=optimizers.RMSprop(lr=2e-4), loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(train_features, train_labels, epochs=20, batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

# ************************* # Check performance # ************************* #
# Visualize images that where wrongly classified
fnames = validation_generator.filenames

ground_truth = validation_generator.classes

label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.iteritems())

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))

# Let us see which images were predicted wrongly
for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]

    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))

    original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
    plt.imshow(original)
    plt.show()
