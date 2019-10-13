'''
Definição do TP01 - Trabalho Prático INDIVIDUAL: "Check-Point" OCR - Optical Character Recognizer

Conforme discutido nas Aula 05 e 06 (Ver Wiki, Material Aulas: Aula 05 e Aula 06 sobre Machine Learning, Deep Learning e Transfer Learning) http://wiki.icmc.usp.br/index.php/Material_SSC0715_2019(fosorio) A proposta do "trabalho padrão" a ser realizado como "check-point" (TP01) da disciplina será:

    Implementar um reconhecedor de caracteres (OCR - Optical Character Recognizer)

    A técnica de aprendizado sugerida é o uso da rede, como a Inception V3 (GoogLeNet), ResNet, MobileNet, Yolo, ou outra similar, com o aprendizado de novas classes. Usaremos a técnica conhecida como: "Image Retraining" ou "Transfer Learning". <=== IMPORTANTE! Este foi o tema abordado inclusive na Aula 06: TENSORFLOW "How to Retrain Inception's Final Layer for New Categories"

        https://www.tensorflow.org/hub/tutorials/image_retraining NEW TUTORIAL 
        https://www.tensorflow.org/tutorials/images/image_recognition OLD TUTORIAL

    A base de dados para treinamento sugerida é a The Chars74K Dataset que pode ser obtida em: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
        (ou solicite uma cópia ao professor)

    Um amplo material de apoio bibliográfico está disponibilizado junto ao material da Aula 06

    O que é um OCR? https://www.newocr.com/
        Entre com uma imagem que contenha um texto e obtenha o texto extraído da imagem

Trabalhos diferentes/alternativos podem também ser propostos mas devem ser informados ao professor: descrever o que pretende fazer. Você deve enviar por e-mail para o professor e o monitor PAE a proposta deste trabalho alternativo. Ver e-mails na página principal da Wiki

As datas de entrega foram definidas no cronograma da disciplina: http://wiki.icmc.usp.br/index.php/Cronograma_SSC0715_2019(fosorio)
'''

# Ensure pip, setuptools, and wheel are up to date:
# python -m pip install --upgrade pip setuptools wheel

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image
import gdown
import tarfile
import time
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset", help="name of the dataset folder (dataset / numbers)")
ap.add_argument("-i", "--id", default=None, help="id to the tgz file on gdrive, you must pass the gdrive folder name as your dataset folder")
ap.add_argument("-e", "--epochs", type=int, default=2, help="number of training iteractions")
ap.add_argument("-p", "--plot", type=bool, nargs="?", const=True, default=False, help="bool whether plotting accuracy/loss or not")
ap.add_argument("-s", "--save", type=bool, nargs="?", const=True, default=False, help="bool whether exporting the model to default path or not")
ap.add_argument("-ep", "--exportpath", default=None, help="the path where to save the model")
args = vars(ap.parse_args())

dataset_id = "1pinqFs9jdiV9qGux_6OBwIWDpiiC4gqk" # =~ 256mb
numbers_id = "1z3uH0rlPxFS7RutbPKPCflqXYnvS7omG" # =~ 6mb

def downloadDataset():
    if not os.path.exists(args["dataset"]):
        url = "https://drive.google.com/uc?id="
        output = args["dataset"] + ".tgz"

        if args["id"] != None:
            url += args["id"]
        elif args["dataset"] == "dataset":
            url += dataset_id
        elif args["dataset"] == "numbers":
            url += numbers_id

        gdown.download(url, output, quiet=False)

        print("Extracting tarball...")
        tar = tarfile.open(output, "r:gz")
        tar.extractall()
        tar.close()

        os.remove(output)

# Class to see training progress
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

if __name__ == "__main__":
    # Download the dataset
    downloadDataset()

    # Download the classifier
    classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    IMAGE_SHAPE = (224, 224)
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
    ])

    # Get the imagenet labels from google
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    # Load the dataset
    data_root = args["dataset"]

    # Load the data into model
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

    # The resulting object is an iterator that returns image_batch, label_batch pairs
    for image_batch, label_batch in image_data:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    # Download the headless model
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

    # Create the feature extractor
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))

    # It returns a 1280-length vector for each image
    feature_batch = feature_extractor_layer(image_batch)
    print("Feature batch shape:", feature_batch.shape)

    # Freeze the variables in the feature extractor layer, so that the training only modifies the new classifier layer
    feature_extractor_layer.trainable = False

    # Attach a classification head
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(image_data.num_classes, activation='softmax')
    ])
    model.summary()
    predictions = model(image_batch)
    print("Predictions shape:", predictions.shape)

    # Train the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'])

    # See training progress
    steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
    batch_stats_callback = CollectBatchStats()

    # Use the .fit method to train the model
    history = model.fit_generator(image_data, epochs=args["epochs"], steps_per_epoch=steps_per_epoch, callbacks = [batch_stats_callback])

    if args["plot"]:
        # See the model loss plot
        #plt.figure()
        plt.subplot(2, 1, 1)
        plt.ylabel("Loss")
        #plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(batch_stats_callback.batch_losses)

        # See the model accuracy plot
        #plt.figure()
        plt.subplot(2, 1, 2)
        plt.ylabel("Accuracy")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(batch_stats_callback.batch_acc)
        plt.show()

    # To redo the plot from before, first get the ordered list of class names
    class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
    class_names = np.array([key.title() for key, value in class_names])
    print("Class names:", class_names)

    # Run the image batch through the model and convert the indices to class names
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    # Plot the result
    label_id = np.argmax(label_batch, axis=-1)
    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.imshow(image_batch[n])
        color = "green" if predicted_id[n] == label_id[n] else "red"
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
    plt.show()

    if args["save"] or args["exportpath"] != None:
        # Export the model
        t = time.time()
        if args["exportpath"] != None:
            export_path = "./saved_models/{}".format(int(t))
        else:
            export_path = args["exportpath"]
        #tf.keras.models.save_model(model, export_path)
        tf.keras.experimental.export_saved_model(model, export_path)
        print("Export path:", export_path)

        # Confirm that we can reload it, and it still gives the same results
        #reloaded = tf.keras.models.load_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})
        reloaded = tf.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})
        result_batch = model.predict(image_batch)
        reloaded_result_batch = reloaded.predict(image_batch)
        print("Reloaded result batch:", abs(reloaded_result_batch - result_batch).max())