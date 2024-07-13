from keras.models import Model 
from keras.layers import Flatten,Dense 
from keras.applications.vgg16 import VGG16  #Import all the necessary modules 
import matplotlib.pyplot as plt 
from glob import glob

IMAGESHAPE = [224, 224, 3] #Provide image size as 224 x 224 this is a fixed-size for VGG16 architecture 
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False) 
#3 signifies that we are working with RGB type of images. 
training_data = '/Users/rising.volkan007/Downloads/chest_xray/train'
testing_data = '/Users/rising.volkan007/Downloads/chest_xray/test' #Give our training and testing path 


for each_layer in vgg_model.layers: 
    each_layer.trainable = False #Set the trainable as False, So that all the layers would not be trained. 
classes = glob('/Users/rising.volkan007/Downloads/chest_xray/train/*') #Finding how many classes present in our train dataset. 
flatten_layer = Flatten()(vgg_model.output) 
prediction = Dense(len(classes), activation='softmax')(flatten_layer) 
final_model = Model(inputs=vgg_model.input, outputs=prediction) #Combine the VGG output and prediction , this all together will create a model. 
final_model.summary() #Displaying the summary 
final_model.compile( #Compiling our model using adam optimizer and optimization metric as accuracy. 
  loss='categorical_crossentropy', 
  optimizer='adam', 
  metrics=['accuracy'] 
) 
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale = 1./255, #importing our dataset to keras using ImageDataGenerator in keras. 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True) 
testing_datagen = ImageDataGenerator(rescale =1. / 255) 
training_set = train_datagen.flow_from_directory('/Users/rising.volkan007/Downloads/chest_xray/train', #inserting the images. 
                                                 target_size = (224, 224), 
                                                 batch_size = 4, 
                                                 class_mode = 'categorical') 
test_set = testing_datagen.flow_from_directory('/Users/rising.volkan007/Downloads/chest_xray/test', 
                                               target_size = (224, 224), 
                                               batch_size = 4, 
                                               class_mode = 'categorical') 
fitted_model = final_model.fit( #Fitting the model. 
  training_set, 
  validation_data=test_set, 
  epochs=5, 
  steps_per_epoch=len(training_set), 
  validation_steps=len(test_set) 
) 
# Plotting the training and validation loss
plt.plot(fitted_model.history['loss'], label='training loss')
plt.plot(fitted_model.history['val_loss'], label='validation loss')
plt.legend()
plt.savefig('LossVal_loss.png')  # Save the plot first
plt.show()  # Then show the plot

# Clear the current plot to avoid overlapping plots
plt.clf()

# Plotting the training and validation accuracy
plt.plot(fitted_model.history['accuracy'], label='training accuracy')
plt.plot(fitted_model.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.savefig('AccVal_acc.png')  # Save the plot first
plt.show()  # Then show the plot

# Clear the current plot to avoid overlapping plots
plt.clf()

final_model.save('pneumonia_model.keras') #Saving the model file. 