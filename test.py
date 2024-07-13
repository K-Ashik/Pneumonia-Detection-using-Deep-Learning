from keras_preprocessing import image 
from keras.models import load_model 
from keras.applications.vgg16 import preprocess_input 
import numpy as np 
model=load_model('pneumonia_model.keras') #Loading our model 
img=image.load_img('/Users/rising.volkan007/Downloads/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg',target_size=(224,224)) 
imagee=image.img_to_array(img) #Converting the X-Ray into pixels 
imagee=np.expand_dims(imagee, axis=0) 
img_data=preprocess_input(imagee) 
prediction=model.predict(img_data) 
if prediction[0][0]>prediction[0][1]: #Printing the prediction of model. 
	print('Person is safe.') 
else: 
	print('Person is affected with Pneumonia.') 
print(f'Predictions: {prediction}') 
