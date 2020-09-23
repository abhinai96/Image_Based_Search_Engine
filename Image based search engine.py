#importing the necessary libraries
import numpy as np
import os
from keras.layers import Input, Dense, Conv2D,Reshape,Activation,LeakyReLU,Dropout,Flatten,Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2


# Reading the image from the google drive
img = cv2.imread(('/content/drive/My Drive/x_train/1.jpg'))
IMG_SHAPE=img.shape
IMG_SHAPE

#Appending all the images into the list
dir_path=r'/content/drive/My Drive/x_train'
lst=[]
for i in os.listdir(dir_path):
    
 
    img=cv2.imread(dir_path+'//'+i)
    print(type(img))
    print(img.shape)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #img=img.astype('float32')/255 
    print(img.shape)
    lst.append(img)
lst

#converting into numpy array
import numpy
lst = numpy.array(lst)
lst

#shape of the list
lst.shape

# center images
lst = lst.astype('float32') / 255.0 
lst

from sklearn.model_selection import train_test_split
# split
X_train, X_test = train_test_split(lst, test_size=0.3, random_state=42)

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
plt.title('sample images')

for i in range(6):
    plt.subplot(2,3,i+1)
    show_image(lst[i])

print("X shape:", lst.shape)
#print("attr shape:", attr.shape)

# try to free memory
del lst
import gc
gc.collect()

#shape of x_train
X_train.shape

#Layers of encoder and decoder
input_dim=(512,512,3)#this is our input image shape
encoder_conv_filters=[32,64,64,64]
encoder_conv_kernel_size=[3,3,3,3]
encoder_conv_strides=[1,2,2,1]

decoder_conv_t_filters=[64,64,32,1]
decoder_conv_t_kernel_size=[3,3,3,3]
decoder_conv_t_strides=[1,2,2,1]

z_dim=2

#representing the layers through for loop
encoder_input=Input(shape=input_dim,name='encoder_input')
n_layers_encoder=len(encoder_conv_filters)
x=encoder_input
for i in range(n_layers_encoder):
    conv_layer=Conv2D(
    filters=encoder_conv_filters[i]
    ,kernel_size=encoder_conv_kernel_size[i]
    ,strides=encoder_conv_strides[i]
    ,padding='same'
    ,name='encoder_conv_'+str(i)
    )
    x=conv_layer(x)
    x=LeakyReLU(name='leaky_relu'+str(i))(x)
shape_before_flattening=K.int_shape(x)[1:]
i=1 #this is merely for initialising a value to flatten layer name only
x=Flatten(name='flatten'+str(i))(x)
encoder_output=Dense(z_dim,name='encoder_output')(x)
encoder=Model(encoder_input,encoder_output)

#gives the summary of encoder
encoder.summary()

#representing the layers through for loop
decoder_input=Input(shape=(z_dim,) ,name='decoder_input')
i=1   #intitlizing a vlaue for naming the dense layer
x=Dense(np.prod(shape_before_flattening),name='dense_'+str(i))(decoder_input)

x=Reshape(shape_before_flattening,name='reshape_'+str(i))(x)
n_layers_decoder=len(decoder_conv_t_filters)
for i in range(n_layers_decoder):
    
    conv_t_layer=Conv2DTranspose(
    filters=decoder_conv_t_filters[i]
    ,kernel_size=decoder_conv_t_kernel_size[i]
    ,strides=decoder_conv_t_strides[i]
    ,padding='same'
    ,name='decoder_conv_t_'+str(i)
    )
    x=conv_t_layer(x)
    
    if i<n_layers_decoder - 1:
        x=LeakyReLU(name='leaky_relu_'+str(i))(x)
    else:
        x=Activation('sigmoid')(x)
decoder_output=x
decoder=Model(decoder_input,decoder_output)

#summary of the decoder
decoder.summary()

model_input=encoder_input
model_output=decoder(encoder_output)
model=Model(model_input,model_output)

optimizer=Adam(lr=0.0005)
def r_loss(y_true,y_pred):
    return K.mean(K.square(y_true - y_pred),axis=[1,2,3])
model.compile(optimizer=optimizer,loss=r_loss,metrics=['accuracy'])

#training the model
model.fit(
x=X_train
,y=X_train,
validation_data=[X_test, X_test]
,batch_size=30
,shuffle=True
,epochs=10
)

#representation for sample 10 images encoded and decoded images 
n_to_show=10

example_idx=np.random.choice(range(len(X_test)),n_to_show)
example_images=X_test[example_idx]


z_points=encoder.predict(example_images)

reconst_images=decoder.predict(z_points)


fig=plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4,wspace=0.4)


for i in range(n_to_show):
    img=example_images[i].squeeze()
    ax=fig.add_subplot(2,n_to_show,i+1)
    ax.axis('off')
    ax.text(0.5,-0.35,str(np.round(z_points[i],1)),fontsize=10,ha='center',transform=ax.transAxes)
    ax.imshow(img,cmap='gray_r')
    
    
for i in range(n_to_show):
    img=reconst_images[i].squeeze()
    ax=fig.add_subplot(2, n_to_show, i+n_to_show+1)
    ax.axis('off')
    ax.imshow(img, cmap='gray_r')

images = X_train
codes = encoder.predict(images) 
assert len(codes) == len(images)

#KNN for calculating the distance of the pixels
from sklearn.neighbors.unsupervised import NearestNeighbors
nei_clf = NearestNeighbors(metric="euclidean")
nei_clf.fit(codes)

def get_similar(image, n_neighbors=5):
    assert image.ndim==3,"image must be [batch,height,width,3]"

    code = encoder.predict(image[None])
    
    (distances,),(idx,) = nei_clf.kneighbors(code,n_neighbors=n_neighbors)
    
    return distances,images[idx]

def show_similar(image):
    
    distances,neighbors = get_similar(image,n_neighbors=3)
    
    plt.figure(figsize=[8,7])
    plt.subplot(1,4,1)
    show_image(image)
    plt.title("Original image")
    
    for i in range(3):
        plt.subplot(1,4,i+2)
        show_image(neighbors[i])
        plt.title("Dist=%.3f"%distances[i])
    plt.show()

    
# cherry picked smile images
show_similar(X_test[45])




























































