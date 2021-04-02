import tensorflow as tf
import matplotlib.pyplot as plt

(x_train_all,y_train_all),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]
print(x_valid.shape,y_valid.shape)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
#读取单张图片
def show_single_img(img_arr):
    plt.imshow(img_arr,cmap="binary")
    plt.show()
#显示多张图片
def show_imgs(n_rows,n_cols,x_data,y_data):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize=(n_cols*1.4,n_rows*1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows,n_cols,index+1)
            plt.imshow(x_data[index],cmap="binary",interpolation="nearest")
            plt.axis("off")
    plt.show()
show_imgs(2,2,x_train,y_train)
#show_single_img(x_train[0])