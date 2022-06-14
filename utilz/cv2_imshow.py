import matplotlib.pyplot as plt


def cv2_imshow(image, title=''):
    plt.title(title)
    # plt.imshow(image[:, :, ::-1])
    plt.imshow(image)
    plt.show()
