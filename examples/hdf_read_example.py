import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = "saved_images/images_20210120.hdf"

with h5py.File(filename, "r") as f:
    # List all groups
    group_list = list(f.keys())
    print(f"Groups: {group_list}")

    # Get the image list
    first_group = list(f.keys())[0]
    image_list = list(f[first_group])
    print(f"In group {first_group}, there are images {image_list}.")

    # get image data
    first_image = image_list[0]
    image_data = np.array(f[first_group][first_image])
    print(f"Image {first_image} is in shape {image_data.shape}.")
    plt.imshow(image_data)
    plt.show()
