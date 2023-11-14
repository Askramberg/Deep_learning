from PIL import Image

def crop_image(image_path, crop_origin, crop_size=(128, 128)):
    with Image.open(image_path) as img:
        cropped_image = img.crop((crop_origin[0], crop_origin[1], crop_origin[0] + crop_size[0], crop_origin[1] + crop_size[1]))
        return cropped_image

# Set the crop origin (x, y)
crop_origin = (0, 0)  # Adjust as needed

# Loop through the images
for i in range(1, 501):  # From 1 to 500
    # Formatting the file name to match the naming convention
    data_file_number = str(i).zfill(4)
    label_file_number = str(i).zfill(3)

    data_image_path = f"data/SOCprist{data_file_number}.tiff"
    label_image_path = f"labels/slice__{label_file_number}.tif"
    

    cropped_data_image = crop_image(data_image_path, crop_origin)
    cropped_label_image = crop_image(label_image_path, crop_origin)

    # Save the cropped images
    cropped_data_image.save(f"data_crop/cropped_data_{data_file_number}.tiff")
    cropped_label_image.save(f"label_crop/cropped_label_{label_file_number}.tiff")
