import os
from PIL import Image

def convert_images_to_png(folder_path):
    # Create a new folder to save the converted images
    output_folder = folder_path + '/converted'
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all the files in the folder
    files = os.listdir(folder_path)

    # Enumerate through the files
    for i, file_name in enumerate(files):
        # Get the file extension
        _, extension = os.path.splitext(file_name)
        extension = extension.lower()

        # Check if the file is a JPEG or PNG image
        if extension == '.jpg' or extension == '.jpeg' or extension == '.png':
            # Open the image file
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            # Convert the image to PNG format
            converted_image = image.convert('RGBA')
            new_file_name = f'{i}.png'
            output_path = os.path.join(output_folder, new_file_name)
            converted_image.save(output_path, 'PNG')

            # Close the image file
            image.close()


def calculate_image_info(folder_path):
    # Get a list of all the PNG files in the folder
    file_list = [file_name for file_name in os.listdir(folder_path) if file_name.lower().endswith('.png')]

    # Enumerate through the PNG files
    for i, file_name in enumerate(file_list):
        image_path = os.path.join(folder_path, file_name)

        # Open the image file
        image = Image.open(image_path)

        # Calculate image information
        width, height = image.size
        white_pixel_count = 0
        black_pixel_count = 0

        for pixel in image.getdata():
            if pixel == (0, 0, 0, 255):  # Assuming black pixels have RGBA value of (0, 0, 0, 255)
                black_pixel_count += 1
            elif pixel == (255, 255, 255, 255):  # Assuming white pixels have RGBA value of (255, 255, 255, 255)
                white_pixel_count += 1

        total_pixel_count = width * height
        black_to_total_ratio = black_pixel_count / total_pixel_count

        # Print image information
        print(f"{file_name} - {total_pixel_count==black_pixel_count+white_pixel_count} - W:{width:4d} - H:{height:4d} - W:{white_pixel_count:8d} - B:{black_pixel_count:8d} - A:{total_pixel_count:9d} - {black_to_total_ratio:.2%}")

        # Close the image file
        image.close()


def resize_images(folder_path, output_folder, new_size):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all the files in the folder
    files = os.listdir(folder_path)

    # Enumerate through the files
    for i, file_name in enumerate(files):
        # Get the file extension
        _, extension = os.path.splitext(file_name)
        extension = extension.lower()

        # Check if the file is a PNG image
        if extension == '.png':
            # Open the image file
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            # Resize the image while maintaining original color mode
            resized_image = image.resize(new_size, resample=Image.NEAREST)

            # Save the resized image
            output_path = os.path.join(output_folder, file_name)
            resized_image.save(output_path)

            # Close the image files
            image.close()
            resized_image.close()


def save_images_within_ratio(folder_path, save_path):
    # Create the save folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Get a list of all the PNG files in the folder
    file_list = [file_name for file_name in os.listdir(folder_path) if file_name.lower().endswith('.png')]

    # Enumerate through the PNG files
    i = 1
    for file_name in file_list:
        image_path = os.path.join(folder_path, file_name)

        # Open the image file
        image = Image.open(image_path)

        # Calculate image information
        width, height = image.size
        white_pixel_count = 0
        black_pixel_count = 0

        for pixel in image.getdata():
            if pixel == (0, 0, 0, 255):  # Assuming black pixels have RGBA value of (0, 0, 0, 255)
                black_pixel_count += 1
            elif pixel == (255, 255, 255, 255):  # Assuming white pixels have RGBA value of (255, 255, 255, 255)
                white_pixel_count += 1

        total_pixel_count = width * height
        black_to_total_ratio = black_pixel_count / total_pixel_count

        # Print image information
        print(f"{file_name} - {total_pixel_count == black_pixel_count + white_pixel_count} - W:{width:4d} - H:{height:4d} - W:{white_pixel_count:8d} - B:{black_pixel_count:8d} - A:{total_pixel_count:9d} - {black_to_total_ratio:.2%}")

        # Save the image if the black-to-total ratio is between 20% and 30%
        if 0.2 <= black_to_total_ratio <= 0.3:
            new_file_name = f'{i}.png'
            output_path = os.path.join(save_path, new_file_name)
            image.save(output_path)
            i += 1

        # Close the image file
        image.close()

##############
# Provide the folder path where the PNG images are located
# input_folder = 'C:\\Users\\alper\\Desktop\\Tangram\\images\\resized'

# Provide the save folder path to save the images within the specified ratio
# save_folder = 'C:\\Users\\alper\\Desktop\\Tangram\\images\\filtered_by_ratio'

# save_images_within_ratio(input_folder, save_folder)

##############
# Provide the folder path where the PNG images are located
# input_folder = 'C:\\Users\\alper\\Desktop\\Tangram\\images\\resized'

# Provide the output folder path to save the resized images
# output_folder = 'C:\\Users\\alper\\Desktop\\Tangram\\images\\filtered_by_ratio'

# Provide the desired new size (width, height) for the images
# new_size = (500, 500)

# resize_images(input_folder, output_folder, new_size)

##############
calculate_image_info('C:\\Users\\alper\\Desktop\\Tangram\\images\\filtered_by_ratio')

##############
# convert_images_to_png('C:\\Users\\alper\\Desktop\\Tangram\\images')


# a = [31.26, 34.82, 28.01, 21.73, 24.62, 18.81, 31.60, 24.90, 34.22, 28.00, 31.74, 11.41, 38.95, 24.57, 44.86, 31.76, 24.01, 14.71, 30.58, 34.79, 29.69, 28.62, 18.25, 20.67, 7.74, 22.87, 17.24, 19.31, 22.83, 26.74, 31.52, 15.82, 25.90, 49.90, 32.56, 17.41, 36.91, 11.15, 11.53, 17.41, 27.47, 11.15, 28.84, 11.53, 28.80, 28.80, 28.80, 28.80, 22.84]