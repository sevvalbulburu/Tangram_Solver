# Tangram_Solver_DFS

## Project Explanation
[Pdf explanation](Report.pdf)

## Install dependencies

- Install Python 3.10.11

-	Install libraries by typing
```
$ pip3 install -r requirements.txt
```

## Code Explanation

### Image Format Manipulation with format_images.py

The `format_images.py` script offers a collection of functions to manipulate and analyze image files using the Python Imaging Library (PIL). Here's a quick overview of each function's role:

- `convert_images_to_png(folder_path)`: Converts JPEG and PNG images within the provided folder path to the PNG format. A new "converted" directory is created to save the converted images.

- `calculate_image_info(folder_path)`: Computes various information about PNG images in the specified folder. This includes metrics like width, height, total pixel count, white pixel count, black pixel count, and the black-to-total pixel ratio. The information is printed for each image.

- `resize_images(folder_path, output_folder, new_size)`: Resizes PNG images in the indicated folder to a new size while preserving the original color mode. The resized images are stored in the output folder.

- `save_images_within_ratio(folder_path, save_path)`: Uses image information to selectively save PNG images that meet a predefined criterion for the black-to-total pixel ratio (ranging from 20% to 30%). The selected images are stored in the provided save path.

The code is accompanied by informative comments that guide users on how to employ these functions. By uncommenting relevant sections and providing the required folder paths, users can execute their desired image operations.

---

### Tangram Puzzle Solver with main.py

The `main.py` script is designed to solve tangram puzzles using a specialized depth-first search approach. Here's a concise breakdown of its functionalities:

1. **Importing Libraries**: The script imports essential libraries including OpenCV (cv2), NumPy (np), the math module, the KMeans class from sklearn.cluster, and the sys module.

2. **Function Definitions**: A series of functions are defined to handle different aspects of the puzzle-solving process. These functions manage tasks such as image display, element creation, image combination, potential position identification, outlier detection, solution validation, pixel counting, clustering, and depth-first search.

3. **Recursion Limit Setting**: To accommodate potential deep recursion, the script increases the recursion limit using `sys.setrecursionlimit(new_limit)`.

4. **Image Processing**: The script processes a list of image indices (images). For each image, it loads the image, creates a list of tangram element sizes (w_h), initializes relevant variables, and displays the original image. The `dfs_tree` function is then invoked to explore and evaluate possible combinations of tangram elements to solve the puzzle. If a successful solution is found, the recursion complexity is printed; otherwise, it indicates an unsuccessful attempt.

This code's goal is to decipher tangram puzzles by identifying valid combinations of tangram elements and their optimal positions within the provided images. It employs iterative placement strategies, utilizing image processing techniques and a depth-first search methodology to navigate potential solutions.


### 🙌 All together
📽️ Refer this video for watching whole simulation on
<a href="https://youtu.be/orxbHXTbhis" target="_blank">YouTube.</a>

### Collaboration
Collaborated with [Alperen Ölçer](https://github.com/Alperenlcr)
