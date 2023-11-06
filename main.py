import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import sys


def show_image(image):
    # Display the image until 'q' is pressed
    while True:
        cv2.imshow('Image', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Close all windows
    cv2.destroyAllWindows()


def create_tangram_elements_list(image):
    # Convert the image to binary (black and white)
    binary_image = np.where(image == 0, 1, 0)

    # Count the black pixels
    black_pixel_count = np.sum(binary_image).T

    # change this according to image info
    if black_pixel_count < 60000:
        pairs = [[50, 50], [70, 60], [50, 100], [70, 80], [100, 60], [110, 60], [120, 50], [90, 90], [100, 110]]  # 55.000
    elif black_pixel_count < 70000:
        pairs = [[50, 70], [70, 70], [50, 110], [70, 90], [110, 60], [100, 80], [120, 60], [100, 100], [100, 130]]   # 65.000
    else:
        #pairs = [[70, 70], [90, 90], [50, 110], [80, 100], [110, 70], [100, 80], [120, 60], [100, 100], [100, 130]]  # 72.400
        pairs = [[100, 100], [110, 120], [80, 130], [140, 120], [130, 170]]  # 72.500

    pair_dict = {f'{x}x{y}': [] for x, y in pairs}
    return pairs, pair_dict, black_pixel_count


def combine_images(img1, img2, center):
    # Convert black image to white
    img2_white = cv2.bitwise_not(img2)

    # Get the dimensions of the first image
    height, width = img1.shape[:2]

    # Calculate the top-left corner coordinates for the second image
    start_x = center[0] - img2_white.shape[1] // 2
    start_y = center[1] - img2_white.shape[0] // 2

    # Ensure the region of interest is within bounds
    roi_start_x = max(0, start_x)
    roi_end_x = min(width, start_x + img2_white.shape[1])
    roi_start_y = max(0, start_y)
    roi_end_y = min(height, start_y + img2_white.shape[0])

    # Calculate the region of interest dimensions
    roi_width = roi_end_x - roi_start_x
    roi_height = roi_end_y - roi_start_y

    # Copy the first image to the result
    result = img1.copy()

    # Perform the bitwise OR operation between the ROI and the second image
    roi = result[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    roi = cv2.bitwise_or(roi, img2_white[:roi_height, :roi_width])

    # Update the result with the modified ROI
    result[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = roi

    # Return the combined image
    return result


def find_coordinates_of_possible_combines(image, rect_size):
    width = rect_size[0]
    height = rect_size[1]
    all_centers = []
    for x in range(0, 500-width-5, 5):
        for y in range(0, 500-height-5, 5):
            cropped_image = image[y:y+height, x:x+width]
            black_pixels_count = count_black_pixels(cropped_image)
            all_centers.append([black_pixels_count, [x+width//2, y+height//2]])
    most_matched = sorted(all_centers)[-1000:]
    liste = [x[1] for x in most_matched]
    most_matched_coordinates = cluster_coordinates(liste)
#    show_image(image)
#    for i in range(10):
#        show_image(draw_rectangles(image, {f'{rect_size[0]}x{rect_size[0]}':most_matched_coordinates[i]}))
    return most_matched_coordinates


def find_outlier(points):
    # Find the most outlier point in a list based on Euclidean distance from the centroid

    n = len(points)
    
    # Calculate the centroid of the points
    centroid_x = sum(point[0] for point in points) / n
    centroid_y = sum(point[1] for point in points) / n
    centroid = (centroid_x, centroid_y)
    
    # Find the point with the maximum distance from the centroid
    max_distance = float('-inf')
    most_outlier_point = None
    
    for point in points:
        distance = math.sqrt((centroid[0] - point[0])**2 + (centroid[1] - point[1])**2)
        if distance > max_distance:
            max_distance = distance
            most_outlier_point = point
    
    return most_outlier_point


def check_the_end_condition(image, black_pixel_count, w_h):
    s = 0
    for a in w_h:
        s+=a[0]*a[1]
    # Convert the image to binary (black and white)
    binary_image = np.where(image == 0, 1, 0)

    # Count the black pixels
    t = np.sum(binary_image)

    # if more than 0.7 of black points are filled and
    # if less than 0.05 is overcover
    # than it is success
    if (black_pixel_count-t) / black_pixel_count > 0.7 and (s-black_pixel_count-t) / s < 0.05:
        return True
    return False



def count_black_pixels(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image

    # Threshold the image to convert it into a binary image
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY)

    # Count the black pixels
    black_pixels_count = np.sum(binary_image == 0)

    return black_pixels_count


def cluster_coordinates(coordinates):
    # Convert the coordinates list to a NumPy array
    coords_array = np.array(coordinates)

    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=5, n_init=5, max_iter=300)
    kmeans.fit(coords_array)

    # Get the cluster labels and cluster centers
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Create a dictionary to store the cluster indices and their respective points
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(coords_array[i])

    # Calculate the middle point of each cluster
    cluster_midpoints = []
    for cluster_points in clusters.values():
        cluster_points = np.array(cluster_points)
        midpoint = np.mean(cluster_points, axis=0)
        temp = midpoint.tolist()
        cluster_midpoints.append([int(temp[0]), int(temp[1])])

    return cluster_midpoints


a= 1
def dfs_tree(stack, w_h, black_pixel_count, result_coordinates):
    global a
    a += 1
    image, level = stack.pop()
   # show_image(image)
    if level == len(w_h)-1:
        if check_the_end_condition(image, black_pixel_count, w_h):
            show_image(image)
            return 1
        elif stack==[]:
            return -1
        else:
            return dfs_tree(stack, w_h, black_pixel_count, result_coordinates)
    # creating black rectangle for biggest piece
    template = np.zeros((w_h[-(level+1)][0], w_h[-(level+1)][1]), dtype=np.uint8)
    # finding possible positions for it
    possible_coordinates = find_coordinates_of_possible_combines(image, [w_h[-(level+1)][0], w_h[-(level+1)][1]]) # returns list of list [[w1, h1],  [w2, h2], ...]
    temp = []
    while len(possible_coordinates) != 0:
        # select from position
        outlier = find_outlier(possible_coordinates) # returns list [w, h]
        # remove for other iterations
        possible_coordinates.remove(outlier)
        temp.append(outlier)


    for i in range(len(temp)):
        coordinate = temp.pop()
        stack.append([combine_images(image, template, coordinate), level+1])

    return dfs_tree(stack, w_h, black_pixel_count, result_coordinates)




new_limit = 10000  # Set the new recursion limit here

sys.setrecursionlimit(new_limit)

images = [11,12,13]
for i in range(len(images)):
    # Provide the path to the binary image
    image_path = f'C:\\Users\\alper\\Desktop\\Tangram\\images\\filtered_by_ratio\\{images[i]}.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    w_h, result_coordinates, black_pixel_count = create_tangram_elements_list(image)
    template = np.zeros((w_h[-1][0], w_h[-1][1]), dtype=np.uint8)

    show_image(image)
    stack = [[image, 0]]
    if dfs_tree(stack, w_h, black_pixel_count, result_coordinates) == 1:
        print(f'Success, recursion complexity: {a}')
    else:
        print(f'Success, recursion complexity: {a}')











######
"""


def draw_rectangles(image, rectangles):
    # Create a copy of the image to draw rectangles on
    image_copy = image.copy()

    # Iterate over the rectangles in the dictionary
    for size, coordinates in rectangles.items():
        # Extract the width and height from the size string
        width, height = map(int, size.split('x'))

        # Extract the x and y coordinates from the coordinates list
        x, y = coordinates

        # Calculate the top-left and bottom-right points of the rectangle
        top_left = (x, y)
        bottom_right = (x + width, y + height)

        # Generate a random color for the rectangle (BGR format)
        color = np.random.randint(0, 256, size=(3,)).tolist()

        # Draw the rectangle on the image
        cv2.rectangle(image_copy, top_left, bottom_right, color, -1)

    return image_copy


    # dfs(image, w_h, result_coordinates, black_pixel_count, {})
    # print(result_coordinates)
    # image_result = draw_rectangles(image, result_coordinates)
    # show_image(image_result)

def dfs(image, w_h, result_coordinates, black_pixel_count, image_dicti):
    if f'{9-len(w_h)}' not in image_dicti:
        image_dicti[f'{9-len(w_h)}'] = image
    if f'{9-len(w_h)+1}' in image_dicti:
        del image_dicti[f'{9-len(w_h)+1}']
    # image is current image and every time one of w_h last is combined w_h pops the last
    # to faster results we choose the most outlier coordinate every time and after selection we remove from dicti
    # result_coordinates stores the current coordinates on image if the image is valid then returns this dictionary
    # end condition
    if len(w_h) == 0:
        if check_the_end_condition(image_dicti[f'{9-len(w_h)}'], black_pixel_count):
            show_image(image_dicti[f'{9-len(w_h)}'])
            print(result_coordinates)
            image_result = draw_rectangles(image_dicti[f'{9-len(w_h)}'], result_coordinates)
            show_image(image_result)
            return result_coordinates
        else:
            return -1
    # creating black rectangle for biggest piece
    template = np.zeros((w_h[-1][0], w_h[-1][1]), dtype=np.uint8)
    # finding possible positions for it
    possible_coordinates = find_coordinates_of_possible_combines(image_dicti[f'{9-len(w_h)}'], [w_h[-1][0], w_h[-1][1]]) # returns list of list [[w1, h1],  [w2, h2], ...]
    # define key to reach result dicti
    key = f'{w_h[-1][0]}x{w_h[-1][1]}'
    # pop
    w_h.pop()
    while len(possible_coordinates) != 0:
        if f'{9-len(w_h)}' in image_dicti:
            del image_dicti[f'{9-len(w_h)}']
        # select from position
        outlier = find_outlier(possible_coordinates) # returns list [w, h]
        # remove for other iterations
        possible_coordinates.remove(outlier)
        # combine the largest with the outlier position
        image = combine_images(image_dicti[f'{9-len(w_h)-1}'], template, outlier)
        # store in result dicti
        result_coordinates[key] = outlier
        # recur
        t = dfs(image, w_h, result_coordinates, black_pixel_count, image_dicti)
        if t != -1:
            return t
    return -1



# write a combine function that takes coordinate and images
# read image, create images dicti (sorted by area) send to function
# find coordinates save them in another temp image
# create queue
# combine
# show combined image
# recursive


"""