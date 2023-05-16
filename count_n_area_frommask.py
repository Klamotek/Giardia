import numpy as np
import cv2
import os
import pandas as pd

mask_dir = '.\\data\\mask'
img_dir = '.\\data\\img'
mask_path = ".\\data\\mask\\1.tif"
img_path = ".\\data\\img\\1.tif"
output_dir = '.\\data\\results_excel'
pixel_size_x20_1um = 6.8275
pixel_size_x10_1um = 3.3933


def count_measure_sort_objects(mask_path, img_path=None, threshold_down=0, threshold_up=0):
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Load the original image if specified
    if img_path:
        img = cv2.imread(img_path)
    else:
        img = None

    # Find contours of objects in the binary mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize object count, area list, and bounding boxes list
    object_count = 0
    object_areas = []
    bounding_boxes = []
    bounding_boxes_height = []
    bounding_boxes_width = []
    thresholded_areas = []
    thresholded_boxes = []

    # Loop through each contour and count objects, measure their areas, and compute their bounding boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)  # calculate area of object
        object_areas.append(area)  # add area to object areas list

        # Compute rotated bounding box of contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw bounding box on original image for visualization
        if img is not None:
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # Extract height and width from rotated rectangle
        (x, y), (w, h), angle = rect
        if w > h:
            height = w
            width = h
        else:
            height = h
            width = w

        # Add bounding box only if the object area passes the threshold
        if threshold_down <= area <= threshold_up:
            bounding_boxes.append(box)
            bounding_boxes_height.append(height)
            bounding_boxes_width.append(width)
            thresholded_areas.append(area)
            thresholded_boxes.append(box)

        # Increment object count
        object_count += 1

    # Sort object areas based on specified threshold
    objects_aft_threshold = len(thresholded_areas)
    sorted_areas = sorted(thresholded_areas)

    # calculate values
    h, w = mask.shape
    mask_area = h * w
    sum_of_objects_areas = sum(sorted_areas)
    confluency = sum_of_objects_areas / mask_area * 100

    try:
        avg_size = sum(thresholded_areas) / objects_aft_threshold
        avg_height = sum(bounding_boxes_height) / objects_aft_threshold
        avg_width = sum(bounding_boxes_width) / objects_aft_threshold
    except ZeroDivisionError:
        avg_size = 0
        avg_height = 0
        avg_width = 0


    mask_name = mask_path.split('\\')[-1]
    # Print object count, areas, and bounding boxes
    converted_thresholded_areas = []
    for area in sorted_areas:
        area = area / pixel_size_x10_1um ** 2
        area = round(area, 2)
        converted_thresholded_areas.append(area)

    # Display the original image with bounding boxes if specified
    if img is not None:
        print(f'--------------------------------------------------------'
          f'\nImage name: {mask_name}'
          f'\nNumber of objects found: {object_count}'
          f'\nObject areas [pixels]: {object_areas}'
          f'\nThreshold range: {threshold_down} - {threshold_up} '
          f'\nNumber of objects after thresholding: {objects_aft_threshold} '
          f'\nAvg. size: {(avg_size / pixel_size_x10_1um**2):.2f} um\u00b2'
          f'\nAvg. height: {(avg_height / pixel_size_x10_1um):.2f} um'
          f'\nAvg. width: {(avg_width / pixel_size_x10_1um):.2f} um'
          f'\nList of objects after thresholding [um\u00b2]: {converted_thresholded_areas}'
          f'\nConfluency: {confluency:.2f}%')

        cv2.imshow("Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return object_count, objects_aft_threshold, object_areas, thresholded_boxes, converted_thresholded_areas, confluency, avg_size, avg_height, avg_width


def process_masks(masks_folder, output_folder, img_folder=None, threshold_down=0, threshold_up=0):
    results = []
    for mask_name in os.listdir(masks_folder):
        mask_path = os.path.join(masks_folder, mask_name)
        img_path = None
        if img_folder:
            img_path = os.path.join(img_folder, mask_name)

        # Call the count_measure_sort_objects function to process each mask
        object_count, objects_aft_threshold, object_areas, thresholded_boxes, converted_thresholded_areas, confluency, avg_size, avg_height, avg_width = count_measure_sort_objects(mask_path, img_path, threshold_down, threshold_up)

        # Add the results to a list
        threshold_range = f'{threshold_down} - {threshold_up}'
        results.append({
            'Mask Name': mask_name,
            'Object found': object_count,
            'Threshold range': threshold_range,
            'Object found after thresholding': objects_aft_threshold,
            'Object Areas [pixels]': object_areas,
            'Thresholded Areas [um2]': converted_thresholded_areas,
            'Confluency [%]': confluency,
            'Avg. area [um2]': (avg_size / pixel_size_x10_1um**2),
            'Avg. Height[um2]': (avg_height / pixel_size_x10_1um),
            'Avg. Width': (avg_width / pixel_size_x10_1um)
        })
    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the results to an Excel file
    excel_path = os.path.join(output_folder, 'results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f'Done! Excel file saved in {excel_path}')
    return df


if __name__ == '__main__':

    # count_measure_sort_objects(mask_path, img_path, threshold_down=500, threshold_up=1500)
    #
    process_masks(mask_dir,
                  output_dir,
                  threshold_down=500,
                  threshold_up=1500)