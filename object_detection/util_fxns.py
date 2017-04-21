"""Helper functions to prepare data for faster rcnn
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.cElementTree as ET 


def bbox_coord_extractor(annotations,file_name,buff=10):
    """Extract the annotations, which originally are formatted as the top left corner (x,y) and the picture's
       height and width. Optionally enlarge the bounding box with a buffer.
       
       Args:
           annotations (dict) : dictionary of image annotations
           file_name (str) : name of the file to extract bounding boxes for
           buff (int) : optionally add additional padding around the bounding box
        
        Returns:
            bboxes (list) : list of bounding boxes for the given image with each
            bounding box in the form: [x_min, y_min, x_max, y_max]
    """
    bboxes=[]
    for a in annotations[file_name]['annotations']:
        x_min,y_min,height,width = a['x']-buff,a['y']-buff,a['height'],a['width']
        x_max = x_min+width+2*buff
        y_max = y_min+height+2*buff
        
        coords = [x_min,y_min,x_max,y_max]
                
        bboxes.append(coords)
    return bboxes

def check_bbox_in_image(bbox, img_shape):
    """Ensure that all annotations are in the image and start at pixel 1 at a minimum
    
    Args:
        bbox (list) : list of [x_min, y_min, x_max, y_max]
        img_shape (tup) : shape of image in the form: (y, x, channels)
    
    Returns:
        bbox (list) : list of cleaned [x_min, y_min, x_max, y_max]
    """
    # check to be sure all coordinates start at at least pixel 1 (required by faster rcnn)
    for i in range(len(bbox)):
        if bbox[i] <= 1.0:
            bbox[i] = 1.0
    # check that the x_max and y_max are in the photo's dimensions
    if bbox[2] >= img_shape[1]: # x_max
        bbox[2] = img_shape[1]-1
    if bbox[3] >= img_shape[0]: # y_max
        bbox[3] = img_shape[0]-1
        
    return bbox

def rotate_bound(image, angle):
    """Rotates an image a specified number of degrees and enlarges the image canvas to 
    accommodate the rotated image
    
    Args:
        image (numpy.array) : raw image matrix
        angle (int) : number of degrees to rotate the image
        
    Returns:
        rotation (numpy.array) : rotated image matrix
    
    source - https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
    """
    
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotation = cv2.warpAffine(image, M, (nW, nH))
    return rotation

def vis_ground_truth(image_path, annotations, fig_size=12, buff=10):
    """Visualize an image and it's ground truth annotations
    
    Args:
        image_path (str) : path to the image to visualize
        annotations (dict) : annotations dictionary to be used as a lookup
        fig_size (int) : size of the matplotlib figsize
        buff (int) : optionally add additional padding around the bounding box
    """
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.cv.CV_BGR2RGB)
    img_shape = im.shape
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(im, aspect='equal')

    #Draw Ground Truth
    file_name = image_path.split("/")[-1]
    bboxes = bbox_coord_extractor(annotations,file_name,buff)
    
    for bbox_true in bboxes:
        bbox_true = check_bbox_in_image(bbox_true,img_shape)
        ax.add_patch(
                plt.Rectangle((bbox_true[0], bbox_true[1]),
                              bbox_true[2] - bbox_true[0],
                              bbox_true[3] - bbox_true[1], fill=False,
                              edgecolor='white', linewidth=2)
                )
        
def vis_crop(image_path, annotations, fig_size=4, buff=10):
    """Visualize the crops from an image
    
    Args:
        image_path (str) : path to the image to visualize
        annotations (dict) : annotations dictionary to be used as a lookup
        fig_size (int) : size of the matplotlib figsize
        buff (int) : optionally add additional padding around the bounding box
    """
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.cv.CV_BGR2RGB)
    file_name = image_path.split("/")[-1]
    img_shape = im.shape
        
    # will throw exception for the no_fish case
    bboxes = bbox_coord_extractor(annotations,pic,buff)

    bbox_counter = 0
    for bbox in bboxes:
        bbox = check_bbox_in_image(bbox, img_shape)

        y = int(math.ceil(bbox[1]))
        x = int(math.ceil(bbox[0]))
        h = int(math.ceil(bbox[3] - bbox[1]))
        w = int(math.ceil(bbox[2] - bbox[0]))

        im_crop = im[y:y+h,x:x+w]
        # rotate images if they are taller than they are wide
        # reorientating images should be helpful to the classifier
        if h / float(w) > 1:
            im_crop = rotate_bound(im_crop,90)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.imshow(im_crop, aspect='equal')
        bbox_counter += 1
        plt.show()
        
# Helper functions
def saveXML(image_path=None,annotations=None,data_set_name="kaggle_fishies",save_dir=None):
    """Create an annotation in XML per image
    
    Args:
        image_path (str) : path to the image
        annotations (dict) : dictionary of image annotations
        data_set_name (str) : dataset name that faster rcnn will use
        save_dir (str) : path to the annotations folder
    """ 
    file_name = image_path.split("/")[-1]
    bbox = bbox_coord_extractor(annotations,file_name,buff=0)
    
    if bbox == []:
        return
    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)
    candidates = list()
    img_shape = img.shape
    candidates.append(img.shape)
    
    for box in bbox:
        new_bbox = check_bbox_in_image(box,img_shape)
        x = new_bbox[0]
        y = new_bbox[1]
        x2 = new_bbox[2]
        y2 = new_bbox[3]
        cls = annotations[file_name]['label']
        region = (x,y,x2,y2,cls)
        candidates.append(region)
    
    root = ET.Element("annotation")
    
    folder = ET.SubElement(root, "folder")
    folder.text = data_set_name
    
    filename = ET.SubElement(root, "filename")
    filename.text = str(file_name)
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(candidates[0][1])
    ET.SubElement(size, "height").text = str(candidates[0][0])
    ET.SubElement(size, "depth").text = str(candidates[0][2])
    
    seg = ET.SubElement(root, "segmented")
    seg.text = '0'
    
    for item in candidates[1:]:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(item[4])
        ET.SubElement(obj, "pose").text = 'Unspecified'
        ET.SubElement(obj, "truncated").text = '0'
        ET.SubElement(obj, "difficult").text = '0'
        
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(item[0])
        ET.SubElement(bbox, "ymin").text = str(item[1])
        ET.SubElement(bbox, "xmax").text = str(item[2])
        ET.SubElement(bbox, "ymax").text = str(item[3])

    tree = ET.ElementTree(root)
    file_to_write = file_name.split(".")[0] + ".xml"
    tree.write(save_dir+file_to_write)
    
