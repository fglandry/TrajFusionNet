import cv2
import numpy as np

from utils.utils import show_image

def process_optical_flow_v2(img_data, target_dim, prev_img_list, feature_type):
    if prev_img_list:
        if ('flow_optical_v3' in feature_type) or ('flow_optical_v5' in feature_type) \
            or ('flow_optical_v6' in feature_type):
            if len(prev_img_list) >= 3:
                prev_img_data = prev_img_list[-2] # calculate optical flow with image from 2 timesteps in the past
            else:
                prev_img_data = None
        else:
            prev_img_data = prev_img_list[-1]
    if prev_img_data is not None:
        prev_img_data_resized = cv2.resize(prev_img_data, target_dim)
        img_data_resized = cv2.resize(img_data, target_dim)
        opt_flow_features = get_optical_flow_v2(prev_img_data_resized, img_data_resized, debug=False)
        if opt_flow_features is not None:
            if 'flow_optical_v2' in feature_type:
                img_features[..., 1] = opt_flow_features[1]
            elif 'flow_optical_v3' in feature_type:
                img_features[..., 2] = opt_flow_features[0]
            elif 'flow_optical_v4' in feature_type:
                img_features = np.append(img_features, np.expand_dims(opt_flow_features[0], 2), axis=2)
            else:
                img_features = np.append(img_features, np.expand_dims(opt_flow_features[0], 2), axis=2) # magnitude
                img_features = np.append(img_features, np.expand_dims(opt_flow_features[1], 2), axis=2) # angle
        #show_image(img_features)
    return img_features

def get_optical_flow_v2(prev_img_data, img_data, debug=False):

    if not isinstance(prev_img_data, np.ndarray):
        return None
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_img_data, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    normalized_magniture = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized_angle = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

    if debug:
        # Creates an image filled with zero 
        # intensities with the same dimensions  
        # as the frame 
        mask = np.zeros_like(img_data) 
        
        # Sets image saturation to maximum 
        mask[..., 1] = 255

        # Sets image hue according to the optical flow  
        # direction 
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        mask[..., 2] = normalized_magniture
        
        # Converts HSV to RGB (BGR) color representation 
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 
        
        # Opens a new window and displays the output frame 
        show_image(img_data) 
        show_image(rgb) 


    return normalized_magniture, normalized_angle
