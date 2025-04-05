import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import itertools
import time
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['image.cmap'] = 'gray'

# Hough parameters
hough_threshold = 30
min_line_length = 50
max_line_gap = 20
        
# RANSAC parameters
ransac_iterations = 100
ransac_threshold = 30
ransac_min_inliers = 30
max_lines = 30
        
# Target size
target_size = (800, 600)
        
# Quad detection
min_quad_area_ratio = 0.2
max_quad_angle_diff = 20

def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        original_height, original_width = gray.shape
        
        max_dimension = 800
        if max(original_height, original_width) > max_dimension:
            scale_factor = max_dimension / max(original_height, original_width)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height))
            
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(binary, mask)
        kernel = np.ones((3, 3), np.uint8)
        
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(opened)
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contours = sorted_contours[:min(5, len(sorted_contours))]
            cv2.drawContours(mask, largest_contours, -1, 255, 2)
        
        v = np.median(blurred)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh, apertureSize=3, L2gradient=True)
        
        combined_edges = cv2.bitwise_or(edges, mask)
        dilated_edges = cv2.dilate(combined_edges, kernel, iterations=1)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        _, sobel_edges = cv2.threshold(sobel_magnitude, 50, 255, cv2.THRESH_BINARY)
        
        final_edges = cv2.bitwise_or(dilated_edges, sobel_edges)
        final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel)
        
        if max(original_height, original_width) > max_dimension:
            final_edges = cv2.resize(final_edges, (original_width, original_height))
            blurred = cv2.resize(blurred, (original_width, original_height))
        
        return blurred, final_edges

def hough_transform(edges):
        height, width = edges.shape
        
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        cv_lines = cv2.HoughLinesP(
            closed_edges, 
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        if cv_lines is None or len(cv_lines) == 0:
            print("No lines detected using OpenCV HoughLinesP, falling back to custom implementation")
            lines, accumulator = custom_hough_transform(closed_edges)
            
            endpoint_lines = []
            for rho, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                # Calculate endpoints
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Clip endpoints
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length >= min_line_length:
                    endpoint_lines.append((x1, y1, x2, y2))
        else:
            endpoint_lines = [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in cv_lines]
            accumulator = None
            
            # Filter lines
            border_margin_x = int(width * 0.1)
            border_margin_y = int(height * 0.1)
            
            filtered_border_lines = []
            for line in endpoint_lines:
                x1, y1, x2, y2 = line
                
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                if ((x1 < border_margin_x or x1 > width - border_margin_x) and 
                    (x2 < border_margin_x or x2 > width - border_margin_x)):
                    continue
                    
                if ((y1 < border_margin_y or y1 > height - border_margin_y) and 
                    (y2 < border_margin_y or y2 > height - border_margin_y)):
                    continue
                
                filtered_border_lines.append(line)
            
            endpoint_lines = filtered_border_lines
            
            endpoint_lines = filter_lines_by_angle(endpoint_lines)
            endpoint_lines = cluster_similar_lines(endpoint_lines, height, width)
            
            # Sort by length
            endpoint_lines.sort(key=lambda line: line_length(line), reverse=True)
            
            if len(endpoint_lines) > max_lines:
                endpoint_lines = endpoint_lines[:max_lines]
        
        if not endpoint_lines:
            print("No lines detected using Hough Transform")
            return [], None
        
        return endpoint_lines, accumulator
    
def filter_lines_by_angle(lines):
        filtered_lines = []
        
        # Get angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) % 180
            angles.append(angle)
        
        angle_histogram = np.zeros(180)
        for angle in angles:
            angle_histogram[int(angle)] += 1
        
        # Smooth histogram
        smooth_histogram = np.convolve(angle_histogram, np.ones(5)/5, mode='same')
        
        # Find peaks
        peaks = []
        for i in range(1, 179):
            if (smooth_histogram[i] > smooth_histogram[i-1] and 
                smooth_histogram[i] > smooth_histogram[i+1] and
                smooth_histogram[i] > np.mean(smooth_histogram) * 1.5):
                peaks.append(i)
        
        if not peaks:
            # Use strict ranges
            horizontal_range = (0, 15)
            vertical_range = (75, 105)
            
            for i, (line, angle) in enumerate(zip(lines, angles)):
                if ((horizontal_range[0] <= angle <= horizontal_range[1]) or 
                    (180 - horizontal_range[1] <= angle <= 180 - horizontal_range[0]) or
                    (vertical_range[0] <= angle <= vertical_range[1])):
                    filtered_lines.append(line)
            
            if len(filtered_lines) < 4:
                # Extended ranges
                extended_h_range = (0, 20)
                extended_v_range = (70, 110)
                
                for i, (line, angle) in enumerate(zip(lines, angles)):
                    if line not in filtered_lines:
                        if ((extended_h_range[0] <= angle <= extended_h_range[1]) or 
                            (180 - extended_h_range[1] <= angle <= 180 - extended_h_range[0]) or
                            (extended_v_range[0] <= angle <= extended_v_range[1])):
                            filtered_lines.append(line)
                            if len(filtered_lines) >= 8:
                                break
        else:
            # Check peaks
            has_horizontal = any(p < 20 or p > 160 for p in peaks)
            has_vertical = any(70 < p < 110 for p in peaks)
            
            if has_horizontal and has_vertical:
                for i, (line, angle) in enumerate(zip(lines, angles)):
                    if (angle < 20 or angle > 160 or (70 < angle < 110)):
                        for peak in peaks:
                            if abs(angle - peak) <= 10 or abs(angle - (180 - peak)) <= 10:
                                filtered_lines.append(line)
                                break
            else:
                # Use all peaks
                for i, (line, angle) in enumerate(zip(lines, angles)):
                    for peak in peaks:
                        if abs(angle - peak) <= 10 or abs(angle - (180 - peak)) <= 10:
                            filtered_lines.append(line)
                            break
        
        # Add remaining lines
        if len(filtered_lines) < 4:
            remaining_lines = [line for line in lines if line not in filtered_lines]
            remaining_lines.sort(key=lambda line: line_length(line), reverse=True)
            filtered_lines.extend(remaining_lines[:max_lines - len(filtered_lines)])
        
        return filtered_lines
    
def line_length(line):
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
def cluster_similar_lines(lines, height, width):
        if len(lines) <= 4:
            return lines
            
        def line_distance(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Line equations
            if x2 == x1:  # Vertical line
                a1, b1, c1 = 1, 0, -x1
            else:
                m1 = (y2 - y1) / (x2 - x1)
                a1, b1, c1 = m1, -1, y1 - m1 * x1
                
            if x4 == x3:  # Vertical line
                a2, b2, c2 = 1, 0, -x3
            else:
                m2 = (y4 - y3) / (x4 - x3)
                a2, b2, c2 = m2, -1, y3 - m2 * x3
            
            # Normalize coefficients
            norm1 = np.sqrt(a1*a1 + b1*b1)
            a1, b1, c1 = a1/norm1, b1/norm1, c1/norm1
            
            norm2 = np.sqrt(a2*a2 + b2*b2)
            a2, b2, c2 = a2/norm2, b2/norm2, c2/norm2
            
            # Angle difference
            cos_angle = abs(a1*a2 + b1*b2)
            angle_diff = np.arccos(min(1.0, cos_angle)) * 180 / np.pi
            
            # Midpoint distance
            mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
            mid_dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                
            normalized_mid_dist = mid_dist / np.sqrt(height**2 + width**2)
            
            # Combined metric
            return angle_diff + normalized_mid_dist * 100
        
        # Create clusters
        clusters = []
        for line in lines:
            added = False
            for cluster in clusters:
                for cluster_line in cluster:
                    if line_distance(line, cluster_line) < 15:  # Similarity threshold
                        cluster.append(line)
                        added = True
                        break
                if added:
                    break
            
            if not added:
                clusters.append([line])
        
        # Merge clusters
        merged_lines = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged_lines.append(cluster[0])
            else:
                # Use longest line
                cluster.sort(key=lambda l: line_length(l), reverse=True)
                merged_lines.append(cluster[0])
        
        return merged_lines
    
def custom_hough_transform(edges):
        height, width = edges.shape
        
        diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
        thetas = np.arange(0, np.pi, np.pi/180)
        rhos = np.arange(-diagonal, diagonal)
        
        # Initialize accumulator
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        
        # Get edge points
        y_idxs, x_idxs = np.where(edges > 0)
        
        # Accumulator voting
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            
            for theta_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = rho + diagonal
                
                if 0 <= rho_idx < len(rhos):
                    accumulator[rho_idx, theta_idx] += 1
        
        # Find peaks
        lines = []
        for rho_idx, theta_idx in zip(*np.where(accumulator > hough_threshold)):
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            lines.append((rho, theta))
        
        # Sort by votes
        lines_with_votes = [(rho, theta, accumulator[rho + diagonal, theta_idx]) 
                          for rho, theta, theta_idx in 
                          [(l[0], l[1], np.where(thetas == l[1])[0][0]) for l in lines]]
        
        sorted_lines = [l[:2] for l in sorted(lines_with_votes, key=lambda x: x[2], reverse=True)]
        
        # Limit lines
        if len(sorted_lines) > max_lines:
            sorted_lines = sorted_lines[:max_lines]
            
        return sorted_lines, accumulator

def ransac_line_fitting(edges, lines):
        height, width = edges.shape
        
        if not lines:
            return []
        
        # Refined lines
        refined_lines = []
        
        # Get edge points
        y_indices, x_indices = np.where(edges > 0)
        
        # Sample subset
        max_points = 25000
        if len(y_indices) > max_points:
            sample_idx = np.random.choice(len(y_indices), max_points, replace=False)
            edge_points = np.column_stack((x_indices[sample_idx], y_indices[sample_idx]))
        else:
            edge_points = np.column_stack((x_indices, y_indices))
        
        # RANSAC parameters
        distance_threshold = ransac_threshold
        min_inliers = max(ransac_min_inliers, int(0.005 * len(edge_points)))
        
        # Process each line
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Skip short lines
            if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < 20:
                continue
            
            # Line equation
            if x2 == x1:  # Vertical line
                a, b, c = 1, 0, -x1
            else:
                m = (y2 - y1) / (x2 - x1)
                a, b, c = m, -1, y1 - m * x1
            
            # Normalize
            norm = np.sqrt(a*a + b*b)
            a, b, c = a/norm, b/norm, c/norm
            
            # Find nearby points
            distances = np.abs(a*edge_points[:, 0] + b*edge_points[:, 1] + c)
            near_points_mask = distances < distance_threshold * 3
            
            # Too few points
            if np.sum(near_points_mask) < 10:
                refined_lines.append((x1, y1, x2, y2))
                continue
                
            near_points = edge_points[near_points_mask]
            
            # Best model
            best_params = (a, b, c)
            best_inliers_count = 0
            best_inliers = None
            
            # RANSAC iterations
            for _ in range(ransac_iterations):
                # Random points
                if len(near_points) > 1:
                    indices = np.random.choice(len(near_points), 2, replace=False)
                    p1, p2 = near_points[indices[0]], near_points[indices[1]]
                    
                    # Points too close
                    if np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) < 10:
                        continue
                    
                    # New line params
                    if abs(p2[0] - p1[0]) < 1e-6:  # Vertical line
                        a_new, b_new, c_new = 1, 0, -p1[0]
                    else:
                        m_new = (p2[1] - p1[1]) / (p2[0] - p1[0])
                        a_new, b_new, c_new = m_new, -1, p1[1] - m_new * p1[0]
                    
                    # Normalize
                    norm = np.sqrt(a_new*a_new + b_new*b_new)
                    if norm < 1e-10:
                        continue
                    a_new, b_new, c_new = a_new/norm, b_new/norm, c_new/norm
                    
                    # Check similarity
                    angle_similarity = abs(a*a_new + b*b_new)
                    if angle_similarity < 0.9:
                        continue
                    
                    # Count inliers
                    distances = np.abs(a_new*near_points[:, 0] + b_new*near_points[:, 1] + c_new)
                    inlier_mask = distances < distance_threshold
                    inliers = np.sum(inlier_mask)
                    
                    # Update best
                    if inliers > best_inliers_count:
                        best_inliers_count = inliers
                        best_params = (a_new, b_new, c_new)
                        best_inliers = near_points[inlier_mask]
            
            # Not enough inliers
            if best_inliers_count < min_inliers:
                refined_lines.append((x1, y1, x2, y2))
                continue
                
            # Get parameters
            a, b, c = best_params
            
            # Calculate direction
            orig_direction = np.array([x2 - x1, y2 - y1])
            orig_length = np.sqrt(np.sum(orig_direction**2))
            
            if orig_length > 0:
                orig_direction = orig_direction / orig_length
                
                # Check alignment
                if best_inliers is not None and len(best_inliers) >= 2:
                    # Get extreme points
                    if abs(b) < 1e-6:  # Near-vertical
                        sorted_inliers = best_inliers[np.argsort(best_inliers[:, 1])]
                        p1 = sorted_inliers[0]
                        p2 = sorted_inliers[-1]
                    else:
                        sorted_inliers = best_inliers[np.argsort(best_inliers[:, 0])]
                        p1 = sorted_inliers[0]
                        p2 = sorted_inliers[-1]
                    
                    refined_direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                    refined_length = np.sqrt(np.sum(refined_direction**2))
                    
                    if refined_length > 0:
                        refined_direction = refined_direction / refined_length
                        direction_similarity = np.abs(np.dot(orig_direction, refined_direction))
                        
                        # Direction too different
                        if direction_similarity < 0.9:
                            refined_lines.append((x1, y1, x2, y2))
                            continue
                        
                        # Use inlier points
                        x_new1, y_new1 = p1
                        x_new2, y_new2 = p2
                        
                        # Extend if needed
                        if refined_length < orig_length * 0.9:
                            mid_x = (x_new1 + x_new2) / 2
                            mid_y = (y_new1 + y_new2) / 2
                            
                            half_orig_length = orig_length / 2
                            x_new1 = mid_x - refined_direction[0] * half_orig_length
                            y_new1 = mid_y - refined_direction[1] * half_orig_length
                            x_new2 = mid_x + refined_direction[0] * half_orig_length
                            y_new2 = mid_y + refined_direction[1] * half_orig_length
                
            # Line boundaries
            else:
                if abs(b) < 1e-6:  # Vertical line
                    x_new1 = x_new2 = -c / a
                    y_new1 = 0
                    y_new2 = height - 1
                elif abs(a) < 1e-6:  # Horizontal line
                    x_new1 = 0
                    x_new2 = width - 1
                    y_new1 = y_new2 = -c / b
                else:
                    # Find intersections
                    points = []
                    
                    # Left border
                    y_left = -c / b
                    if 0 <= y_left <= height-1:
                        points.append((0, y_left))
                    
                    # Right border
                    y_right = -(a*(width-1) + c) / b
                    if 0 <= y_right <= height-1:
                        points.append((width-1, y_right))
                    
                    # Top border
                    x_top = -c / a
                    if 0 <= x_top <= width-1:
                        points.append((x_top, 0))
                    
                    # Bottom border
                    x_bottom = -(b*(height-1) + c) / a
                    if 0 <= x_bottom <= width-1:
                        points.append((x_bottom, height-1))
                    
                    # Find farthest points
                    if len(points) >= 2:
                        max_dist = 0
                        x_new1, y_new1, x_new2, y_new2 = 0, 0, 0, 0
                        
                        for i in range(len(points)):
                            for j in range(i+1, len(points)):
                                p1, p2 = points[i], points[j]
                                dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                                
                                if dist > max_dist:
                                    max_dist = dist
                                    x_new1, y_new1 = p1
                                    x_new2, y_new2 = p2
                    else:
                        # Fallback
                        x_new1, y_new1, x_new2, y_new2 = x1, y1, x2, y2
            
            # Integer values
            x_new1, y_new1 = int(x_new1), int(y_new1)
            x_new2, y_new2 = int(x_new2), int(y_new2)
            
            # Check length
            current_line_length = np.sqrt((x_new2 - x_new1)**2 + (y_new2 - y_new1)**2)
            if current_line_length < 20:
                refined_lines.append((x1, y1, x2, y2))
                continue
            
            # Add line
            refined_lines.append((x_new1, y_new1, x_new2, y_new2))
        
        # Ensure 4+ lines
        if len(refined_lines) < 4:
            print("RANSAC produced too few lines, using original Hough lines")
            return lines
        
        # Sort by length
        refined_lines.sort(key=lambda line: line_length(line), reverse=True)
        
        return refined_lines

def find_quadrilateral(lines, image):
        if len(lines) < 4:
            return None
            
        # Get dimensions
        if image is not None:
            height, width = image.shape[:2]
        else:
            max_x, max_y = 0, 0
            for x1, y1, x2, y2 in lines:
                max_x = max(max_x, x1, x2)
                max_y = max(max_y, y1, y2)
            width = max_x + 100
            height = max_y + 100
        
        # Helper functions
        def line_angle(line):
            x1, y1, x2, y2 = line
            return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        def line_length(line):
            x1, y1, x2, y2 = line
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Standard form
            a1 = y2 - y1
            b1 = x1 - x2
            c1 = x2 * y1 - x1 * y2
            
            a2 = y4 - y3
            b2 = x3 - x4
            c2 = x4 * y3 - x3 * y4
            
            # Determinant
            det = a1 * b2 - a2 * b1
            
            if abs(det) < 1e-6:  # Parallel lines
                return None
                
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            
            # Check boundaries
            margin = 200
            if -margin <= x <= width + margin and -margin <= y <= height + margin:
                return (float(x), float(y))
                
            return None
        
        # Sort lines by length
        lines = sorted(lines, key=line_length, reverse=True)
        
        # Group by angle
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            angle = line_angle(line)
            if angle < 30 or angle > 150:  # Horizontal
                horizontal_lines.append(line)
            elif 60 < angle < 120:  # Vertical
                vertical_lines.append(line)
        
        # Keep longest
        horizontal_lines = horizontal_lines[:min(5, len(horizontal_lines))]
        vertical_lines = vertical_lines[:min(5, len(vertical_lines))]
        
        print(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
        
        # Need minimum lines
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            print("Not enough horizontal or vertical lines.")
            return None
        
        # Try all combinations
        best_quad = None
        best_score = 0
        
        for h_lines in itertools.combinations(horizontal_lines, 2):
            for v_lines in itertools.combinations(vertical_lines, 2):
                # Get intersections
                intersections = []
                for h in h_lines:
                    for v in v_lines:
                        intersection = line_intersection(h, v)
                        if intersection:
                            intersections.append(intersection)
                
                # Check quad
                if len(intersections) == 4:
                    # Order points
                    ordered = order_points(np.array(intersections))
                    
                    # Calculate metrics
                    sides = []
                    angles = []
                    
                    for i in range(4):
                        next_i = (i + 1) % 4
                        # Side length
                        side = np.sqrt((ordered[next_i][0] - ordered[i][0])**2 + 
                                     (ordered[next_i][1] - ordered[i][1])**2)
                        sides.append(side)
                        
                        # Corner angle
                        if i < 3:
                            next_next_i = (i + 2) % 4
                            v1 = [ordered[next_i][0] - ordered[i][0], ordered[next_i][1] - ordered[i][1]]
                            v2 = [ordered[next_next_i][0] - ordered[next_i][0], ordered[next_next_i][1] - ordered[next_i][1]]
                            
                            # Normalize
                            v1_norm = np.sqrt(v1[0]**2 + v1[1]**2)
                            v2_norm = np.sqrt(v2[0]**2 + v2[1]**2)
                            
                            if v1_norm > 0 and v2_norm > 0:
                                v1 = [v1[0]/v1_norm, v1[1]/v1_norm]
                                v2 = [v2[0]/v2_norm, v2[1]/v2_norm]
                                
                                # Calculate angle
                                dot = v1[0]*v2[0] + v1[1]*v2[1]
                                angle = np.arccos(np.clip(dot, -1.0, 1.0)) * 180 / np.pi
                                angles.append(angle)
                    
                    # Calculate area
                    x = [p[0] for p in ordered]
                    y = [p[1] for p in ordered]
                    area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(3)) + 
                                   x[3]*y[0] - x[0]*y[3])
                    
                    # Area ratio
                    area_ratio = area / (width * height)
                    
                    # Angle quality
                    angle_deviation = sum([abs(a - 90) for a in angles]) / len(angles)
                    
                    # Aspect ratio
                    width_avg = (sides[0] + sides[2]) / 2
                    height_avg = (sides[1] + sides[3]) / 2
                    aspect_ratio = max(width_avg, height_avg) / min(width_avg, height_avg)
                    
                    # Combined score
                    score = area_ratio * (1 - angle_deviation / 90)
                    
                    # Quality checks
                    min_side = min(sides)
                    if min_side < 20:  # Too small
                        continue
                        
                    # Center check
                    center_x = sum(x) / 4
                    center_y = sum(y) / 4
                    center_dist = np.sqrt((center_x - width/2)**2 + (center_y - height/2)**2)
                    max_dist = np.sqrt((width/2)**2 + (height/2)**2)
                    if center_dist > 0.5 * max_dist:  # Too far from center
                        continue
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        best_quad = ordered.tolist()
        
        if best_quad:
            print(f"Found quadrilateral with score: {best_score:.4f}")
            return best_quad
            
        print("No suitable quadrilateral found.")
        return None

def perspective_correction(image, quad_points):
        if quad_points is None or len(quad_points) != 4:
            return image
        
        # Convert to array
        points = np.array(quad_points, dtype=np.float32)
        
        # Order points
        rect = order_points(points)
        
        print(f"Ordered points for perspective correction: {rect}")
        
        # Calculate dimensions
        width_top = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        width_bottom = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        max_width = max(int(width_top), int(width_bottom))
        
        height_right = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_left = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        max_height = max(int(height_right), int(height_left))
        
        # Check proportions
        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_right + height_left) / 2
        
        # Nearly square
        if abs(avg_width - avg_height) / max(avg_width, avg_height) <= 0.15:
            print("Document dimensions are nearly square, preserving original dimensions")
            pass
        else:
            # Portrait/landscape check
            if avg_height > avg_width:  # Portrait
                target_ratio = 1.414
                current_ratio = avg_height / avg_width
                
                print(f"Portrait orientation detected. Current ratio: {current_ratio:.3f}, target: {target_ratio:.3f}")
                
                if abs(current_ratio - target_ratio) > 0.15:  # Need adjustment
                    if current_ratio < target_ratio:
                        # Make taller
                        max_height = int(max_width * target_ratio)
                        print(f"Adjusting height to match A4 portrait ratio: {max_width}x{max_height}")
                    else:
                        # Make wider
                        max_width = int(max_height / target_ratio)
                        print(f"Adjusting width to match A4 portrait ratio: {max_width}x{max_height}")
            else:  # Landscape
                target_ratio = 1.414
                current_ratio = avg_width / avg_height
                
                print(f"Landscape orientation detected. Current ratio: {current_ratio:.3f}, target: {target_ratio:.3f}")
                
                if abs(current_ratio - target_ratio) > 0.15:  # Need adjustment
                    if current_ratio < target_ratio:
                        # Make wider
                        max_width = int(max_height * target_ratio)
                        print(f"Adjusting width to match A4 landscape ratio: {max_width}x{max_height}")
                    else:
                        # Make taller
                        max_height = int(max_width / target_ratio)
                        print(f"Adjusting height to match A4 landscape ratio: {max_width}x{max_height}")
        
        # Size limits
        if max_width > 3000 or max_height > 3000:
            # Scale down
            aspect_ratio = max_width / max_height
            if aspect_ratio > 1:  # wider
                max_width = min(max_width, 1200)
                max_height = int(max_width / aspect_ratio)
            else:  # taller
                max_height = min(max_height, 1200)
                max_width = int(max_height * aspect_ratio)
        
        # Minimum size
        max_width = max(max_width, 300)
        max_height = max(max_height, 300)
        
        print(f"Output dimensions: {max_width}x{max_height}")
        
        # Destination points
        dst = np.array([
            [0, 0],  # top-left
            [max_width - 1, 0],  # top-right
            [max_width - 1, max_height - 1],  # bottom-right
            [0, max_height - 1]  # bottom-left
        ], dtype=np.float32)
        
        # Transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply warp
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
def order_points(pts):
        # Initialize array
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum coordinates
        s = pts.sum(axis=1)
        
        # Top-left (smallest sum)
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right (largest sum)
        rect[2] = pts[np.argmax(s)]
        
        # Difference coordinates
        diff = np.diff(pts, axis=1)
        
        # Top-right (smallest diff)
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left (largest diff)
        rect[3] = pts[np.argmax(diff)]
        
        return rect

def process_image(input_path, show_steps=True):
    try:
        # Start timing
        start_time = time.time()
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image at {input_path}")
            return None
        
        original = image.copy()
        
        # Show original
        if show_steps:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title("1. Original Image")
            plt.axis('off')
            plt.show()
    
        # Preprocess
        print("Preprocessing image...")
        preprocess_start = time.time()
        gray, edges = preprocess_image(image)
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
        
        # Show preprocessed
        if show_steps:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(gray, cmap='gray')
            plt.title("2a. Grayscale Preprocessed")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(edges, cmap='gray')
            plt.title("2b. Edge Detection")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Detect lines
        print("Applying Hough Transform...")
        hough_start = time.time()
        lines, accumulator = hough_transform(edges)
        hough_time = time.time() - hough_start
        print(f"Hough Transform detected {len(lines)} lines in {hough_time:.2f} seconds")
    
        if len(lines) < 4:
            print("Warning: Fewer than 4 lines detected. Document corners may not be found correctly.")
        
        # Show lines
        if show_steps:
            hough_lines_image = original.copy()
            for x1, y1, x2, y2 in lines:
                cv2.line(hough_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(hough_lines_image, cv2.COLOR_BGR2RGB))
            plt.title(f"3. Hough Transform Lines ({len(lines)} lines)")
            plt.axis('off')
            plt.show()
    
        # Refine lines
        print("Refining lines with RANSAC...")
        ransac_start = time.time()
        refined_lines = ransac_line_fitting(edges, lines)
        ransac_time = time.time() - ransac_start
        print(f"RANSAC refined to {len(refined_lines)} lines in {ransac_time:.2f} seconds")
    
        # Fallback if needed
        if len(refined_lines) < 4 and len(lines) >= 4:
            print("Warning: RANSAC reduced lines below 4. Using original Hough lines.")
            refined_lines = lines
        
        # Show refined
        if show_steps:
            ransac_image = original.copy()
            for x1, y1, x2, y2 in refined_lines:
                cv2.line(ransac_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(ransac_image, cv2.COLOR_BGR2RGB))
            plt.title(f"4. RANSAC Refined Lines ({len(refined_lines)} lines)")
            plt.axis('off')
            plt.show()
    
        # Find quadrilateral
        print("Finding quadrilateral...")
        quad_start = time.time()
        quad_points = find_quadrilateral(refined_lines, image)
        
        # Try original lines
        if not quad_points and refined_lines != lines:
            print("No quadrilateral found with RANSAC lines. Trying with original Hough lines...")
            quad_points = find_quadrilateral(lines, image)
        
        quad_time = time.time() - quad_start
        print(f"Quadrilateral detection completed in {quad_time:.2f} seconds")
        
        # Show quadrilateral
        if show_steps and quad_points:
            quad_image = original.copy()
            
            for x1, y1, x2, y2 in refined_lines:
                cv2.line(quad_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            points = np.array(quad_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(quad_image, [points], True, (255, 0, 0), 3)
            
            for i, (x, y) in enumerate(quad_points):
                cv2.circle(quad_image, (int(x), int(y)), 10, (0, 255, 255), -1)
                cv2.putText(quad_image, f"P{i}", (int(x)+10, int(y)+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(quad_image, cv2.COLOR_BGR2RGB))
            plt.title("5. Detected Document Quadrilateral")
            plt.axis('off')
            plt.show()
    
        # Apply correction
        if quad_points:
            print(f"Found quadrilateral with corners: {quad_points}")
            persp_start = time.time()
            corrected = perspective_correction(image, quad_points)
            persp_time = time.time() - persp_start
            print(f"Perspective correction completed in {persp_time:.2f} seconds")
            
            # Show result
            if show_steps:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
                plt.title("6. Perspective Corrected Image")
                plt.axis('off')
                plt.show()
                
                # Compare images
                plt.figure(figsize=(15, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
                plt.title("Corrected Image")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        else:
            print("Could not find a proper quadrilateral. Original image will be returned.")
            corrected = image.copy()
            
            # Show original
            if show_steps:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.title("No Quadrilateral Found - Original Image")
                plt.axis('off')
                plt.show()
        
        # Total time
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds\n")
        
        return corrected
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original on error
        return image if 'image' in locals() else None

def evaluate_correction(corrected, ground_truth):
        # Convert to grayscale
        if len(corrected.shape) == 3:
            corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        else:
            corrected_gray = corrected
            
        if len(ground_truth.shape) == 3:
            gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        else:
            gt_gray = ground_truth
        
        # Match dimensions
        if corrected_gray.shape != gt_gray.shape:
            corrected_gray = cv2.resize(corrected_gray, (gt_gray.shape[1], gt_gray.shape[0]))
        
        # Convert to float
        corrected_gray = corrected_gray.astype(np.float32)
        gt_gray = gt_gray.astype(np.float32)
        
        # Calculate SSIM
        ssim_score = ssim(corrected_gray, gt_gray, data_range=255)
        return ssim_score

# Dataset paths
dataset_path = 'C:/Users/ufuka/IdeaProjects/Downloads/WarpDoc/WarpDoc/distorted'
ground_truth_dataset_path = 'C:/Users/ufuka/IdeaProjects/Downloads/WarpDoc/WarpDoc/digital'
image_path = os.path.join(dataset_path, 'rotate', '0004.jpg')
corrected_image = process_image(image_path, show_steps=True)

if corrected_image is not None:
    # Find ground truth
    image_name = os.path.basename(image_path)
    image_class = os.path.basename(os.path.dirname(image_path))
    ground_truth_path = os.path.join(ground_truth_dataset_path, image_class, image_name)
    
    if os.path.exists(ground_truth_path):
        print("Comparing with ground truth...")
        ground_truth_image = cv2.imread(ground_truth_path)
        if ground_truth_image is not None:
            ssim_value = evaluate_correction(corrected_image, ground_truth_image)
            print(f"SSIM comparison with ground truth: {ssim_value:.4f}")
        else:
            print(f"Could not read ground truth image at {ground_truth_path}")
    else:
        print(f"Ground truth image not found at {ground_truth_path}")
else:
    print("Image processing failed")