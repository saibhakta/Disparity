import sys
import os
import argparse
import json
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QLabel, QMessageBox,
    QSizePolicy, QFrame, QGraphicsEllipseItem, QGraphicsRectItem # Added QGraphicsRectItem
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont, QTransform
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
from utils import CalibrationData

# --- Configuration for Point Markers ---
# POINT_DIAMETER is in scene (image pixel) units. A diameter of 1.0 covers one pixel.
POINT_DIAMETER = 1.0
POINT_COLOR_LEFT_PENDING = QColor(Qt.GlobalColor.red)
POINT_COLOR_COMPLETE = QColor(Qt.GlobalColor.green)
TEXT_COLOR = QColor(Qt.GlobalColor.yellow)
TEXT_FONT_SIZE = 10 # In points, so screen size is relatively constant
# Offset for text from the point's edge, in scene units
TEXT_DISPLAY_OFFSET_X_SCENE = 1.0
TEXT_DISPLAY_OFFSET_Y_SCENE = 1.0


class ZoomableView(QGraphicsView):
    point_clicked = pyqtSignal(QPointF)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        # Turn off antialiasing for sharp pixels when zoomed
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Default to NoDrag for selection clicks. Pan is initiated by Ctrl/Cmd + Click.
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor) # Default cursor for selection
        
        self._is_panning_active = False # Tracks if a pan is currently engaged by Ctrl/Cmd+Click
        self.pixmap_item = None # Ensure pixmap_item is initialized

    def set_pixmap(self, pixmap):
        if self.pixmap_item:
            self.scene().removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())

        if self.viewport().size().isValid() and not self.viewport().size().isEmpty():
             self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        current_scale = self.transform().m11()
        
        MIN_SCALE = 0.005 
        MAX_SCALE = 200.0  

        if event.angleDelta().y() > 0: 
            if current_scale * zoom_in_factor <= MAX_SCALE:
                self.scale(zoom_in_factor, zoom_in_factor)
        else: 
            if current_scale * zoom_out_factor >= MIN_SCALE:
                self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        # Use QApplication.keyboardModifiers() for the current global state of modifiers
        modifiers = QApplication.keyboardModifiers()
        is_pan_modifier_pressed = (modifiers & Qt.KeyboardModifier.ControlModifier) or \
                                  (modifiers & Qt.KeyboardModifier.MetaModifier)

        if event.button() == Qt.MouseButton.LeftButton:
            if is_pan_modifier_pressed:
                self._is_panning_active = True
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                # QGraphicsView automatically handles OpenHand/ClosedHand cursors in ScrollHandDrag mode
            else:
                # Not a pan, treat as potential selection. Ensure NoDrag mode and Arrow cursor.
                self._is_panning_active = False 
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor) 
                
                scene_pos = self.mapToScene(event.pos())
                if self.pixmap_item and self.pixmap_item.boundingRect().contains(scene_pos):
                    self.point_clicked.emit(scene_pos)
        
        super().mousePressEvent(event) # CRITICAL: QGraphicsView processes the event based on current DragMode

    def mouseMoveEvent(self, event):
        # If panning is active (_is_panning_active is true, left button is pressed,
        # and DragMode is ScrollHandDrag), QGraphicsView handles the panning and cursor.
        # If not panning, ensure cursor is Arrow if it was changed.
        if not self._is_panning_active and self.dragMode() == QGraphicsView.DragMode.NoDrag:
            if self.viewport().cursor().shape() != Qt.CursorShape.ArrowCursor:
                 self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_panning_active:
                self._is_panning_active = False
                # QGraphicsView in ScrollHandDrag mode will change cursor back from ClosedHand
                # when mouse is released. We then set NoDrag and ArrowCursor.
            
            # Always reset to NoDrag and ArrowCursor after a left-button release
            # to be ready for the next selection or pan initiation.
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self._is_panning_active: # If a pan was initiated (Ctrl/Cmd + MouseDown)
                self._is_panning_active = False
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                
                # This should be sufficient to stop the view from panning further
                # if the mouse button is still held down, as the drag mode is now NoDrag.
                # The QGraphicsView itself handles its internal state based on DragMode.
                print("Panning cancelled by Escape key.") # For debugging
                event.accept() 
                return
        super().keyPressEvent(event)


class AnnotationWindow(QMainWindow):
    def __init__(self, images_dir, calibration_file_path, annotations_dir):
        super().__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        os.makedirs(self.annotations_dir, exist_ok=True)

        try:
            self.calibration_data = CalibrationData(calibration_file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load calibration file: {e}")
            sys.exit(1)

        self.all_image_pairs = []
        self.current_pair_index = -1
        
        self.current_left_img_path_global = None
        self.current_right_img_path_global = None
        self.current_image_pair_name_global = None

        self.points_left_coords = []  # Store QPointF (pixel centers)
        self.points_right_coords = [] # Store QPointF (pixel centers)
        self.point_items_left = []
        self.point_items_right = []
        
        self._initial_fit_done = False # Flag for initial zoom to fit

        self._setup_ui()
        self._scan_image_pairs()

        if not self.all_image_pairs:
            QMessageBox.information(self, "No Images", "No image pairs found in the specified directory.")
            self.save_button.setEnabled(False)
            self.undo_button.setEnabled(False)
        else:
            self._load_next_unannotated_pair()
            # If _load_next_unannotated_pair reaches the end without finding an unannotated pair
            if self.current_pair_index >= len(self.all_image_pairs) or not self.current_left_img_path_global:
                 # Message is already shown by _load_next_unannotated_pair
                 self.save_button.setEnabled(False)
                 self.undo_button.setEnabled(False)

    def showEvent(self, event):
        super().showEvent(event)
        # This ensures that after the window is shown and laid out for the first time,
        # we fit the images properly.
        if not self._initial_fit_done:
            if self.left_view.pixmap_item and self.left_view.viewport().size().isValid():
                self.left_view.fitInView(self.left_view.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            if self.right_view.pixmap_item and self.right_view.viewport().size().isValid():
                self.right_view.fitInView(self.right_view.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._initial_fit_done = True

    def _setup_ui(self):
        self.setWindowTitle("Ground Truth Disparity Annotation Tool")
        self.setGeometry(100, 100, 1600, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        image_layout = QHBoxLayout()
        self.left_scene = QGraphicsScene(self)
        self.left_view = ZoomableView(self.left_scene, self)
        self.left_view.point_clicked.connect(self._handle_left_image_click)

        self.right_scene = QGraphicsScene(self)
        self.right_view = ZoomableView(self.right_scene, self)
        self.right_view.point_clicked.connect(self._handle_right_image_click)

        image_layout.addWidget(self.left_view)
        image_layout.addWidget(self.right_view)
        main_layout.addLayout(image_layout, 1) # Stretch factor 1

        status_layout = QHBoxLayout()
        self.left_image_label = QLabel("Left Image: N/A")
        self.right_image_label = QLabel("Right Image: N/A")
        self.points_count_label = QLabel("Points: 0")
        status_layout.addWidget(self.left_image_label)
        status_layout.addWidget(self.right_image_label)
        status_layout.addWidget(self.points_count_label)
        main_layout.addLayout(status_layout)
        
        self.instruction_label = QLabel("Instructions: Click on the left image, then the corresponding point on the right image.")
        main_layout.addWidget(self.instruction_label)

        button_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo Last Point")
        self.undo_button.clicked.connect(self._on_undo_button_click)
        self.save_button = QPushButton("Save & Next Image Pair")
        self.save_button.clicked.connect(self._on_save_button_click)

        button_layout.addWidget(self.undo_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        main_layout.addLayout(button_layout)

        self.undo_button.setEnabled(False)

    def _scan_image_pairs(self):
        left_dir = os.path.join(self.images_dir, "left")
        right_dir = os.path.join(self.images_dir, "right")

        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
            QMessageBox.warning(self, "Directory Error", f"Could not find 'left' or 'right' subdirectories in {self.images_dir}")
            return

        left_files = sorted([f for f in os.listdir(left_dir) if os.path.isfile(os.path.join(left_dir, f))])
        
        self.all_image_pairs = []
        for lf in left_files:
            rf_name = lf.replace("left", "right", 1) 
            rf_path = os.path.join(right_dir, rf_name)
            lf_path = os.path.join(left_dir, lf)

            if os.path.exists(rf_path):
                base_name = os.path.splitext(lf)[0].replace("left", "", 1).lstrip('_').lstrip('0')
                if not base_name: 
                    base_name = os.path.splitext(lf)[0] # Fallback if stripping results in empty
                self.all_image_pairs.append((base_name, lf_path, rf_path))
            else:
                print(f"Warning: No corresponding right image found for {lf_path} (expected {rf_name})")
        
        if not self.all_image_pairs:
             print("No image pairs found after scanning.")
        else:
             print(f"Found {len(self.all_image_pairs)} potential image pairs.")

    def _load_next_unannotated_pair(self):
        self.current_pair_index += 1
        while self.current_pair_index < len(self.all_image_pairs):
            base_name, left_path, right_path = self.all_image_pairs[self.current_pair_index]
            # Ensure base_name is filesystem-friendly if it might contain problematic chars
            safe_base_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name)
            annotation_filename = os.path.join(self.annotations_dir, f"{safe_base_name}.json")


            if not os.path.exists(annotation_filename):
                self.current_image_pair_name_global = safe_base_name # Use safe name for saving
                self.current_left_img_path_global = left_path
                self.current_right_img_path_global = right_path
                self._display_current_pair()
                print(f"You have done {self.current_pair_index} pairs and have {len(self.all_image_pairs) - self.current_pair_index} left.")
                return
            self.current_pair_index += 1

        QMessageBox.information(self, "All Done", "All image pairs have been annotated!")
        self.save_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.left_image_label.setText("Left Image: All Done")
        self.right_image_label.setText("Right Image: All Done")
        self.instruction_label.setText("All image pairs processed.")
        self.current_left_img_path_global = None 
        self.current_right_img_path_global = None

    def _clear_points_from_scene(self, scene, point_items_list):
        for items in point_items_list:
            for item in items:
                scene.removeItem(item)
        point_items_list.clear()

    def _display_current_pair(self):
        self._clear_points_from_scene(self.left_scene, self.point_items_left)
        self._clear_points_from_scene(self.right_scene, self.point_items_right)
        
        self.points_left_coords.clear()
        self.points_right_coords.clear()

        img_l_bgr = cv2.imread(self.current_left_img_path_global)
        img_r_bgr = cv2.imread(self.current_right_img_path_global)

        if img_l_bgr is None or img_r_bgr is None:
            QMessageBox.critical(self, "Error", f"Could not load images: {self.current_left_img_path_global} or {self.current_right_img_path_global}")
            self._load_next_unannotated_pair() 
            return

        try:
            if self.calibration_data.image_size != img_l_bgr.shape[1::-1]:
                 self.calibration_data.init_rectification_maps(img_l_bgr.shape[1::-1])
            
            rect_l, rect_r = self.calibration_data.rectify_image_pair(img_l_bgr, img_r_bgr)
        except Exception as e:
            QMessageBox.critical(self, "Rectification Error", f"Error during image rectification: {e}")
            self._load_next_unannotated_pair() 
            return

        q_img_l = QImage(rect_l.data, rect_l.shape[1], rect_l.shape[0], rect_l.strides[0], QImage.Format.Format_BGR888)
        q_pixmap_l = QPixmap.fromImage(q_img_l)
        self.left_view.set_pixmap(q_pixmap_l)

        q_img_r = QImage(rect_r.data, rect_r.shape[1], rect_r.shape[0], rect_r.strides[0], QImage.Format.Format_BGR888)
        q_pixmap_r = QPixmap.fromImage(q_img_r)
        self.right_view.set_pixmap(q_pixmap_r)
        
        self.left_image_label.setText(f"Left: {os.path.basename(self.current_left_img_path_global)}")
        self.right_image_label.setText(f"Right: {os.path.basename(self.current_right_img_path_global)}")
        self._update_status_and_instructions()
        self.undo_button.setEnabled(False)

    def _get_pixel_center_from_scene_pos(self, scene_pos: QPointF) -> QPointF:
        # Snaps a scene coordinate to the center of the pixel it falls into.
        # E.g., if scene_pos is (10.3, 20.8), it's within pixel (10,20).
        # The center of pixel (x_int, y_int) is (x_int + 0.5, y_int + 0.5).
        return QPointF(np.floor(scene_pos.x()) + 0.5, np.floor(scene_pos.y()) + 0.5)

    def _add_point_marker(self, scene: QGraphicsScene, pos: QPointF, number: int, color: QColor, point_items_list: list):
        # pen = QPen(color) # For outline, if desired
        # pen.setWidthF(0.1) # Very thin outline in scene units
        pen = QPen(Qt.PenStyle.NoPen) # No outline, rely on fill for 1x1 pixel marker
        brush = QBrush(color)

        # pos is the center of the pixel, e.g., (100.5, 50.5)
        # To draw a 1x1 scene unit marker centered at pos:
        marker_x = pos.x() - POINT_DIAMETER / 2.0
        marker_y = pos.y() - POINT_DIAMETER / 2.0
        # Using addRect for a sharp 1x1 pixel square
        marker = scene.addRect(marker_x, marker_y, POINT_DIAMETER, POINT_DIAMETER, pen, brush)

        # Text label
        text_item = scene.addSimpleText(str(number))
        text_item.setBrush(QBrush(TEXT_COLOR))
        font = QFont()
        font.setPointSize(TEXT_FONT_SIZE)
        font.setBold(True)
        text_item.setFont(font)
        
        # Position text's top-left corner to be top-right of the point marker
        # text_item.boundingRect().height() is height in local (unscaled by view) text units
        text_height_in_scene_units = text_item.boundingRect().height()
        
        text_x = pos.x() + (POINT_DIAMETER / 2.0) + TEXT_DISPLAY_OFFSET_X_SCENE
        text_y = pos.y() - (POINT_DIAMETER / 2.0) - text_height_in_scene_units - TEXT_DISPLAY_OFFSET_Y_SCENE
        
        text_item.setPos(text_x, text_y)
        
        point_items_list.append([marker, text_item])

    def _handle_left_image_click(self, scene_pos: QPointF):
        if len(self.points_left_coords) > len(self.points_right_coords):
            QMessageBox.warning(self, "Selection Order", "Please select the corresponding point on the RIGHT image.")
            return
        
        pixel_center_pos = self._get_pixel_center_from_scene_pos(scene_pos)
        self.points_left_coords.append(pixel_center_pos)
        point_num = len(self.points_left_coords)
        self._add_point_marker(self.left_scene, pixel_center_pos, point_num, POINT_COLOR_LEFT_PENDING, self.point_items_left)
        self.undo_button.setEnabled(True)
        self._update_status_and_instructions()

    def _handle_right_image_click(self, scene_pos: QPointF):
        if len(self.points_right_coords) >= len(self.points_left_coords):
            QMessageBox.warning(self, "Selection Order", "Please select a point on the LEFT image first.")
            return

        pixel_center_pos = self._get_pixel_center_from_scene_pos(scene_pos)
        self.points_right_coords.append(pixel_center_pos)
        point_num = len(self.points_right_coords)

        # Update color of corresponding left point to 'complete'
        if self.point_items_left and len(self.point_items_left) >= point_num:
            left_marker_items = self.point_items_left[point_num-1]
            # left_marker_items[0] is the QGraphicsRectItem
            if left_marker_items and isinstance(left_marker_items[0], QGraphicsRectItem):
                 left_marker_items[0].setBrush(QBrush(POINT_COLOR_COMPLETE))
                 # left_marker_items[0].setPen(QPen(POINT_COLOR_COMPLETE)) # If using pen
                 left_marker_items[0].setPen(QPen(Qt.PenStyle.NoPen)) # Consistent with creation


        self._add_point_marker(self.right_scene, pixel_center_pos, point_num, POINT_COLOR_COMPLETE, self.point_items_right)
        self.undo_button.setEnabled(True)
        self._update_status_and_instructions()

    def _update_status_and_instructions(self):
        num_pairs = len(self.points_right_coords)
        self.points_count_label.setText(f"Completed Point Pairs: {num_pairs}")

        if len(self.points_left_coords) > len(self.points_right_coords):
            self.instruction_label.setText(f"Click on RIGHT image for point {len(self.points_left_coords)}.")
        else:
            self.instruction_label.setText(f"Click on LEFT image for point {len(self.points_left_coords) + 1}.")
        
        if not self.points_left_coords and not self.points_right_coords:
            self.undo_button.setEnabled(False)
        else:
            self.undo_button.setEnabled(True)

    def _on_undo_button_click(self):
        if self.point_items_right: 
            items_to_remove = self.point_items_right.pop()
            for item in items_to_remove:
                self.right_scene.removeItem(item)
            self.points_right_coords.pop()

            # Revert color of corresponding left point
            if self.point_items_left and len(self.point_items_left) > len(self.points_right_coords): 
                left_marker_items = self.point_items_left[len(self.points_right_coords)] 
                if left_marker_items and isinstance(left_marker_items[0], QGraphicsRectItem):
                    left_marker_items[0].setBrush(QBrush(POINT_COLOR_LEFT_PENDING))
                    left_marker_items[0].setPen(QPen(Qt.PenStyle.NoPen)) 

        elif self.point_items_left: 
            items_to_remove = self.point_items_left.pop()
            for item in items_to_remove:
                self.left_scene.removeItem(item)
            self.points_left_coords.pop()
        
        self._update_status_and_instructions()

    def _on_save_button_click(self):
        if not self.points_left_coords or len(self.points_left_coords) != len(self.points_right_coords):
            QMessageBox.warning(self, "Cannot Save", "Ensure all left points have corresponding right points, and there's at least one pair.")
            return
        if not self.points_left_coords: # Explicitly check for no points
            QMessageBox.warning(self, "Cannot Save", "No points have been annotated for this image pair.")
            return


        annotated_points_for_json = []
        disparities = []

        for i in range(len(self.points_left_coords)):
            lp = self.points_left_coords[i] # QPointF (center of pixel)
            rp = self.points_right_coords[i]# QPointF (center of pixel)
            
            # Disparity is calculated from the x-coordinates of the pixel centers
            disparity = lp.x() - rp.x() 
            disparities.append(disparity)

            annotated_points_for_json.append({
                "left_x_center": lp.x(), "left_y_center": lp.y(),
                "right_x_center": rp.x(), "right_y_center": rp.y(),
                "disparity": disparity
            })
        
        avg_disparity = np.mean(disparities) if disparities else 0.0
        std_dev_disparity = np.std(disparities) if disparities else 0.0

        annotation_data = {
            "image_pair_name": self.current_image_pair_name_global,
            "left_image": os.path.basename(self.current_left_img_path_global),
            "right_image": os.path.basename(self.current_right_img_path_global),
            "annotated_points": annotated_points_for_json, # Stores pixel center coordinates
            "avg_disparity": float(avg_disparity),
            "std_dev_disparity": float(std_dev_disparity),
            "num_points": len(self.points_left_coords)
        }

        annotation_filepath = os.path.join(self.annotations_dir, f"{self.current_image_pair_name_global}.json")
        try:
            with open(annotation_filepath, 'w') as f:
                json.dump(annotation_data, f, indent=4)
            print(f"Annotation saved to {annotation_filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save annotation: {e}")
            return

        self._load_next_unannotated_pair()

    def closeEvent(self, event):
        super().closeEvent(event)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ground Truth Disparity Annotation Tool")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing 'left' and 'right' subdirs with raw stereo pairs.")
    parser.add_argument("--calibration_file", type=str, required=True, help="Path to the stereo_calibration.npz file.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory to save ground truth JSON annotation files.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = AnnotationWindow(args.images_dir, args.calibration_file, args.annotations_dir)
    window.show()
    sys.exit(app.exec())