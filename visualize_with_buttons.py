#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine Visualization Tool with Interactive Button Bar
"""

import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

class MineVisualizerWithButtons:
    def __init__(self, base_path="landmine_final - Copy"):
        self.base_path = Path(base_path)
        self.colors = {
            'ap_metal': (0, 0, 255),      # Red
            'at_metal': (0, 165, 255),    # Orange
            'ap_plastic': (0, 255, 0),    # Green
            'at_plastic': (255, 0, 0)     # Blue
        }
        self.xml_files = []
        self.current_index = 0
        self.button_height = 50
        self.button_width = 74
        self.pending_action = None
        self.running = True
        self.window_name = "Mine Viewer"
        self.saved_marked_files = set()
        self.show_boxes = True
    
    def find_xml_files(self, limit=None):
        """Find XML files; if limit is None, include all files."""
        xml_files = list(self.base_path.rglob('*.xml'))
        sorted_files = sorted(xml_files)
        if limit is not None:
            sorted_files = sorted_files[:limit]
        self.xml_files = [str(f.relative_to(self.base_path)) for f in sorted_files]
        return len(self.xml_files)
    
    def parse_xml(self, xml_path):
        """Extract mine information from XML"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            mines = []
            for obj in root.findall('object'):
                mine_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                mines.append({
                    'type': mine_name,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'width': xmax - xmin,
                    'height': ymax - ymin,
                })
            return mines
        except (ET.ParseError, AttributeError, TypeError, ValueError) as exc:
            print(f"ERROR: Failed to parse XML {xml_path}: {exc}")
            return []
    
    def draw_mines_on_image(self, image_path, mines):
        """Draw mines on the image"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Cannot open image: {image_path}")
            return None
        
        img_with_boxes = img.copy()
        
        if self.show_boxes:
            for i, mine in enumerate(mines):
                color = self.colors.get(mine['type'], (255, 255, 255))
                cv2.rectangle(
                    img_with_boxes,
                    (mine['xmin'], mine['ymin']),
                    (mine['xmax'], mine['ymax']),
                    color,
                    2,
                )
                
                text = f"{i+1}. {mine['type']}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = mine['xmin']
                text_y = max(mine['ymin'] - 5, text_size[1] + 4)
                
                cv2.rectangle(
                    img_with_boxes,
                    (text_x, text_y - text_size[1] - 4),
                    (text_x + text_size[0] + 4, text_y),
                    color,
                    -1,
                )
                cv2.putText(
                    img_with_boxes,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
        
        self.add_legend(img_with_boxes)
        return img_with_boxes
    
    def add_legend(self, img):
        """Add color legend to image"""
        legend_y = 20
        legend_x = 20
        spacing = 25
        
        for mine_type, color in self.colors.items():
            cv2.rectangle(img,
                        (legend_x, legend_y),
                        (legend_x + 15, legend_y + 15),
                        color, -1)
            
            cv2.putText(img, mine_type,
                      (legend_x + 20, legend_y + 12),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.4, (255, 255, 255), 1)
            
            legend_y += spacing
    
    def create_button_bar(self, width, button_height=50):
        """Create button bar"""
        bar = np.zeros((button_height, width, 3), dtype=np.uint8)
        bar[:] = (50, 50, 50)
        
        buttons = []
        button_configs = [
            ("FIRST", 10, 0xFF6B6B),
            ("PREV", 92, 0xFF8E53),
            ("BOXES", 174, 0xB784F7),
            ("LIST", 256, 0xFFC93C),
            ("NEXT", 338, 0x6BCF7F),
            ("LAST", 420, 0x4D96FF),
            ("EXIT", 502, 0xA8A8A8),
        ]
        
        for label, x_pos, color in button_configs:
            b = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            r = color & 0xFF
            
            x_end = x_pos + self.button_width
            cv2.rectangle(bar, (x_pos, 5), (x_end, button_height - 5), 
                         (b, g, r), -1)
            cv2.rectangle(bar, (x_pos, 5), (x_end, button_height - 5), 
                         (255, 255, 255), 2)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = x_pos + (self.button_width - text_size[0]) // 2
            text_y = (button_height + text_size[1]) // 2
            
            cv2.putText(bar, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            buttons.append((x_pos, x_end, 5, button_height - 5, label))
        
        return bar, buttons
    
    def mouse_callback(self, event, x, y, flags, buttons):
        """Handle button clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            for x_min, x_max, y_min, y_max, label in buttons:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    self.pending_action = label
                    return
    
    def show_file_list(self):
        """Show file list"""
        print("\n" + "="*70)
        print("FILE LIST:")
        print("="*70)
        
        start = max(0, self.current_index - 5)
        end = min(len(self.xml_files), self.current_index + 6)
        
        for i in range(start, end):
            if i == self.current_index:
                print(f"  >>> {i+1:4d}. {self.xml_files[i]} <-- CURRENT")
            else:
                print(f"      {i+1:4d}. {self.xml_files[i]}")
        
        print("="*70)
    
    def apply_action(self, action):
        """Apply UI action; return True if frame should be redrawn."""
        if action == "FIRST":
            if self.current_index != 0:
                self.current_index = 0
                return True
            return False
        
        if action == "PREV":
            if self.current_index > 0:
                self.current_index -= 1
                return True
            return False
        
        if action == "NEXT":
            if self.current_index < len(self.xml_files) - 1:
                self.current_index += 1
                return True
            return False
        
        if action == "LAST":
            last_index = len(self.xml_files) - 1
            if self.current_index != last_index:
                self.current_index = last_index
                return True
            return False
        
        if action == "LIST":
            self.show_file_list()
            return False
        
        if action == "BOXES":
            self.show_boxes = not self.show_boxes
            print(f"BOXES: {'ON' if self.show_boxes else 'OFF'}")
            return True
        
        if action == "EXIT":
            self.running = False
            return False
        
        return False
    
    def build_current_frame(self):
        """Build display frame and metadata for current index."""
        xml_file = self.xml_files[self.current_index]
        xml_path = self.base_path / xml_file
        
        if not xml_path.exists():
            print(f"ERROR: Missing XML file: {xml_path}")
            return None
        
        try:
            root = ET.parse(xml_path).getroot()
            image_filename = root.find("filename").text
        except (ET.ParseError, AttributeError, TypeError) as exc:
            print(f"ERROR: Invalid XML header in {xml_path}: {exc}")
            return None
        
        image_path = xml_path.parent / image_filename
        if not image_path.exists():
            print(f"ERROR: Missing image file: {image_path}")
            return None
        
        mines = self.parse_xml(xml_path)
        image_with_boxes = self.draw_mines_on_image(str(image_path), mines)
        if image_with_boxes is None:
            return None
        
        output_file = f"marked_{Path(xml_file).stem}.jpg"
        output_path = xml_path.parent / output_file
        if str(output_path) not in self.saved_marked_files:
            cv2.imwrite(str(output_path), image_with_boxes)
            self.saved_marked_files.add(str(output_path))
        
        return {
            "xml_file": xml_file,
            "image_filename": image_filename,
            "mines": mines,
            "image": image_with_boxes,
        }
    
    def visualize_with_buttons(self):
        """Display images with button bar"""
        if not self.xml_files:
            print("ERROR: No XML files found")
            return
        
        print(f"\nFOUND: {len(self.xml_files)} XML files")
        print("USE: Click buttons or press keys to navigate")
        print("="*70 + "\n")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 1000)
        needs_redraw = True
        last_rendered_index = -1
        
        while self.running:
            if needs_redraw or last_rendered_index != self.current_index:
                frame_data = self.build_current_frame()
                if frame_data is None:
                    if self.current_index < len(self.xml_files) - 1:
                        self.current_index += 1
                        continue
                    if self.current_index > 0:
                        self.current_index -= 1
                        continue
                    print("ERROR: Could not render any valid frame.")
                    break
                
                button_bar, buttons = self.create_button_bar(frame_data["image"].shape[1])
                composed = np.vstack([button_bar, frame_data["image"]])
                try:
                    cv2.imshow(self.window_name, composed)
                    cv2.setMouseCallback(self.window_name, self.mouse_callback, buttons)
                except cv2.error as exc:
                    print(f"ERROR: OpenCV display error: {exc}")
                    break
                
                mines = frame_data["mines"]
                print(f"{'='*70}")
                print(f"FILE: {self.current_index + 1}/{len(self.xml_files)}")
                print(f"XML:  {frame_data['xml_file']}")
                print(f"IMG:  {frame_data['image_filename']}")
                print(f"MINES: {len(mines)}")
                print(f"BOXES: {'ON' if self.show_boxes else 'OFF'}")
                print(f"{'='*70}")
                
                last_rendered_index = self.current_index
                needs_redraw = False
            
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n"):
                needs_redraw = self.apply_action("NEXT") or needs_redraw
            elif key == ord("p"):
                needs_redraw = self.apply_action("PREV") or needs_redraw
            elif key == ord("f"):
                needs_redraw = self.apply_action("FIRST") or needs_redraw
            elif key == ord("l"):
                needs_redraw = self.apply_action("LAST") or needs_redraw
            elif key == ord("s"):
                self.apply_action("LIST")
            elif key == ord("b"):
                needs_redraw = self.apply_action("BOXES") or needs_redraw
            elif key == ord("g"):
                try:
                    num = int(input(f"\nGo to file number (1-{len(self.xml_files)}): "))
                    if 1 <= num <= len(self.xml_files):
                        self.current_index = num - 1
                        needs_redraw = True
                    else:
                        print("ERROR: Invalid file number.")
                except ValueError:
                    print("ERROR: Please enter a valid number.")
            elif key == ord("q") or key == 27:
                self.running = False
            
            if self.pending_action is not None:
                action = self.pending_action
                self.pending_action = None
                needs_redraw = self.apply_action(action) or needs_redraw
        
        cv2.destroyAllWindows()
        print("\nThank you for using the Mine Viewer!")

def main():
    visualizer = MineVisualizerWithButtons()
    
    print("\n" + "="*70)
    print("MINE VISUALIZATION TOOL WITH INTERACTIVE BUTTON BAR")
    print("="*70)
    
    # Find XML files
    count = visualizer.find_xml_files()
    print(f"FOUND: {count} XML files\n")
    
    if count > 0:
        # Display help
        print("INSTRUCTIONS:")
        print("  MOUSE: Click buttons at the top to navigate")
        print("  KEYBOARD SHORTCUTS:")
        print("    [n] Next file")
        print("    [p] Previous file")
        print("    [f] First file")
        print("    [l] Last file")
        print("    [b] Show/Hide mine boxes")
        print("    [s] Show file list")
        print("    [g] Go to specific file number")
        print("    [q] or [ESC] Exit")
        print("="*70 + "\n")
        
        visualizer.visualize_with_buttons()

if __name__ == '__main__':
    main()
