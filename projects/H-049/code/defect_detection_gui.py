#!/usr/bin/env python3
"""
GUI for Defect Detection

A nice-looking graphical interface for running defect detection on microscopy images.
Allows users to:
- Open an image file
- Run defect detection
- View results with bounding boxes overlaid
- See timing statistics

Usage:
    python defect_detection_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Import functions from main.py
from main import load_images, find_defects_yolo
from ultralytics import YOLO
from pathlib import Path
import os


class DefectDetectionGUI:
    def __init__(self, root, debug_mode=False, model_path=None, pixel_width=22.8):
        self.root = root
        self.root.title("Defect Detection" + (" [DEBUG MODE]" if debug_mode else ""))
        self.root.geometry("1200x800")

        # Debug mode flag
        self.debug_mode = debug_mode

        # Scale bar configuration
        self.pixel_width = pixel_width  # nanometers per pixel

        # Load YOLO model
        if model_path is None:
            model_path = Path(os.getcwd()) / "trained_models" / "yolov11_obb" / "model_4" / "weights" / "best.pt"

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("YOLO model loaded successfully")

        # Data storage
        self.image_path = None
        self.original_image = None
        self.bounding_boxes = None
        self.detection_time = 0
        self.num_defects = 0

        # Zoom state
        self.zoom_level = 1.0
        self.base_xlim = None
        self.base_ylim = None

        # Pan state
        self.panning = False
        self.pan_start = None

        # Scale bar artists
        self.scale_bar = None
        self.scale_bar_outline = None
        self.scale_text = None

        # Configure style
        self.setup_styles()

        # Create UI
        self.create_widgets()

    def setup_styles(self):
        """Configure ttk styles for a modern look."""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        bg_color = '#f0f0f0'
        accent_color = '#0066cc'

        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Helvetica', 10))
        style.configure('Status.TLabel', font=('Helvetica', 11), foreground=accent_color)
        style.configure('Action.TButton', font=('Helvetica', 10, 'bold'), padding=10)

    def create_widgets(self):
        """Create all GUI widgets."""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Top control panel
        self.create_control_panel(main_frame)

        # Image display area
        self.create_display_area(main_frame)

        # Bottom status bar
        self.create_status_bar(main_frame)

    def create_control_panel(self, parent):
        """Create the top control panel with buttons and settings."""

        control_frame = ttk.Frame(parent, padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Title
        title_label = ttk.Label(control_frame, text="Defect Detection System",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))

        # Instructions
        instructions = ttk.Label(control_frame,
                               text="💡 Mouse wheel: zoom | Click & drag: pan | Press 'R': reset view",
                               style='Subtitle.TLabel')
        instructions.grid(row=1, column=0, columnspan=3, pady=(0, 10))

        # File selection
        ttk.Label(control_frame, text="Image File:").grid(row=2, column=0, sticky=tk.W, padx=5)

        self.file_entry = ttk.Entry(control_frame, width=60)
        self.file_entry.grid(row=2, column=1, padx=5, sticky=(tk.W, tk.E))

        browse_btn = ttk.Button(control_frame, text="Browse...",
                               command=self.browse_file, style='Action.TButton')
        browse_btn.grid(row=2, column=2, padx=5)

        # Run detection button
        self.detect_btn = ttk.Button(control_frame, text="Run Defect Detection",
                                     command=self.run_detection, style='Action.TButton',
                                     state='disabled')
        self.detect_btn.grid(row=3, column=2, padx=5, pady=(10, 0))

        # Debug mode controls
        if self.debug_mode:
            # Load bounding boxes button (debug only)
            self.load_bbox_btn = ttk.Button(control_frame, text="Load Bounding Boxes (Debug)",
                                           command=self.load_bounding_boxes, style='Action.TButton',
                                           state='disabled')
            self.load_bbox_btn.grid(row=4, column=2, padx=5, pady=(5, 0))

            # Debug info label
            debug_label = ttk.Label(control_frame, text="🐛 Debug Mode: Load ground truth labels",
                                   style='Subtitle.TLabel', foreground='orange')
            debug_label.grid(row=4, column=0, columnspan=2, pady=(5, 0), sticky=tk.W)

        # Configure column weights
        control_frame.columnconfigure(1, weight=1)

    def create_display_area(self, parent):
        """Create the image display area."""

        display_frame = ttk.LabelFrame(parent, text="Image Display", padding="10")
        display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Create matplotlib figure with two subplots
        self.fig, (self.ax, self.ax_bar) = plt.subplots(1, 2, figsize=(14, 7),
                                                         gridspec_kw={'width_ratios': [3, 1]})

        # Setup image axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, 'No image loaded\nClick "Browse..." to select an image',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=self.ax.transAxes)

        # Setup bar chart axis
        self.setup_bar_chart()

        # Adjust layout
        self.fig.tight_layout(pad=2.0)

        # Embed matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bind mouse wheel for zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Bind mouse for pan
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Bind keyboard for zoom reset
        self.root.bind('<r>', self.reset_zoom)
        self.root.bind('<R>', self.reset_zoom)

    def setup_bar_chart(self):
        """Setup the bar chart for defect class counts."""
        self.ax_bar.clear()
        self.ax_bar.set_title('Defect Counts by Class', fontsize=10, pad=10)
        self.ax_bar.set_xlabel('Class', fontsize=9)
        self.ax_bar.set_ylabel('Count', fontsize=9)
        self.ax_bar.set_xticks([0, 1, 2])
        self.ax_bar.set_xticklabels(['Grain\nBoundary', 'Vacancy', 'Interstitial'])
        self.ax_bar.set_ylim([0, 1])
        self.ax_bar.grid(axis='y', alpha=0.3)

    def update_bar_chart(self, bboxes):
        """Update the bar chart with defect class counts."""
        # Count defects by class
        class_counts = {0: 0, 1: 0, 2: 0}
        if bboxes is not None:
            for box in bboxes:
                cls = int(box[0])
                if cls in class_counts:
                    class_counts[cls] += 1

        # Update bar chart
        self.ax_bar.clear()
        classes = [0, 1, 2]
        counts = [class_counts[0], class_counts[1], class_counts[2]]
        colors = ['magenta', 'orange', 'cyan']
        labels = ['Grain\nBoundary', 'Vacancy', 'Interstitial']

        bars = self.ax_bar.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')

        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if count > 0:
                self.ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(count)}',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')

        self.ax_bar.set_title('Defect Counts by Class', fontsize=15, pad=10)
        self.ax_bar.set_xlabel('Class', fontsize=9)
        self.ax_bar.set_ylabel('Count', fontsize=9)
        self.ax_bar.set_xticks(classes)
        self.ax_bar.set_xticklabels(labels, fontsize=8)

        # Set y-axis limit with some headroom
        max_count = max(counts) if max(counts) > 0 else 1
        self.ax_bar.set_ylim([0, max_count * 1.2])
        self.ax_bar.grid(axis='y', alpha=0.3)

    def create_status_bar(self, parent):
        """Create the bottom status bar."""

        status_frame = ttk.Frame(parent, padding="5")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_label = ttk.Label(status_frame,
                                      text="Ready. Please select an image to begin.",
                                      style='Status.TLabel')
        self.status_label.grid(row=0, column=0, sticky=tk.W)

    def browse_file(self):
        """Open file dialog to select an image."""

        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.tif *.tiff *.png *.jpg *.jpeg"),
                ("TIFF Files", "*.tif *.tiff"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("All Files", "*.*")
            ],
            initialdir="/home/jed/git/2025_Hackathon"
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """Load and display an image."""

        try:
            self.image_path = file_path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

            # Load image
            self.original_image = load_images(file_path)

            # Display image
            self.display_image(self.original_image, title="Original Image")

            # Enable detection button
            self.detect_btn.config(state='normal')

            # Enable debug buttons if in debug mode
            if self.debug_mode:
                self.load_bbox_btn.config(state='normal')

            # Update status
            h, w = self.original_image.shape
            self.status_label.config(
                text=f"Loaded: {file_path.split('/')[-1]} ({w}×{h} pixels)"
            )

        except Exception as e:
            messagebox.showerror("Error Loading Image",
                               f"Failed to load image:\n{str(e)}")
            self.status_label.config(text="Error loading image")

    def display_image(self, image, bboxes=None, title="Image"):
        """Display image with optional bounding boxes."""

        self.ax.clear()
        self.ax.imshow(image, cmap='gray', interpolation='nearest')
        self.ax.set_title(title, fontsize=18, pad=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Set aspect ratio to 'equal' to prevent distortion
        self.ax.set_aspect('equal', adjustable='box')

        # Store base limits for zoom
        h, w = image.shape
        self.base_xlim = (-0.5, w - 0.5)
        self.base_ylim = (h - 0.5, -0.5)
        self.zoom_level = 1.0

        # Add bounding boxes if provided
        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                cls = box[0]
                x = list(box[1::2])
                y = list(box[2::2])
                if cls == 0:
                    color = 'magenta'  # grain boundary
                elif cls == 1:
                    color = 'orange' # vacancy
                elif cls == 2:
                    color = 'cyan' # interstitial
                self.ax.plot(
                    x + [x[0]],
                    y + [y[0]],
                    color=color, linewidth=1)

        # Update bar chart with class counts
        self.update_bar_chart(bboxes)
        self.ax.set_xlim(self.base_xlim)
        self.ax.set_ylim(self.base_ylim)

        # Add scale bar
        self.update_scale_bar()

        self.canvas.draw()

    def update_scale_bar(self):
        """Update the scale bar based on current zoom level."""
        if self.original_image is None:
            return

        # Remove old scale bar elements if they exist
        if self.scale_bar is not None:
            self.scale_bar.remove()
            self.scale_bar = None
        if self.scale_bar_outline is not None:
            self.scale_bar_outline.remove()
            self.scale_bar_outline = None
        if self.scale_text is not None:
            self.scale_text.remove()
            self.scale_text = None

        # Get current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate visible width in pixels
        visible_width_pixels = abs(xlim[1] - xlim[0])

        # Calculate visible width in nanometers
        visible_width_nm = visible_width_pixels * self.pixel_width

        # Choose appropriate scale bar length (aim for ~20% of visible width)
        target_length_nm = visible_width_nm * 0.2

        # Round to nice values
        if target_length_nm >= 1000000:  # >= 1mm
            # Use mm
            scale_options = [1000000, 2000000, 5000000, 10000000]
            unit = 'mm'
            divisor = 1000000
        elif target_length_nm >= 1000:  # >= 1µm
            # Use µm
            scale_options = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
            unit = 'µm'
            divisor = 1000
        else:  # < 1µm
            # Use nm
            scale_options = [10, 20, 50, 100, 200, 500]
            unit = 'nm'
            divisor = 1

        # Find closest scale option
        scale_length_nm = min(scale_options, key=lambda x: abs(x - target_length_nm))
        scale_length_pixels = scale_length_nm / self.pixel_width

        # Position scale bar in bottom-right corner
        margin_x = visible_width_pixels * 0.05  # 5% margin
        margin_y = abs(ylim[1] - ylim[0]) * 0.05  # 5% margin

        # Scale bar coordinates
        x_start = xlim[1] - margin_x - scale_length_pixels
        x_end = xlim[1] - margin_x
        y_pos = ylim[0] - margin_y  # Bottom (note: y-axis is inverted)

        # Draw black outline first
        self.scale_bar_outline = self.ax.plot(
            [x_start, x_end],
            [y_pos, y_pos],
            color='black',
            linewidth=5,
            solid_capstyle='butt'
        )[0]

        # Draw white bar on top
        self.scale_bar = self.ax.plot(
            [x_start, x_end],
            [y_pos, y_pos],
            color='white',
            linewidth=3,
            solid_capstyle='butt',
            zorder=self.scale_bar_outline.get_zorder() + 1
        )[0]

        # Add text label
        scale_value = scale_length_nm / divisor
        if scale_value.is_integer():
            label = f'{int(scale_value)} {unit}'
        else:
            label = f'{scale_value:.1f} {unit}'

        # Position text above the scale bar
        text_y = y_pos - abs(ylim[1] - ylim[0]) * 0.03

        self.scale_text = self.ax.text(
            (x_start + x_end) / 2,
            text_y,
            label,
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='none', alpha=0.7)
        )

    def on_scroll(self, event):
        """Handle mouse wheel scroll for zoom."""

        if self.original_image is None or self.base_xlim is None:
            return

        # Get current axis limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            # Mouse outside axes
            return

        # Zoom factor
        zoom_factor = 1.2 if event.button == 'up' else 1 / 1.2

        # Calculate new limits centered on mouse position
        # For x-axis
        x_range = cur_xlim[1] - cur_xlim[0]
        new_x_range = x_range / zoom_factor

        # Calculate how far the mouse is from the left edge (0 to 1)
        x_fraction = (xdata - cur_xlim[0]) / x_range
        new_xlim = [xdata - new_x_range * x_fraction,
                    xdata + new_x_range * (1 - x_fraction)]

        # For y-axis
        y_range = cur_ylim[1] - cur_ylim[0]
        new_y_range = y_range / zoom_factor

        # Calculate how far the mouse is from the bottom edge (0 to 1)
        y_fraction = (ydata - cur_ylim[0]) / y_range
        new_ylim = [ydata - new_y_range * y_fraction,
                    ydata + new_y_range * (1 - y_fraction)]

        # Constrain to image bounds
        base_x_range = self.base_xlim[1] - self.base_xlim[0]
        base_y_range = self.base_ylim[1] - self.base_ylim[0]

        # Don't allow zooming out beyond minimum zoom level
        if new_x_range > base_x_range:
            # At maximum zoom out, center the image in the view
            new_xlim = self.base_xlim
        else:
            # Zoomed in - constrain panning to image bounds
            if new_xlim[0] < self.base_xlim[0]:
                new_xlim[0] = self.base_xlim[0]
            if new_xlim[1] > self.base_xlim[1]:
                new_xlim[1] = self.base_xlim[1]
        new_x_range = new_xlim[1] - new_xlim[0]

        if new_y_range > abs(base_y_range):
            # At maximum zoom out, center the image in the view
            new_ylim = self.base_ylim
        elif new_y_range < abs(base_y_range):
            # Zoomed in - constrain panning to image bounds (note: y-axis is inverted)
            if new_ylim[0] > self.base_ylim[0]:
                new_ylim[0] = self.base_ylim[0]
            if new_ylim[1] < self.base_ylim[1]:
                new_ylim[1] = self.base_ylim[1]
        new_y_range = new_ylim[1] - new_ylim[0]

        # Apply new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        # Update scale bar for new zoom level
        self.update_scale_bar()

        self.canvas.draw()

        # Update zoom level for display
        self.zoom_level = base_x_range / (new_xlim[1] - new_xlim[0])

        # Update status bar with zoom level
        if hasattr(self, 'num_defects') and self.num_defects > 0:
            status_text = f"Detected {self.num_defects} defects in your image in {self.detection_time:.1f}ms | Zoom: {self.zoom_level:.1f}x"
        elif self.original_image is not None:
            h, w = self.original_image.shape
            status_text = f"Loaded: {self.image_path.split('/')[-1]} ({w}×{h} pixels) | Zoom: {self.zoom_level:.1f}x"
        else:
            status_text = f"Zoom: {self.zoom_level:.1f}x"
        self.status_label.config(text=status_text)

    def reset_zoom(self, event=None):
        """Reset zoom to original view."""

        if self.base_xlim is None or self.base_ylim is None:
            return

        self.ax.set_xlim(self.base_xlim)
        self.ax.set_ylim(self.base_ylim)
        self.zoom_level = 1.0

        # Update scale bar for reset view
        self.update_scale_bar()

        self.canvas.draw()

        # Update status bar
        if hasattr(self, 'num_defects') and self.num_defects > 0:
            status_text = f"Detected {self.num_defects} defects in your image in {self.detection_time:.1f}ms"
        elif self.original_image is not None:
            h, w = self.original_image.shape
            status_text = f"Loaded: {self.image_path.split('/')[-1]} ({w}×{h} pixels)"
        else:
            status_text = "Ready. Please select an image to begin."
        self.status_label.config(text=status_text)

    def on_press(self, event):
        """Handle mouse button press to start panning."""

        if self.original_image is None or self.base_xlim is None:
            return

        # Only pan with left mouse button
        if event.button != 1:
            return

        # Check if click is inside axes
        if event.inaxes != self.ax:
            return

        # Start panning
        self.panning = True
        self.pan_start = (event.xdata, event.ydata)

        # Change cursor to indicate panning mode
        self.canvas.get_tk_widget().config(cursor="fleur")

    def on_release(self, event):
        """Handle mouse button release to stop panning."""

        if not self.panning:
            return

        # Stop panning
        self.panning = False
        self.pan_start = None

        # Restore cursor
        self.canvas.get_tk_widget().config(cursor="")

    def on_motion(self, event):
        """Handle mouse motion for panning."""

        if not self.panning or self.pan_start is None:
            return

        # Check if still inside axes
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Calculate pan delta
        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata

        # Get current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Calculate new limits
        new_xlim = [cur_xlim[0] + dx, cur_xlim[1] + dx]
        new_ylim = [cur_ylim[0] + dy, cur_ylim[1] + dy]

        # Get base ranges
        base_x_range = self.base_xlim[1] - self.base_xlim[0]
        base_y_range = self.base_ylim[1] - self.base_ylim[0]

        cur_x_range = cur_xlim[1] - cur_xlim[0]
        cur_y_range = cur_ylim[1] - cur_ylim[0]

        # Only constrain panning when zoomed in (prevents panning outside image)
        if cur_x_range < base_x_range:
            # Zoomed in - constrain to image bounds
            if new_xlim[0] < self.base_xlim[0]:
                shift = self.base_xlim[0] - new_xlim[0]
                new_xlim[0] += shift
                new_xlim[1] += shift
            if new_xlim[1] > self.base_xlim[1]:
                shift = new_xlim[1] - self.base_xlim[1]
                new_xlim[0] -= shift
                new_xlim[1] -= shift

        if cur_y_range < abs(base_y_range):
            # Zoomed in - constrain to image bounds (note: y-axis is inverted)
            if new_ylim[0] > self.base_ylim[0]:
                shift = new_ylim[0] - self.base_ylim[0]
                new_ylim[0] -= shift
                new_ylim[1] -= shift
            if new_ylim[1] < self.base_ylim[1]:
                shift = self.base_ylim[1] - new_ylim[1]
                new_ylim[0] += shift
                new_ylim[1] += shift

        # Apply new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        # Update scale bar for new view
        self.update_scale_bar()

        self.canvas.draw()

    def load_bounding_boxes(self):
        """Load bounding boxes from a text file (debug mode only)."""

        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Open file dialog to select bounding box file
        bbox_file = filedialog.askopenfilename(
            title="Select Bounding Box File",
            filetypes=[
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ],
            initialdir="/home/jed/git/2025_Hackathon"
        )

        if not bbox_file:
            return

        try:
            # Parse bounding box file
            bboxes = []
            with open(bbox_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    # Parse the line: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    parts = line.split()
                    if len(parts) >= 9:
                        # Convert to floats/ints
                        class_id = int(float(parts[0]))
                        coords = [float(p) * self.original_image.shape[i % 2] for i, p in enumerate(parts[1:9])]

                        # Create bbox tuple: (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
                        bbox = (class_id, *coords)
                        bboxes.append(bbox)

            # Store loaded bounding boxes
            self.bounding_boxes = bboxes
            self.num_defects = len(bboxes)
            self.detection_time = 0  # No detection was run

            # Display results
            title = f"Loaded Bounding Boxes from File"
            self.display_image(self.original_image, self.bounding_boxes, title=title)

            # Count by class for status message
            class_counts = {0: 0, 1: 0, 2: 0}
            for box in self.bounding_boxes:
                cls = int(box[0])
                if cls in class_counts:
                    class_counts[cls] += 1

            # Update status
            status_text = f"Loaded {self.num_defects} bounding boxes from file (GB:{class_counts[0]}, V:{class_counts[1]}, I:{class_counts[2]})"
            self.status_label.config(text=status_text)

            # Show info dialog
            messagebox.showinfo(
                "Bounding Boxes Loaded",
                f"Loaded {self.num_defects} bounding boxes from:\n{bbox_file.split('/')[-1]}\n\n"
                f"  - Grain Boundary: {class_counts[0]}\n"
                f"  - Vacancy: {class_counts[1]}\n"
                f"  - Interstitial: {class_counts[2]}\n"
            )

        except Exception as e:
            messagebox.showerror("Error Loading Bounding Boxes",
                               f"Failed to load bounding boxes:\n{str(e)}")
            self.status_label.config(text="Error loading bounding boxes")

    def run_detection(self):
        """Run defect detection on the loaded image."""

        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            # Update status
            self.status_label.config(text="Running YOLO defect detection...")
            self.root.update()

            # Time the detection
            start_time = time.time()

            # Find defects using YOLO
            self.bounding_boxes = find_defects_yolo(self.model, self.original_image)

            # Calculate timing
            end_time = time.time()
            self.detection_time = (end_time - start_time) * 1000  # Convert to ms
            self.num_defects = len(self.bounding_boxes)

            # Display results
            self.display_results()

        except Exception as e:
            messagebox.showerror("Detection Error",
                               f"Failed to run detection:\n{str(e)}")
            self.status_label.config(text="Error during detection")

    def display_results(self):
        """Display detection results with overlay."""

        # Display image with bounding boxes
        title = f"Defect Detection Results"
        self.display_image(self.original_image, self.bounding_boxes, title=title)

        # Count defects by class
        class_counts = {0: 0, 1: 0, 2: 0}
        for box in self.bounding_boxes:
            cls = int(box[0])
            if cls in class_counts:
                class_counts[cls] += 1

        # Update status with timing info
        status_text = f"Detected {self.num_defects} defects in your image in {self.detection_time:.1f}ms"
        self.status_label.config(text=status_text)

        # Show info dialog with results
        messagebox.showinfo(
            "Detection Complete",
            f"Detection completed successfully!\n\n"
            f"Total defects: {self.num_defects}\n"
            f"  - Grain Boundary: {class_counts[0]}\n"
            f"  - Vacancy: {class_counts[1]}\n"
            f"  - Interstitial: {class_counts[2]}\n\n"
            f"Processing time: {self.detection_time:.1f} ms\n"
        )


def main():
    """Main entry point for the GUI application."""
    import sys
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Defect Detection GUI')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (allows loading ground truth bounding box files)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to YOLO model file (default: trained_models/yolov11_obb/model_4/weights/best.pt)')
    parser.add_argument('--pixel-width', type=float, default=22.8,
                       help='Pixel width in nanometers for scale bar (default: 22.8 nm)')
    args = parser.parse_args()

    root = tk.Tk()
    app = DefectDetectionGUI(root, debug_mode=args.debug, model_path=args.model, pixel_width=args.pixel_width)
    root.mainloop()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
