import sys
import numpy as np
import numba
import scipy.ndimage
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

# Assuming leniasetup.py is in the same directory
try:
    from leniasetup import give_conv_from_name, pattern
except ImportError:
    print("Error: leniasetup.py not found or contains errors.")
    print("Please ensure leniasetup.py is in the same directory and is correct.")
    sys.exit(1)


# Your Lenia helper functions (bell, growth, update) remain the same.
@numba.jit(nopython=True)
def bell(x, m, s):
    return np.exp(-((x - m) / s) ** 2 / 2)

@numba.jit(nopython=True)
def growth(U, m, s):
    return (bell(U, m, s) * 2 - 1)

@numba.jit(nopython=True, parallel=False)
def update(imgarr, fKs, kernelms, _lenia_size, T): # Renamed 'size' to '_lenia_size'
    total = np.zeros((_lenia_size, _lenia_size), dtype=np.float32) # Use _lenia_size
    for i in range(len(fKs)):
        fft_input = imgarr.astype(np.complex64)
        field = np.real(np.fft.ifft2(fKs[i] * np.fft.fft2(fft_input)))
        total += growth(field, kernelms[i][0], kernelms[i][1])
    
    imgarr = np.clip(imgarr + 1/T * total/len(fKs), 0, 1)
    return imgarr.astype(np.float32)

@numba.jit(nopython=True)
def splash(imgarr, center_y, center_x, radius=20, intensity=0.5):
    """Adds a splash effect to imgarr at the correct position."""
    # Ensure center_x, center_y are integers
    center_x = int(center_x)
    center_y = int(center_y)

    rows, cols = imgarr.shape
    
    # Create meshgrid using Numba-compatible arange and reshape
    y_coords = np.arange(rows).reshape(-1, 1)
    x_coords = np.arange(cols).reshape(1, -1)
    
    # Calculate squared distance from the center
    dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
    
    # Calculate mask using the inverse distance function
    mask = (1 + dist_sq / (radius**2)) ** -1.2
    
    imgarr += intensity * mask
    return np.clip(imgarr, 0, 1)

@numba.jit(nopython=True)
def add_pulse(imgarr, time, frequency=0.005, speed=0.02, intensity=0.2, direction="horizontal"):
    """Adds a sweeping periodic pulse to the simulation."""
    _lenia_size = imgarr.shape[0]  # Assuming square grid
    t = time * speed  # Time-based shift

    # Generate sinusoidal pulse
    if direction == "horizontal":
        pulse = np.sin(2 * np.pi * frequency * (np.arange(_lenia_size) - t)).reshape(1, -1)
    elif direction == "vertical":
        pulse = np.sin(2 * np.pi * frequency * (np.arange(_lenia_size) - t)).reshape(-1, 1)

    # Normalize pulse to be between 0 and 1
    pulse = (pulse + 1) / 2  

    # Add the pulse to the image
    imgarr += intensity * pulse * np.random.normal(1,0.02,imgarr.shape )
    return np.clip(imgarr, 0, 1)


class LeniaVisualizer2D(QtWidgets.QMainWindow): # Inherit from QMainWindow for better layout
    def __init__(self, _lenia_size=512, scale=2): # Renamed 'size' to '_lenia_size'
        """
        Initialize the graphics window, 2D image view, and Lenia simulation.
        """
        super().__init__()
        
        # --- Basic application setup ---
        self.setWindowTitle('Lenia 2D Visualizer')
        self.setGeometry(100, 100, 1000, 800) # Increased window size
        
        # Main widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # --- ImageView Widget ---
        self.view = pg.ImageView()
        # Use a built-in colormap (e.g., 'magma', 'viridis', 'inferno')
        colormap = pg.colormap.get('magma')
        # You could define a custom colormap like this if pridepy could export color points:
        # colors = np.array([[0,0,0,255], [..your pridepy colors as RGB tuples..], [255,255,255,255]])
        # pos = np.linspace(0.0, 1.0, len(colors))
        # custom_cmap = pg.ColorMap(pos, colors)
        # self.view.setColorMap(custom_cmap)

        self.view.setColorMap(colormap)
        
        # Hide the colorbar histogram
        self.view.ui.histogram.hide()

        # Disable default mouse interaction (panning/zooming) for the ImageView
        self.view.getView().setMouseEnabled(x=False, y=False)


        # --- FPS Label (moved to control panel) ---
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: white; font-size: 14pt;") # Style for visibility

        # --- Lenia Text (will be drawn into imgarr) ---
        self.text_content = "LENIA"
        self.text_font_size = 72 # Large font for visibility
        # Initial text position (centered) - will be calculated after mask is generated
        self.text_x = 0
        self.text_y = 0
        self.vel_x, self.vel_y = 2, 2 # Initial velocity for bouncing
        self.text_intensity = 0.0 # Will be controlled by sin wave
        self.text_mask = None # Store the rendered text mask
        
        # --- Lenia Simulation Setup ---
        self._lenia_size = _lenia_size # Store the Lenia grid size
        self.scale = scale
        self.mid = self._lenia_size // 2 # Use _lenia_size
        
        # Default creature and parameters (initialized early)
        self.animal_name = "orbium"
        self.current_creature_data = None
        self.R = 0
        self.T = 0
        self.kernels = []
        self.cells = []

        # Mouse interaction flags
        self.mouse_held = False
        
        # Feature flags
        self.pulse_enabled = False
        self.sign_enabled = False # for the bouncing text
        self.text_pulse_enabled = False # New flag for text intensity pulse

        # --- Control Panel (GUI) ---
        # Initialize control panel widgets BEFORE calling load_creature
        self.control_panel = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QGridLayout(self.control_panel)
        
        # R input
        self.r_label = QtWidgets.QLabel("R:")
        self.r_input = QtWidgets.QLineEdit(str(self.R))
        self.r_input.setValidator(pg.QtGui.QDoubleValidator()) # Allow only numbers
        self.r_input.editingFinished.connect(self.update_lenia_params)
        
        self.control_layout.addWidget(self.r_label, 0, 0)
        self.control_layout.addWidget(self.r_input, 0, 1)

        # Creature name input
        self.creature_label = QtWidgets.QLabel("Creature:")
        self.creature_input = QtWidgets.QLineEdit(self.animal_name)
        self.creature_input.editingFinished.connect(self.load_new_creature_from_input)
        
        self.control_layout.addWidget(self.creature_label, 1, 0)
        self.control_layout.addWidget(self.creature_input, 1, 1)

        # Load Creature Button
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.clicked.connect(self.load_new_creature_from_input)
        self.control_layout.addWidget(self.load_button, 1, 2)

        # SinBack Button
        self.sinback_button = QtWidgets.QPushButton("SinBack OFF")
        self.sinback_button.clicked.connect(self.toggle_sinback)
        self.control_layout.addWidget(self.sinback_button, 2, 0, 1, 3) # Span 3 columns

        # Text Toggle Button
        self.text_toggle_button = QtWidgets.QPushButton("Text OFF")
        self.text_toggle_button.clicked.connect(self.toggle_text_display)
        self.control_layout.addWidget(self.text_toggle_button, 3, 0, 1, 3)

        # Text Pulse Toggle Button
        self.text_pulse_button = QtWidgets.QPushButton("Text Pulse OFF")
        self.text_pulse_button.clicked.connect(self.toggle_text_pulse)
        self.control_layout.addWidget(self.text_pulse_button, 4, 0, 1, 3)
        
        # Reset Button
        self.reset_button = QtWidgets.QPushButton("Reset Simulation")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.control_layout.addWidget(self.reset_button, 5, 0, 1, 3)

        # Clear Button
        self.clear_button = QtWidgets.QPushButton("Clear Grid")
        self.clear_button.clicked.connect(self.clear_grid)
        self.control_layout.addWidget(self.clear_button, 6, 0, 1, 3)

        # Add FPS label to control panel
        self.control_layout.addWidget(self.fps_label, 7, 0, 1, 3) # Row 7, span 3 columns

        self.control_layout.setRowStretch(8, 1) # Push widgets to top

        # Now call load_creature, as GUI elements are ready
        self.load_creature(self.animal_name) # Initial load

        # Set the initial image and fix the levels to prevent auto-scaling flicker
        self.view.setImage(self.imgarr.T, levels=(0.0, 1.0))

        # --- Main Layout Arrangement ---
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.control_panel)
        self.splitter.setSizes([800, 200]) # Initial sizes for splitter panes
        
        self.main_layout.addWidget(self.splitter)

        # --- Connect mouse events directly to the view's scene ---
        # sigMouseClicked passes QMouseEvent
        self.view.scene.sigMouseClicked.connect(self.mouse_clicked)
        # sigMouseMoved passes QMouseEvent
        self.view.scene.sigMouseMoved.connect(self._on_mouse_moved)
        
        # --- Install event filter for keyboard input on the view ---
        self.event_filter = LeniaEventFilter(self)
        self.view.installEventFilter(self.event_filter)

        # Continuous time for animations
        self.total_elapsed_ms = 0


    def calc_kernels(self):
        y, x = np.ogrid[-self.mid:self.mid, -self.mid:self.mid]
        D = np.sqrt(x**2 + y**2) / self.R
        Ds = [D * len(k['b']) for k in self.kernels]
        # Adjusted to handle cases where k['b'] might be a single value
        Ks = [(D < len(k['b'])) * np.asarray(k['b'])[np.minimum(D.astype(int), len(k['b']) - 1)] * bell(D % 1, 0.5, 0.15) if isinstance(k['b'], list) else \
              (D < 1) * np.asarray([k['b']])[np.minimum(D.astype(int), 0)] * bell(D % 1, 0.5, 0.15) for D, k in zip(Ds, self.kernels)]
        
        nKs = [K / np.sum(K) for K in Ks]
        fKs = [np.fft.fft2(np.fft.fftshift(K)).astype(np.complex128) for K in nKs]
        return fKs

    def setup_lenia(self):
        imgarr = np.zeros([self._lenia_size, self._lenia_size], dtype=np.float32) # Use _lenia_size
        C = np.asarray(self.cells)
        C = scipy.ndimage.zoom(C, self.scale, order=0)
        cx = cy = self.mid - C.shape[0] // 2
        imgarr[cx:cx+C.shape[0], cy:cy+C.shape[1]] += C
        return imgarr
    
    def load_creature(self, name):
        creature_data = give_conv_from_name(name)
        if creature_data:
            self.current_creature_data = creature_data
            
            # Update creature properties
            self.R = creature_data.get('R', self.R) * self.scale # Apply scale immediately
            self.T = creature_data.get('T', self.T)
            self.cells = creature_data.get('cells', [])
            self.animal_name = name

            # Determine kernels: prioritize 'kernels' list, then fall back to top-level m, s, b
            if 'kernels' in creature_data:
                self.kernels = creature_data['kernels']
            else:
                # Ensure m, s, b exist before trying to access them
                if all(k in creature_data for k in ['m', 's', 'b']):
                    self.kernels = [{'m': creature_data['m'], 's': creature_data['s'], 'b': creature_data['b']}]
                else:
                    QtWidgets.QMessageBox.warning(self, "Error", 
                                                  f"Creature '{name}' is missing 'kernels' list or top-level 'm', 's', 'b' parameters.")
                    return # Exit if essential data is missing

            self.kernelms = [(k['m'], k['s']) for k in self.kernels]
            self.fKs = self.calc_kernels()
            self.imgarr = self.setup_lenia()
            self.view.setImage(self.imgarr.T, levels=(0.0, 1.0))
            # Update GUI fields - these are now guaranteed to exist
            self.r_input.setText(str(self.R / self.scale)) # Show R without scale in GUI
            self.creature_input.setText(self.animal_name)
            print(f"Loaded creature: {self.animal_name}")
        else:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not load creature data for '{name}'. Check animal name or leniasetup.py.")

    def update_lenia_params(self):
        try:
            new_R = float(self.r_input.text()) * self.scale # Apply scale for internal use
            if new_R > 0:
                self.R = new_R
                self.fKs = self.calc_kernels()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for R.")

    def load_new_creature_from_input(self):
        new_creature_name = self.creature_input.text().strip()
        self.load_creature(new_creature_name)

    def toggle_sinback(self):
        self.pulse_enabled = not self.pulse_enabled
        self.sinback_button.setText("SinBack ON" if self.pulse_enabled else "SinBack OFF")

    def toggle_text_display(self):
        self.sign_enabled = not self.sign_enabled
        self.text_toggle_button.setText("Text ON" if self.sign_enabled else "Text OFF")
        # When toggling text display, re-render the mask if it's not already
        if self.sign_enabled and self.text_mask is None:
            self.text_mask = self._render_text_mask(self.text_content, self.text_font_size, self._lenia_size)
            # Initialize text position to center it
            text_mask_height, text_mask_width = self.text_mask.shape
            self.text_x = (self._lenia_size - text_mask_width) / 2
            self.text_y = (self._lenia_size - text_mask_height) / 2
            self.vel_x, self.vel_y = 2, 2 # Reset velocity when text is enabled
        # The text opacity is handled in update_simulation by blending with imgarr

    def toggle_text_pulse(self):
        self.text_pulse_enabled = not self.text_pulse_enabled
        self.text_pulse_button.setText("Text Pulse ON" if self.text_pulse_enabled else "Text Pulse OFF")
        # If pulse is turned off, reset intensity to full visible if text is enabled
        if not self.text_pulse_enabled and self.sign_enabled:
            self.text_intensity = 1.0


    def reset_simulation(self):
        self.imgarr = self.setup_lenia()
        self.view.setImage(self.imgarr.T, levels=(0.0, 1.0))

    def clear_grid(self):
        self.imgarr = np.zeros([self._lenia_size, self._lenia_size], dtype=np.float32) # Use _lenia_size
        self.view.setImage(self.imgarr.T, levels=(0.0, 1.0))

    def spawn(self, pos, scale=5):
        """Places a predefined Lenia creature at the clicked position."""
        # pos is already in imgarr coordinates
        px, py = int(pos.x()), int(pos.y())

        creature_pattern = np.asarray(self.cells)
        creature_resized = scipy.ndimage.zoom(creature_pattern, scale, order=0)
        
        h, w = creature_resized.shape
        
        # Calculate top-left corner to center the creature at the click
        x_start = px - w // 2
        y_start = py - h // 2

        # Clamp to bounds
        x_start = max(0, min(x_start, self._lenia_size - w)) # Use _lenia_size
        y_start = max(0, min(y_start, self._lenia_size - h)) # Use _lenia_size

        x_end = x_start + w
        y_end = y_start + h

        # Paste the creature into imgarr
        self.imgarr[y_start:y_end, x_start:x_end] += creature_resized[:y_end - y_start, :x_end - x_start]
        self.imgarr = np.clip(self.imgarr, 0, 1)

    def mouse_clicked(self, event):
        # Map event position to view's internal data coordinates
        pos = self.view.getImageItem().mapFromScene(event.scenePos())

        # Check if the click is within the simulation area
        if 0 <= pos.x() < self._lenia_size and \
           0 <= pos.y() < self._lenia_size: # Use _lenia_size
            if event.button() == QtCore.Qt.LeftButton:
                # Removed splash on left click as per user request
                self.mouse_held = True # Still track mouse held for potential future features
            elif event.button() == QtCore.Qt.RightButton:
                self.spawn(pos, self.scale)
            event.accept() # Accept the event to prevent propagation (e.g., context menu)
        else:
            event.ignore() # Ignore clicks outside the simulation area
        
    def _on_mouse_moved(self, event): # Only 'event' is passed by sigMouseMoved
        # 'event' is QMouseEvent
        if self.mouse_held and event.buttons() & QtCore.Qt.LeftButton:
            # Removed splash on left-button drag as per user request
            # mapped_pos = self.view.getImageItem().mapFromScene(event.scenePos())
            # if 0 <= mapped_pos.x() < self._lenia_size and \
            #    0 <= mapped_pos.y() < self._lenia_size: # Use _lenia_size
            #     self.imgarr = splash(self.imgarr, mapped_pos.y(), mapped_pos.x(), radius=10, intensity=-2)
            event.accept() # Accept the event to prevent propagation
        else:
            # If mouse is not held or it's not a left-button drag, let other handlers process
            event.ignore() 

    def _render_text_mask(self, text, font_size, _lenia_size):
        """Renders text to a QImage and returns its alpha channel as a NumPy mask."""
        font = QtGui.QFont("Arial", font_size) # You can choose a different font
        
        # Create a temporary QImage to measure text size
        temp_image = QtGui.QImage(1, 1, QtGui.QImage.Format_Alpha8)
        temp_painter = QtGui.QPainter(temp_image)
        temp_painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        text_rect = metrics.boundingRect(text)
        temp_painter.end()

        # Create the actual QImage with the correct size for the text
        image = QtGui.QImage(text_rect.width(), text_rect.height(), QtGui.QImage.Format_Alpha8)
        image.fill(QtCore.Qt.transparent) # Start with a transparent image

        painter = QtGui.QPainter(image)
        painter.setFont(font)
        
        # Draw text at (0,0) of this new, correctly sized image
        # Adjust position to account for potential negative offsets from boundingRect
        painter.setPen(QtCore.Qt.white) # Draw white text on transparent background
        painter.drawText(QtCore.QPointF(-text_rect.x(), -text_rect.y()), text)
        painter.end()

        # Convert QImage to NumPy array (alpha channel)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        # Reshape to (height, width)
        text_mask = np.array(ptr).reshape(image.height(), image.width()).astype(np.float32) / 255.0
        return text_mask


    def update_simulation(self):
        """
        Core animation loop:
        1. Update the Lenia simulation state.
        2. Apply optional effects.
        3. Set the image view to the new state.
        """
        self.imgarr = update(self.imgarr, self.fKs, self.kernelms, self._lenia_size, self.T) # Use _lenia_size
        
        # Update total elapsed time
        self.total_elapsed_ms += self.timer.interval()

        # Apply pulse effect if enabled
        if self.pulse_enabled:
            # Use total_elapsed_ms for continuous time
            self.imgarr = add_pulse(self.imgarr, self.total_elapsed_ms, frequency=0.008, speed=0.07, intensity=0.05, direction="horizontal")
            self.imgarr = add_pulse(self.imgarr, self.total_elapsed_ms, frequency=0.008, speed=0.05, intensity=0.05, direction="vertical")

        # Handle bouncing text (applied directly to imgarr)
        if self.sign_enabled:
            # Ensure text_mask is generated
            if self.text_mask is None:
                self.text_mask = self._render_text_mask(self.text_content, self.text_font_size, self._lenia_size)
                # If text_mask was just generated, re-center its position
                text_mask_height, text_mask_width = self.text_mask.shape
                self.text_x = (self._lenia_size - text_mask_width) / 2
                self.text_y = (self._lenia_size - text_mask_height) / 2
                self.vel_x, self.vel_y = 2, 2 # Reset velocity when text is enabled

            time_for_text = self.total_elapsed_ms
            
            # Get text dimensions from the mask
            text_mask_height, text_mask_width = self.text_mask.shape
            
            self.text_x += self.vel_x
            self.text_y += self.vel_y

            # Bounce off walls within the _lenia_size x _lenia_size area
            # text_x, text_y now represent the top-left corner of the text mask within imgarr
            if self.text_x <= 0 or self.text_x + text_mask_width >= self._lenia_size:
                self.vel_x = -self.vel_x
            if self.text_y <= 0 or self.text_y + text_mask_height >= self._lenia_size:
                self.vel_y = -self.vel_y
            
            # Apply fading intensity based on text_pulse_enabled
            if self.text_pulse_enabled:
                self.text_intensity = (np.sin(time_for_text / 400) + 1) / 2 # Smoother fade 0 to 1
            else:
                self.text_intensity = 1.0 # Full intensity if pulse is off

            # Apply the text mask to the imgarr
            x_start = int(self.text_x)
            y_start = int(self.text_y)
            
            x_end = min(x_start + text_mask_width, self._lenia_size)
            y_end = min(y_start + text_mask_height, self._lenia_size)

            # Ensure indices are within bounds
            x_start = max(0, x_start)
            y_start = max(0, y_start)

            # Calculate the portion of the mask to apply
            # Ensure slicing doesn't go out of bounds for the mask itself
            mask_slice_x_end = x_end - x_start
            mask_slice_y_end = y_end - y_start

            mask_slice = self.text_mask[0:mask_slice_y_end, 0:mask_slice_x_end]
            
            # Blend the text into the image array
            # Use a weighted average to blend, or simple addition/clipping
            # Here, we'll add it, but you might want to experiment with different blending modes
            self.imgarr[y_start:y_end, x_start:x_end] = np.clip(
                self.imgarr[y_start:y_end, x_start:x_end] + self.text_intensity * mask_slice,
                0, 1
            )


        self.view.setImage(self.imgarr.T, autoRange=False, autoLevels=False)
        
        # Update FPS display
        self.frame_count += 1
        current_time_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        if current_time_ms - self.last_fps_update_time >= 1000: # Update FPS every second
            fps = self.frame_count / ((current_time_ms - self.last_fps_update_time) / 1000.0)
            self.fps_label.setText(f"FPS: {int(fps)}") # Update QLabel
            self.last_fps_update_time = current_time_ms
            self.frame_count = 0


    def start_animation(self, frametime=16): # ~60 FPS
        """
        Uses a QTimer to call the update method in a loop.
        """
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(frametime)
        
        self.frame_count = 0
        self.last_fps_update_time = QtCore.QDateTime.currentMSecsSinceEpoch()
        
        self.show() # Show the main window
        # The QApplication instance should be created once outside the class
        # and its exec_() method called to start the event loop.
        # This part is handled in the __main__ block.

# Custom Event Filter for keyboard input on the ImageView
class LeniaEventFilter(QtCore.QObject):
    def __init__(self, parent_visualizer):
        super().__init__()
        self.visualizer = parent_visualizer

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Up:
                self.visualizer.R += self.visualizer.scale # Adjust R by scale
                self.visualizer.r_input.setText(str(self.visualizer.R / self.visualizer.scale))
                self.visualizer.fKs = self.visualizer.calc_kernels()
                return True # Event handled
            elif event.key() == QtCore.Qt.Key_Down:
                if self.visualizer.R > self.visualizer.scale: # Prevent R from becoming non-positive
                    self.visualizer.R -= self.visualizer.scale
                    self.visualizer.r_input.setText(str(self.visualizer.R / self.visualizer.scale))
                    self.visualizer.fKs = self.visualizer.calc_kernels()
                return True
            elif event.key() == QtCore.Qt.Key_R: # Reset simulation
                self.visualizer.reset_simulation()
                return True
            elif event.key() == QtCore.Qt.Key_C: # Clear grid
                self.visualizer.clear_grid()
                return True
        return super().eventFilter(obj, event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # Create QApplication instance once
    v = LeniaVisualizer2D(_lenia_size=512, scale=5) # Pass _lenia_size, set scale to 5
    v.start_animation()
    sys.exit(app.exec()) # Ensure the application exits cleanly
