import pygame
import numpy as np
import numba
from leniasetup import *
import pygame_gui


import matplotlib.pyplot as plt
from matplotlib import cm
import pridepy as pp

# color scheme
color_scheme = ['purple', 'pink', 'orange'] #pp.full_rainbow
test_colors = pp.paintkit.filter(tags={'dark'}).ordered_swatches(color_scheme).colors
# test_colors = pp.paintkit.filter(tags={'dark'}).ordered_swatches(pp.full_rainbow).colors

test_colors = [swatch.hex for swatch in test_colors]
# test_colors = [pp.paintkit.filter(tags={'dark'}).filter(tags={'pink'}).colors[0].hex]

test_colors = ['#0000']+ test_colors
# positions = [0.0, 0.5, 1.0]
cmap = pp.colormaps.perceptual_colormap_nonuniform(test_colors)
# cmap = cm.magma

def viridisshow(imgarr, gamma=0.7):
    # Normalize image to [0, 1] range
    imgarr = np.log1p(imgarr)  # Log transformation
    imgarr /= (imgarr.max() )
    # imgscale = 2 * np.exp(-(imgarr-0.15)**2/0.17)-1
    # imgscale = np.real(np.fft.ifft2(fKs[0] * np.fft.fft2(imgarr)))
    # imgscale = 2 * np.exp(-(imgscale-0.15)**2/0.001)-1
    # imgscale /= (imgarr.max() +0.1)
    # Apply the colormap
    color_mapped = cmap(imgarr)

    # Convert to Pygame-compatible RGB format
    color_mapped = (color_mapped[:, :, :3] *255 ).astype(np.uint8) #255

    return pygame.surfarray.make_surface(color_mapped)


@numba.jit(nopython=True)
def bell(x, m, s):
    return np.exp(-((x - m) / s) ** 2 / 2)

# Numba optimized growth function
@numba.jit(nopython=True)
def growth(U, m, s):
    return (bell(U, m, s) * 2 - 1)


@numba.jit(nopython=True,parallel=False)
def update(imgarr,fKs,kernelms,size,T):
 
  total = np.zeros((size,size))

  for i in range(len(fKs)):
    total += growth(np.real(np.fft.ifft2(fKs[i] * np.fft.fft2(imgarr))),kernelms[i][0],kernelms[i][1] )

  imgarr = np.clip(imgarr + 1/T * total/len(fKs), 0, 1)

  return imgarr



#set up start
#names: serratus velox(spin), Hexadentium,"serratus velox Tetravolvium
animal_name = "velox"
creature = pattern["orbium"]
# creature = give_conv_from_name(animal_name)

size = 512;  mid = size // 2;  scale = 5;  cx, cy = 80, 80

globals().update(creature)
R *= scale
if 'kernels' not in creature:
    kernels = [{'m':creature['m'],
                's':creature['s'],
                'b':creature['b']}]
kernelms = [( k['m'], k['s']) for k in kernels]
def calc_kernels(kernels,R):
    y, x = np.ogrid[-mid:mid, -mid:mid]  # Creates a grid of coordinates
    D = np.sqrt(x**2 + y**2) / R

    Ds = [ D * len(k['b']) for k in kernels ]
    Ks = [ (D<len(k['b'])) * np.asarray(k['b'])[np.minimum(D.astype(int),len(k['b'])-1)] * bell(D%1, 0.5, 0.15) for D,k in zip(Ds,kernels) ]
    nKs = [ K / np.sum(K) for K in Ks ]
    fKs = [ np.fft.fft2(np.fft.fftshift(K)) for K in nKs ]
    return fKs
fKs = calc_kernels(kernels,R)

def setup():

    imgarr = np.zeros([size, size])
    cord_list = [(80,80)]
    for cx,cy in cord_list:
        C = np.asarray(cells)
        C = scipy.ndimage.zoom(C, scale, order=0)
        imgarr[cx:cx+C.shape[0], cy:cy+C.shape[1]] += C
    return imgarr

imgarr = setup()

def splash(imgarr, pos, radius=20, intensity=0.5):
    """Adds a splash effect to imgarr at the correct position."""
    cy, cx = pos  # Flip Pygame's (x, y) to NumPy's (row, col)
    cx -= border/2 ; cy -= border/2
    y, x = np.ogrid[:imgarr.shape[0], :imgarr.shape[1]]
    dist = (x - cx)**2 + (y - cy)**2
    # mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * radius**2))
    mask = (1 + dist / (radius**2)) ** -1.2
    imgarr += intensity * mask
    return np.clip(imgarr, 0, 1)

def spawn(imgarr, pos, scale= 5):
    """Places a predefined Lenia creature at the clicked position."""
    cy, cx = pos  # Convert (x, y) to (row, col) for NumPy
    

    creature_pattern = np.asarray(cells)  # Assuming 'cells' contains a predefined pattern
    creature_resized = scipy.ndimage.zoom(creature_pattern, scale, order=0)  # Scale the creature
    cx -= border//2 + creature_resized.shape[0]//2 ; cy -= border//2 + creature_resized.shape[1]//2
    h, w = creature_resized.shape  # Get dimensions of the scaled pattern

    # Ensure the creature is within bounds
    x_start, x_end = max(cx, 0), min(cx + w, imgarr.shape[1])
    y_start, y_end = max(cy, 0), min(cy + h, imgarr.shape[0])

    # Paste the creature into imgarr
    imgarr[y_start:y_end, x_start:x_end] += creature_resized[:y_end - y_start, :x_end - x_start]
    
    return np.clip(imgarr, 0, 1)  # Ensure values stay in range

def add_noise(imgarr, noise_level=0.01):
    noise = (-np.random.rand(*imgarr.shape)) * noise_level
    imgarr *= noise
    return np.clip(imgarr, 0, 1)

# @numba.njit()
def add_pulse(imgarr, time, frequency=0.005, speed=0.02, intensity=0.2, direction="horizontal"):
    """Adds a sweeping periodic pulse to the simulation."""
    size = imgarr.shape[0]  # Assuming square grid
    t = time * speed  # Time-based shift

    # Generate sinusoidal pulse
    if direction == "horizontal":
        pulse = np.sin(2 * np.pi * frequency * (np.arange(size) - t)).reshape(1, -1)
    elif direction == "vertical":
        pulse = np.sin(2 * np.pi * frequency * (np.arange(size) - t)).reshape(-1, 1)

    # Normalize pulse to be between 0 and 1
    pulse = (pulse + 1) / 2  

    # Add the pulse to the image
    imgarr += intensity * pulse * np.random.normal(1,0.02,imgarr.shape )
    return np.clip(imgarr, 0, 1)

def wipe(imgarr, time, speed=0.0001, intensity=-0.1, direction="horizontal", width=20):
    """Creates a single traveling peak effect across the simulation."""
    size = imgarr.shape[0]  # Assuming square grid
    t = int(time * speed * size) % size  # Compute moving position

    # Generate a Gaussian peak
    if direction == "horizontal":
        pulse = np.exp(-((np.arange(size) - t) ** 2) / (2 * width ** 2)).reshape(1, -1)
    elif direction == "vertical":
        pulse = np.exp(-((np.arange(size) - t) ** 2) / (2 * width ** 2)).reshape(-1, 1)

    # Normalize pulse to be between 0 and 1
    pulse /= pulse.max()  

    # Apply effect to the image with noise
    imgarr += intensity * pulse * np.random.normal(1, 0.2, imgarr.shape)
    return np.clip(imgarr, 0, 1)


def add_text_outline(imgarr, text="LENIA", pos=(100, 100), font_size=50, thickness=2, period=3000):
    """Creates an outline of text in imgarr by setting outline pixels to 0."""
    current_time = pygame.time.get_ticks()  # Get elapsed time in milliseconds

    # Only apply the text effect periodically
    if current_time % period > period / 2:
        return imgarr  # Do nothing outside active period

    # Create font
    font = pygame.font.Font(None, font_size)  # Default font, customizable size
    text_surface = font.render(text, True, (255, 255, 255))  # White text

    # Convert to NumPy array
    text_array = pygame.surfarray.array_alpha(text_surface)

    # Create a binary mask (1 where text is, 0 elsewhere)
    text_mask = text_array > 128  

    # # Find edges (outline) using dilation - subtraction
    # from scipy.ndimage import binary_dilation
    # outline_mask = binary_dilation(text_mask, iterations=thickness) & ~text_mask

    # Apply outline to imgarr by setting those pixels to 0
    x, y = pos
    h, w = text_mask.shape
    imgarr[x:x+h, y:y+w][text_mask] = 1  # Set outline pixels to 0

    return imgarr

def add_fading_text(imgarr, text_surface,phase, intensity, pos=(100,100)):

    # Convert to NumPy array (alpha values)
    text_array = pygame.surfarray.array_alpha(text_surface)

    # Create a binary mask (1 where text is, 0 elsewhere)
    text_mask = text_array > 128  

    # Apply the fading effect by blending with the current imgarr values
    x, y = pos
    h, w = text_mask.shape
    if x + h < imgarr.shape[0] and y + w < imgarr.shape[1]:  # Ensure within bounds
        imgarr[x:x+h, y:y+w][text_mask] += intensity #* phase  # Gradual fade-in

    return np.clip(imgarr, 0, 1)  # Ensure values stay in range

# Initialize bouncing text position and velocity
text_x, text_y = 50, 50  # Starting position
vel_x, vel_y = 2, 0  # Speed in pixels per frame

def bounce_text(imgarr, text_surface, intensity=0.1, offset = 0):
    """Moves and fades in text, making it bounce inside the (size, size) screen."""
    global text_x, text_y, vel_x, vel_y  # Allow updates to global variables

    # Update position
    text_x += vel_x
    text_y += vel_y

    # Bounce off walls
    if text_x <= 0 or text_x + text_surface.get_width() >= size-5:
        vel_x = -vel_x  # Reverse x-direction
    if text_y <= 0 or text_y + text_surface.get_height() >= size-5:
        vel_y = -vel_y  # Reverse y-direction

    # Compute fade-in phase
    # fade_phase = min((time_elapsed - prev_time) / fade_duration, 1.0)

    # Apply text fade-in at new position
    imgarr = add_fading_text(imgarr, text_surface, 1, intensity=intensity, pos=(text_x, text_y+offset))

    return imgarr

# Initialize Pygame
pygame.init()

# Screen settings
border = 100
WIDTH, HEIGHT = size + border, size + border
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lenia Simulation")

# GUI Manager
manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# Create input box for R
r_input = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((10, 10), (100, 30)), manager=manager
)
r_input.set_text(str(R))  # Set initial R value in GUI

# Create input box for creature name
creature_input = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((120, 10), (150, 30)), manager=manager
)
creature_input.set_text(creature['name'])  # Set initial creature name

# Create button to load new creature
load_creature_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((280, 10), (60, 30)), 
    text="Load", 
    manager=manager
)

# Initialize feature flag
pulse_enabled = False

# Create button to toggle the feature on and off
toggle_feature_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((350, 10), (60, 30)),
    text="SinBack",
    manager=manager
)
# Initialize feature flag
pulse_enabled = False

# Create button to toggle the feature on and off
toggle_sign_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((420, 10), (60, 30)),
    text="Text",
    manager=manager
)
sign_enabled = False

# Font settings
font_size = 150
text = "LENIA"
font = pygame.font.Font(None, font_size)
text_surface = font.render(text, True, (255, 255, 255))  # White solid text
font_clock = pygame.font.Font(None, 24)
WHITE = (255, 255, 255)

# Clock for FPS
clock = pygame.time.Clock()
running = True

mouse_held = False

def load_new_creature(animal_name):
    global imgarr, creature, fKs, kernelms, R
    new = give_conv_from_name(animal_name)
    if new is None: pass
    creature = new
    globals().update(creature)
    R *= scale
    kernelms = [(k['m'], k['s']) for k in kernels]
    fKs = calc_kernels(kernels, R)
    imgarr = setup()

mass = []

while running:
    time_delta = clock.tick(30) / 1000.0  # Convert ms to seconds

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == toggle_feature_button:
                pulse_enabled = not pulse_enabled  # Toggle feature flag
                if pulse_enabled:
                    toggle_feature_button.set_text("sinOFF")
                else:
                    toggle_feature_button.set_text("sinON")

            if event.ui_element == toggle_sign_button:
                sign_enabled = not sign_enabled  # Toggle feature flag
                if pulse_enabled:
                    toggle_sign_button.set_text("TextOFF")
                else:
                    toggle_sign_button.set_text("TextON")

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_held = True  # Start holding the mouse down
                imgarr = splash(imgarr, event.pos, radius=20, intensity=-3)
            elif event.button == 3:  # Right click
                imgarr = spawn(imgarr, event.pos,scale)

        elif event.type == pygame.MOUSEMOTION:
            if mouse_held and event.buttons[0] == 1:  # If mouse button is held down
                imgarr = splash(imgarr, event.pos, radius=10, intensity=-2)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_held = False  # Stop holding the mouse down
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                R += 1
                r_input.set_text(str(R))  
                fKs = calc_kernels(kernels, R)
            elif event.key == pygame.K_DOWN:
                R -= 1
                r_input.set_text(str(R))  
                fKs = calc_kernels(kernels, R)
            elif event.key == pygame.K_r:
                imgarr = setup()
                fKs = calc_kernels(kernels, R)
            elif event.key == pygame.K_c:
                imgarr = np.zeros([size, size])


        # Handle GUI input
        manager.process_events(event)

        # Update R when Enter is pressed
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            try:
                new_R = float(r_input.get_text())
                if new_R > 0:
                    R = new_R
                    fKs = calc_kernels(kernels, R)
            except ValueError:
                print("Invalid R value! Enter a valid number.")
        if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == load_creature_button:
                new_creature_name = creature_input.get_text().strip()
                if give_conv_from_name(new_creature_name) is not None:
                    load_new_creature(new_creature_name)
                    r_input.set_text(str(R))  
                    print(f"Loaded creature: {new_creature_name}")
                else: 
                    print("failed")


    
    time_elapsed = pygame.time.get_ticks() 

    

    if pulse_enabled:
         # Get elapsed time in ms
        imgarr = add_pulse(imgarr, time_elapsed, frequency=0.008, speed=0.07, intensity=0.05, direction="horizontal")
        imgarr = add_pulse(imgarr, time_elapsed, frequency=0.008, speed=0.05, intensity=0.05, direction="vertical")
        bpm = 125
        # imgarr += (np.sin(time_elapsed* (2 * np.pi * bpm / 60000))+1)**2/20 * np.real(np.fft.ifft2(fKs[0] * np.fft.fft2(imgarr)))
        
    # imgarr = wipe(imgarr, time_elapsed)
    # imgarr = wipe(imgarr, time_elapsed, speed=0.0001, intensity=-0.1, direction="vertical", width=20)
    
    if sign_enabled:
        imgarr = bounce_text(imgarr, text_surface, intensity=(np.sin(time_elapsed/400)+1)/10, offset=150)    

    # GUI update
    manager.update(time_delta)

    # Update the simulation
    imgarr = update(imgarr, fKs, kernelms, size, T)
    # Clear screen
    screen.fill((0, 0, 0))
    # Render simulation
    screen.blit(viridisshow(imgarr), (50, 50))
    mass.append(np.sum(imgarr))

    # Display FPS
    fps = int(clock.get_fps())
    fps_text = font_clock.render(f"FPS: {fps}", True, WHITE)
    screen.blit(fps_text, (WIDTH - 80, 5))

    # Draw GUI elements
    manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
# import matplotlib.pyplot as plt
# plt.plot(mass[-200:])
# plt.show()
