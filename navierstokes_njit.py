import pygame
import sys
import math as m
import numpy as np
import scipy.ndimage as sim
from numba import njit, prange

CONTROLS = [
    {
        "Title" : "Model preset",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Combobox",
        "Value" : 0,
        "Items" : [
            "Implicit Diffusion",
        ]
    },
    {
        "Title" : "State preset",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Combobox",
        "Value" : 1,
        "Items" : [
            "Stagnant",
            "Wave",
            "Shear layer",
            "Spiral",
            "Checkerboard",
        ]
    },
    {
        "Title" : "Fluid preset",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Combobox",
        "Value" : 1,
        "Items" : [
            "Gasoline",
            "Water",
            "Ethanol",
            "Olive oil",
            "Glycerin",
            "Honey",
            "Mercury",
        ]
    },
    {
        "Title" : "Density",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Slider",
        "Unit" : "kg/m3",
        "Value" : 997.047,
        "Min" : 600,
        "Max" : 13500,
    },
    {
        "Title" : "Viscosity",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Slider",
        "Unit" : "mPa·s",
        "Value" : 0.89,
        "Min" : 0.3,
        "Max" : 10000,
    },
    {
        "Title" : "Time step",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Slider",
        "Unit" : "s",
        "Value" : 0.1,
        "Min" : 0.1,
        "Max" : 1,
    },
    {
        
        "Title" : "Start",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Button",
        "BgColor" : (26, 233, 3),       # lime background
        "FgColor" : (255, 255, 255),    # white text
    },
    {
        
        "Title" : "Pause",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Button",
        "BgColor" : (248, 131, 4),      # orange background
        "FgColor" : (255, 255, 255),    # white text
    },
    {
        
        "Title" : "Reset",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Button",
        "BgColor" : (150, 147, 143),    # gray background
        "FgColor" : (255, 255, 255),    # white text
    },
]

COLORISTICS = {
    "BG_COLOR" : (255, 255, 255),                  # white background
    "PANEL_COLOR" : (220, 220, 220),               # light gray background
    "CANVAS_COLOR" : (0, 0, 0),                    # gray background
    
    "SLIDER_ENABLED_BG"       : (200, 200, 255),   # light blue background
    "SLIDER_ENABLED_FILL"     : (0, 0, 200),       # dark blue fill
    "SLIDER_ENABLED_TEXT"     : (0, 0, 0),         # black text

    "SLIDER_DISABLED_BG"      : (200, 200, 200),   # gray background
    "SLIDER_DISABLED_FILL"    : (150, 150, 150),   # darker gray fill
    "SLIDER_DISABLED_TEXT"    : (100, 100, 100),   # dark gray text

    "COMBOBOX_ENABLED_BORDER" : (0, 0, 0),         # black border
    "COMBOBOX_ENABLED_BG"     : (255, 255, 255),   # white background
    "COMBOBOX_ENABLED_TEXT"   : (0, 0, 0),         # black text

    "COMBOBOX_DISABLED_BORDER": (120, 120, 120),   # gray border
    "COMBOBOX_DISABLED_BG"    : (200, 200, 200),   # gray background
    "COMBOBOX_DISABLED_TEXT"  : (150, 150, 150),   # gray text
    
    "BUTTON_DISABLED_BG"    : (200, 200, 200),     # gray background
    "BUTTON_DISABLED_TEXT"  : (150, 150, 150),     # gray text
}

def get_control(name):
    for control in CONTROLS:
        if(control.get("Title") == name):
            return control
    
    return None

def show_control(name):
    control = get_control(name)
    if control != None:
        control["Visible"] = True

def hide_control(name):
    control = get_control(name)
    if control != None:
        control["Visible"] = False

def enable_control(name):
    control = get_control(name)
    if control != None:
        control["Enabled"] = True

def disable_control(name):
    control = get_control(name)
    if control != None:
        control["Enabled"] = False

def render_control(control, screen, font, xpos, ypos, width, height):
    if not control.get("Visible", True):
        return

    rect = pygame.Rect(xpos, ypos, width, height)
    control["Rect"] = rect

    enabled = control.get("Enabled", True)
    if control["Type"] == "Button":
        bg = control.get("BgColor", (100, 100, 100)) if enabled else COLORISTICS["BUTTON_DISABLED_BG"]
        fg = control.get("FgColor", (255, 255, 255)) if enabled else COLORISTICS["BUTTON_DISABLED_TEXT"]
        pygame.draw.rect(screen, bg, rect)
        text_surf = font.render(control["Title"], True, fg)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    elif control["Type"] == "Slider":
        bg_color = COLORISTICS["SLIDER_ENABLED_BG"] if enabled else COLORISTICS["SLIDER_DISABLED_BG"]
        fill_color = COLORISTICS["SLIDER_ENABLED_FILL"] if enabled else COLORISTICS["SLIDER_DISABLED_FILL"]
        text_color = COLORISTICS["SLIDER_ENABLED_TEXT"] if enabled else COLORISTICS["SLIDER_DISABLED_TEXT"]

        pygame.draw.rect(screen, bg_color, rect)

        min_val = control.get("Min", 0)
        max_val = control.get("Max", 1)
        val = control.get("Value", 0)
        fill_width = int((val - min_val) / (max_val - min_val) * width)
        pygame.draw.rect(screen, fill_color, (xpos, ypos, fill_width, height))

        label = "{}: {:.2f} {}".format(control['Title'], val, control.get('Unit',''))
        text_surf = font.render(label, True, text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    elif control["Type"] == "Combobox":
        border_color = COLORISTICS["COMBOBOX_ENABLED_BORDER"] if enabled else COLORISTICS["COMBOBOX_DISABLED_BORDER"]
        bg_color = COLORISTICS["COMBOBOX_ENABLED_BG"] if enabled else COLORISTICS["COMBOBOX_DISABLED_BG"]
        text_color = COLORISTICS["COMBOBOX_ENABLED_TEXT"] if enabled else COLORISTICS["COMBOBOX_DISABLED_TEXT"]

        pygame.draw.rect(screen, border_color, rect)
        inner_rect = rect.inflate(-2, -2)
        pygame.draw.rect(screen, bg_color, inner_rect)

        items = control.get("Items", [])
        value_index = control.get("Value", 0)
        value_index = max(0, min(value_index, len(items) - 1))
        value_text = items[value_index] if items else ""
        label = f"{control['Title']}: {value_text}"

        text_surf = font.render(label, True, text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

        if control.get("Expanded", False):
            item_height = 30
            for i, item in enumerate(control["Items"]):
                item_rect = pygame.Rect(rect.left - rect.width, rect.top + i * item_height, rect.width, item_height)
                pygame.draw.rect(screen, bg_color, item_rect)
                pygame.draw.rect(screen, border_color, item_rect, 2)
                item_text_surf = font.render(item, True, text_color)
                item_text_rect = item_text_surf.get_rect(center=item_rect.center)
                screen.blit(item_text_surf, item_text_rect)
    
def render_gui(screen, font, xpos, ypos, width, height):
    CONTROL_HEIGHT = 40

    cypos = ypos
    panel_rect = pygame.Rect(xpos, ypos, width, height)
    pygame.draw.rect(screen, COLORISTICS["PANEL_COLOR"], panel_rect)
    for control in CONTROLS:
        render_control(control, screen, font, xpos + 2, cypos + 4, width - 4, CONTROL_HEIGHT - 8)
        cypos = cypos + CONTROL_HEIGHT

def get_control_from_pt(mouse_x, mouse_y):
    for control in CONTROLS:
        rect = control.get("Rect")
        
        if rect != None:
            if rect.collidepoint(mouse_x, mouse_y):
                return control
            
            if control["Type"] == "Combobox":
                if control.get("Expanded", False):
                    item_height = 30
                    items_rect = pygame.Rect(rect.left - rect.width, rect.top, rect.width, item_height * len(control["Items"]))
                    if items_rect.collidepoint(mouse_x, mouse_y):
                        return control
        
    return None

def collapse_all_comboboxes():
    for control in CONTROLS:
        if control["Type"] == "Combobox":
            control["Expanded"] = False

def on_click_slider(control, mouse_x, _):
    rect = control["Rect"]
    min_val = control["Min"]
    max_val = control["Max"]
    
    px = mouse_x - rect.left
    val = px / rect.width * (max_val - min_val)
    
    control["Value"] = max(min_val, min(val, max_val))
    
    if "OnChange" in control:
        control["OnChange"](control["Value"])

def on_click_combobox(control, mouse_x, mouse_y):
    rect = control["Rect"]
    
    if rect.collidepoint(mouse_x, mouse_y):
        collapse_all_comboboxes()
        control["Expanded"] = True

    if control.get("Expanded", False):
        item_height = 30
        for i, _ in enumerate(control["Items"]):
            item_rect = pygame.Rect(rect.left - rect.width, rect.top + i * item_height, rect.width, item_height)
            if item_rect.collidepoint(mouse_x, mouse_y):
                control["Value"] = i
                control["Expanded"] = False
    
                if "OnChange" in control:
                    control["OnChange"](control["Value"])
                break

FLUID_PRESETS = [
    {
        "Density" : 737,
        "Viscosity" : 0.55,
    },
    {
        "Density" : 997,
        "Viscosity" : 0.89,
    },
    {
        "Density" : 789,
        "Viscosity" : 1.2,
    },
    {
        "Density" : 920,
        "Viscosity" : 84,
    },
    {
        "Density" : 1260,
        "Viscosity" : 1500,
    },
    {
        "Density" : 1420,
        "Viscosity" : 10000,
    },
    {
        "Density" : 13534,
        "Viscosity" : 1.55,
    }
]

def on_change_fluid_presets(index):
    if index < len(FLUID_PRESETS):
        get_control("Density")["Value"] = FLUID_PRESETS[index]["Density"]
        get_control("Viscosity")["Value"] = FLUID_PRESETS[index]["Viscosity"]

def on_change_state_presets(index):
    if index == 0:
        initialize_state("stagnant")
    elif index == 1:
        initialize_state("wave")
    elif index == 2:
        initialize_state("shear layer")
    elif index == 3:
        initialize_state("spiral")
    elif index == 4:
        initialize_state("checkerboard")

RUNNING = False
def on_start(control, mouse_x, mouse_y):
    get_control("Model preset")["Enabled"] = False
    get_control("State preset")["Enabled"] = False
    get_control("Fluid preset")["Enabled"] = False
    
    get_control("Start")["Enabled"] = False
    get_control("Pause")["Enabled"] = True
    get_control("Reset")["Enabled"] = True
    
    collapse_all_comboboxes()
    
    global RUNNING
    RUNNING = True

def on_pause(control, mouse_x, mouse_y):
    get_control("Model preset")["Enabled"] = False
    get_control("State preset")["Enabled"] = False
    get_control("Fluid preset")["Enabled"] = False
    
    get_control("Start")["Enabled"] = True
    get_control("Pause")["Enabled"] = False
    get_control("Reset")["Enabled"] = True
    
    collapse_all_comboboxes()
    
    global RUNNING
    RUNNING = False

T = 0.0
def on_reset(control, mouse_x, mouse_y):
    global RUNNING, T
    RUNNING = False
    T = 0
    
    get_control("Model preset")["Enabled"] = True
    get_control("State preset")["Enabled"] = True
    get_control("Fluid preset")["Enabled"] = True
    
    get_control("Start")["Enabled"] = True
    get_control("Pause")["Enabled"] = False
    get_control("Reset")["Enabled"] = False
    
    collapse_all_comboboxes()
    
    on_change_state_presets(get_control("State preset")["Value"])
    

U = np.array([])
V = np.array([])
P = np.array([])
SOLID = np.array([])
FIELD_WIDTH = 0
FIELD_HEIGHT = 0
CELL_SIZE = 0
def initialize_state(state_type):
    global U, V, P, SOLID

    U = np.zeros((FIELD_WIDTH + 1, FIELD_HEIGHT))
    V = np.zeros((FIELD_WIDTH, FIELD_HEIGHT + 1))
    P = np.zeros((FIELD_WIDTH, FIELD_HEIGHT))
    SOLID = np.zeros((FIELD_WIDTH, FIELD_HEIGHT))

    cx = FIELD_WIDTH / 2
    cy = FIELD_HEIGHT / 2
    scale = max(FIELD_WIDTH, FIELD_HEIGHT)

    def radial(x, y):
        dx = x - cx
        dy = y - cy
        return m.sqrt(dx * dx + dy * dy)

    if state_type == "stagnant":
        P.fill(0.0)

    elif state_type == "wave":
        for y in range(FIELD_HEIGHT):
            for x in range(FIELD_WIDTH):
                P[x, y] = 8.0 * m.sin(2 * m.pi * x / FIELD_WIDTH)

    elif state_type == "shear layer":
        for y in range(FIELD_HEIGHT):
            pressure = 10.0 if y < cy else -10.0
            P[:, y] = pressure

    elif state_type == "spiral":
        for y in range(FIELD_HEIGHT):
            for x in range(FIELD_WIDTH):
                dx = x - cx
                dy = y - cy
                r = m.sqrt(dx ** 2 + dy ** 2) / scale * 10.0
                theta = m.atan2(dy, dx)
                P[x, y] = 10.0 * m.sin(4 * theta + r)

    elif state_type == "checkerboard":
        for y in range(FIELD_HEIGHT):
            for x in range(FIELD_WIDTH):
                val = ((x // 8 + y // 8) % 2) * 2 - 1
                P[x, y] = val * 10.0

    P[0, :] = P[1, :]
    P[-1, :] = P[-2, :]
    P[:, 0] = P[:, 1]
    P[:, -1] = P[:, -2]

    U.fill(0.0)
    V.fill(0.0)
    SOLID.fill(False)
    SOLID[[0, -1], :] = True
    SOLID[:, [0, -1]] = True

COLOR_STOPS = {
    0.00: (0,   0,   139),   # dark blue
    0.25: (0,  255,   0),    # green
    0.50: (255, 255, 0),     # yellow
    0.75: (255, 165, 0),     # orange
    1.00: (255,   0, 0),     # red
}
def get_pressure_color(val):
    val = clamp(val, 0.0, 1.0)

    keys = sorted(COLOR_STOPS.keys())

    for i in range(len(keys) - 1):
        k1 = keys[i]
        k2 = keys[i + 1]

        if k1 <= val <= k2:
            t = (val - k1) / (k2 - k1)

            c1 = COLOR_STOPS[k1]
            c2 = COLOR_STOPS[k2]

            r = int(lerp(c1[0], c2[0], t))
            g = int(lerp(c1[1], c2[1], t))
            b = int(lerp(c1[2], c2[2], t))

            return (r, g, b)


def draw_arrow(screen, start_pos, direction, color, length=50, arrow_head_size=10, arrow_shaft_size=3):
    direction = pygame.math.Vector2(direction)
    if direction.length() == 0:
        return

    direction = direction.normalize()

    arrow_head_dist = arrow_head_size * m.sqrt(3) / 2.0
    if length > arrow_head_dist:
        mid_pos = pygame.math.Vector2(start_pos) + direction * (length - arrow_head_dist)
        pygame.draw.line(screen, color, start_pos, mid_pos, arrow_shaft_size)

    end_pos = pygame.math.Vector2(start_pos) + direction * length
    arrow_angle = m.atan2(direction.y, direction.x)
    left_arrow = (
        end_pos.x - arrow_head_size * m.cos(arrow_angle - m.pi / 6),
        end_pos.y - arrow_head_size * m.sin(arrow_angle - m.pi / 6)
    )
    right_arrow = (
        end_pos.x - arrow_head_size * m.cos(arrow_angle + m.pi / 6),
        end_pos.y - arrow_head_size * m.sin(arrow_angle + m.pi / 6)
    )

    points = [tuple(map(int, p)) for p in [end_pos, left_arrow, right_arrow]]
    pygame.draw.polygon(screen, color, points)

def render_scale(screen, font, xpos, ypos, width, height, pressure_min, pressure_max, ticks):
    vals = np.linspace(0.0, 1.0, height)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i, v in enumerate(vals):
        color = get_pressure_color(v)

        img[i, :, :] = color

    surf = pygame.surfarray.make_surface(np.rot90(img, 3))
    screen.blit(surf, (xpos, ypos))

    pygame.draw.rect(screen, (0, 0, 0), (xpos, ypos, width, height), 3)

    for i in range(ticks):
        t = i / (ticks - 1)
        py = int(ypos + (1.0 - t) * height)
        if i == 0:
            py -= 2

        pygame.draw.line(screen, (0, 0, 0),
                         (xpos + width, py),
                         (xpos + width + 6, py), 2)

        pval = pressure_min + t * (pressure_max - pressure_min)
        text = font.render(f"{pval:.2f}", True, (0, 0, 0))
        screen.blit(text, (xpos + width + 10, py - text.get_height()//2))

def render_scene(screen, font, xpos, ypos):
    pressure_min = +1e9
    pressure_max = -1e9

    velocity_max = 0.0
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT):
            # pressure_max = max(pressure_max, P[x, y])
            
            velocity = (U[x, y], V[x, y])
            velocity_mag = m.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
            velocity_max = max(velocity_max, velocity_mag)
            
            p = get_pressure(P, x, y)
            pressure_min = min(pressure_min, p)
            pressure_max = max(pressure_max, p)
    
    smooth_P = sim.zoom(P, zoom=CELL_SIZE, order=1)    
    for px in range(smooth_P.shape[0]):
        for py in range(smooth_P.shape[1]):
            if pressure_max == pressure_min:
                val = 0
            else:
                p = smooth_P[px, py]
                val = (p - pressure_min) / (pressure_max - pressure_min)
            
            color = get_pressure_color(val)
            screen.set_at((px + xpos, py + ypos), color)
            
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT):
            if is_solid(SOLID, x, y):
                pixel_rect = pygame.Rect(xpos + x * CELL_SIZE, ypos + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), pixel_rect)
    
    arrow_max_len = 5
    arrow_head = 5
    arrow_shaft = 3
    arrow_color = (0, 0, 0)
    arrow_freq = 5
    for x in range(arrow_freq // 2, FIELD_WIDTH, arrow_freq):
        for y in range(arrow_freq // 2, FIELD_HEIGHT, arrow_freq):
                velocity = (U[x, y], V[x, y])
                velocity_mag = m.sqrt((velocity[0] * CELL_SIZE) ** 2 + (velocity[1] * CELL_SIZE) ** 2)
                if velocity_max != 0:
                    draw_arrow(screen, (xpos + x * CELL_SIZE, ypos + y * CELL_SIZE), velocity, arrow_color, arrow_max_len * velocity_mag / velocity_max, arrow_head, arrow_shaft)
                    
    render_scale(screen, font, xpos + FIELD_WIDTH * CELL_SIZE + 50, ypos, 30, FIELD_HEIGHT * CELL_SIZE, pressure_min, pressure_max, 7)
                
def render_info(screen, font, xpos, ypos):
    text = font.render(f"Time elapsed: {T:.2f} s", True, (0, 0, 0))
    screen.blit(text, (xpos, ypos))

DEFAULT_METHOD = 0
def on_change_method_presets(index):
    global DEFAULT_METHOD
    DEFAULT_METHOD = index

@njit
def calculate_velocity_divergence_at_cell(U, V, x, y):
    """
    This function is used to get divergence value for given cell.\n
    Arguments:
        x, y - coordinates
    Returns:
        Divergence value based on current velocity matrices
    """
    
    vT = V[x + 0, y + 1]
    vL = U[x + 0, y + 0]
    vR = U[x + 0, y + 1]
    vB = V[x + 0, y + 0]
    
    gX = (vR - vL) / CELL_SIZE
    gY = (vT - vB) / CELL_SIZE
    
    divergence = gX + gY
    
    return divergence

@njit
def get_pressure(P, x, y):
    """
    This function is used to safely get pressure value.\n
    Arguments:
        x, y - coordinates
    Returns:
        0 if coordinates out of bounds, pressure value otherwise
    """
    
    oob = (x < 0 or x >= P.shape[0] or y < 0 or y >= P.shape[1])
    return 0 if oob else P[x, y]

@njit
def is_solid(S, x, y):
    """
    This function is used to safely get solid value.\n
    Arguments:
        x, y - coordinates
    Returns:
        True if tile is solid or for coordinates out of bounds, False otherwise
    """
    
    oob = (x < 0 or x >= S.shape[0] or y < 0 or y >= S.shape[1])
    return True if oob else S[x, y]

@njit(parallel=True)
def diffuse_field(F, F0, a, S, iterations=10):
    width = F.shape[0]
    height = F.shape[1]
    
    for _ in range(iterations):
        NewF = F.copy()

        for x in prange(1, width - 1):
            for y in range(1, height - 1):
                if is_solid(S, x, y):
                    continue

                fL = F[x+1,y] 
                fR = F[x-1,y]
                fT = F[x,y+1]
                fB = F[x,y-1]
                af0 = F0[x,y] * a
                
                NewF[x, y] = (fL + fR + fT + fB + af0) / (4 + a)

        F[:] = NewF

    return F

@njit
def clamp(x, a, b):
    """
    This function is used to clamp x argument in a range [a, b].\n
    Arguments:
        x - argument
        a, b - bounds
    Returns:
        Clamped x value
    """
    return min(max(x, a), b)

@njit
def lerp(a, b, t):
    """
    This function is used for linear interpolation between two values a and b.\n
    Arguments:
        a, b - bounds
        t - fraction
    Returns:
        Interpolated value
    """
    return a + (b - a) * t

@njit
def sample_bilinear(F, worldPos):
    """
    This function is used for sampling eV matrix at worldPos.\n
    Arguments:
        eV - matrix to sample
        eCX, eCY - matrix dimensions
        worldPos - coordinates
    Returns:
        Sampled value at given coordinates.
    Algorithm:
        1. Detect cell to sample
        2. Detect edges of a cell
        3. Interpolate by X and Y axis
    """
    
    width = (F.shape[0] - 1) * CELL_SIZE
    height = (F.shape[1] - 1) * CELL_SIZE
    
    px = (worldPos[0] + width // 2) // CELL_SIZE
    py = (worldPos[1] + height // 2) // CELL_SIZE
    
    left = int(clamp(px, 0, F.shape[0] - 2))
    bottom = int(clamp(py, 0, F.shape[1] - 2))
    right = left + 1
    top = bottom + 1
    
    xFrac = clamp(px - left, 0, 1)
    yFrac = clamp(py - bottom, 0, 1)
    
    fT = lerp(F[left, top], F[right, top], xFrac)
    fB = lerp(F[left, bottom], F[right, bottom], xFrac)
    
    return lerp(fB, fT, yFrac)

@njit
def get_vel_at_world_pos(U, V, worldPos):
    """
    This function is used for sampling velocities at worldPos.\n
    Arguments:
        worldPos - coordinates
    Returns:
        Sampled velocity vector at given coordinates.
    """
    
    velX = sample_bilinear(U, worldPos)
    velY = sample_bilinear(V, worldPos)
    
    return velX, velY

@njit(parallel=True)
def advect_field(U, V, dt, h, axis, S):
    F = U if axis == 0 else V
    newF = F.copy()
    width = F.shape[0]
    height = F.shape[1]
    
    axisx = 1 if axis == 0 else 0
    axisy = 1 if axis == 1 else 0
    
    for x in prange(width):
        for y in range(height):

            if is_solid(S, x, y) or is_solid(S, x-1, y):
                newF[x][y] = 0
                continue

            wx = (x - (width - axisx)/2) * h
            wy = (y - (height - axisy)/2) * h

            vel = get_vel_at_world_pos(U, V, (wx, wy))

            back = (wx - vel[0] * dt,
                    wy - vel[1] * dt)

            newF[x][y] = sample_bilinear(F, back)

    return newF

@njit
def pressure_solve_for_cell(U, V, P, S, x, y, rho, dt, h):
    """
    This function is used to calculate new pressure matrix based on current values.\n
    Arguments:
        x, y - cell coordinates to calculate
        rho - density
        dt - time step
    Returns:
        New pressure matrix
    Algorithm:
        1. Check which pressures must be accounted. Skip if solid or surrounded by solid cells
        2. Calculate divergence
        3. Calculate new pressure value
    """
    
    fT = 0 if is_solid(S, x + 0, y + 1) else 1
    fL = 0 if is_solid(S, x - 1, y + 0) else 1
    fR = 0 if is_solid(S, x + 1, y + 0) else 1
    fB = 0 if is_solid(S, x + 0, y - 1) else 1

    eC = fT + fL + fR + fB
    if eC == 0 or is_solid(S, x, y):
        return 0.0

    div = calculate_velocity_divergence_at_cell(U, V, x, y)

    pT = get_pressure(P, x + 0, y + 1) * fT
    pL = get_pressure(P, x - 1, y + 0) * fL
    pR = get_pressure(P, x + 1, y + 0) * fR
    pB = get_pressure(P, x + 0, y - 1) * fB

    alpha = -rho * h * h / dt

    return (pL + pR + pT + pB + alpha * div) / eC

@njit(parallel=True)
def pressure_poisson_solve(U, V, P, S, rho, dt, h, iterations=10):
    """
    This function is used to iteratively solve pressure matrix.
    Arguments:
        P - current pressure matrix
        rho - density
        dt - time step
        iterations - Jacobi iterations count
    Returns:
        New pressure matrix
    Algorithm:
        1. For every iteration, iterate every cell
        2. For every cell, solve current pressure value
        3. At the end of each iteration, copy to current pressure matrix
    """
    
    width = P.shape[0]
    height = P.shape[1]
    for _ in range(iterations):
        New_P = np.zeros_like(P)
        for x in prange(width):
            for y in range(height):
                New_P[x, y] = pressure_solve_for_cell(U, V, P, S, x, y, rho, dt, h)
        P[:] = New_P
        
    return P

@njit(parallel=True)
def project_field(F, k, axis, S):
    width = F.shape[0]
    height = F.shape[1]
    
    for x in prange(0, width):
        for y in range(0, height):
            if is_solid(S, x, y - 1) or is_solid(S, x, y):
                F[x, y] = 0.0
                continue
            
            p1 = get_pressure(P, x - 1, y + 0) if axis == 0 else get_pressure(P, x + 0, y + 0)
            p2 = get_pressure(P, x + 0, y + 0) if axis == 0 else get_pressure(P, x + 0, y - 1)
            F[x, y] = F[x, y] - k * (p1 - p2)
    
    return F
        
def step_implicit_diffusion_method():
    """
    This is main step function of Jacobi Iterative Solver for the Diffusion Equation.\n
    Algorithm:
        1. Fetch viscosity, dynamic density and time step, calculate kinematic viscocity
        2. Apply kinematic viscosity
        3. Advance velocity field
        4. Solve pressure
        5. Correct velocities by new pressure values
        6. Advance time by a time step
    """
    
    global U, V, P, T
    
    mu = get_control("Viscosity")["Value"] / 1000.0
    rho = get_control("Density")["Value"]
    dt = get_control("Time step")["Value"]
    
    nu = mu / rho
    
    a = dt * nu / (CELL_SIZE * CELL_SIZE)
    U0 = U.copy()
    V0 = V.copy()
    
    U = diffuse_field(U, U0, a, SOLID)
    V = diffuse_field(V, V0, a, SOLID)
    
    U = advect_field(U, V, dt, CELL_SIZE, 0, SOLID)
    V = advect_field(U, V, dt, CELL_SIZE, 1, SOLID)
    
    P = pressure_poisson_solve(U, V, P, SOLID, rho, dt, CELL_SIZE)
            
    k = dt / (rho * CELL_SIZE)
    U = project_field(U, k, 0, SOLID)
    V = project_field(V, k, 0, SOLID)
        
    T = T + dt

def step_simulation():
    if DEFAULT_METHOD == 0:
        step_implicit_diffusion_method()

def init(field_width, field_height, cellsize):
    get_control("Model preset")["OnClick"] = on_click_combobox
    get_control("Model preset")["OnChange"] = on_change_method_presets
    get_control("State preset")["OnClick"] = on_click_combobox
    get_control("State preset")["OnChange"] = on_change_state_presets
    get_control("Fluid preset")["OnClick"] = on_click_combobox
    get_control("Fluid preset")["OnChange"] = on_change_fluid_presets
    get_control("Density")["OnClick"] = on_click_slider
    get_control("Viscosity")["OnClick"] = on_click_slider
    get_control("Time step")["OnClick"] = on_click_slider
    get_control("Start")["OnClick"] = on_start
    get_control("Pause")["OnClick"] = on_pause
    get_control("Reset")["OnClick"] = on_reset
    
    get_control("Start")["Enabled"] = True
    get_control("Pause")["Enabled"] = False
    get_control("Reset")["Enabled"] = False
    
    global FIELD_WIDTH, FIELD_HEIGHT, CELL_SIZE
    FIELD_WIDTH = field_width // cellsize
    FIELD_HEIGHT = field_height // cellsize
    CELL_SIZE = cellsize
    initialize_state("wave")

def main():
    WIDTH, HEIGHT = 800, 480
    PANEL_WIDTH = 200
    FONT_SIZE = 16
    SIM_SIZE = 300
    CELL_SIZE = 5
    
    pygame.init()
    init(SIM_SIZE, SIM_SIZE, CELL_SIZE)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Navier Stokes")

    font = pygame.font.SysFont(None, FONT_SIZE)

    clock = pygame.time.Clock()
    previous_time = pygame.time.get_ticks()

    dragging_control = None
    while True:
        current_time = pygame.time.get_ticks()
        delta_time = current_time - previous_time
        
        if delta_time >= 50:
            previous_time = current_time
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    
                    control = get_control_from_pt(mouse_x, mouse_y)
                    if not control:
                        continue
                    if not control.get("Enabled", False):
                        continue
                    if "OnClick" in control:
                        control["OnClick"](control, mouse_x, mouse_y)
                        dragging_control = control
                        
                if event.type == pygame.MOUSEMOTION:
                    if dragging_control:
                        mouse_x, mouse_y = event.pos
                        if "OnClick" in dragging_control:
                            dragging_control["OnClick"](dragging_control, mouse_x, mouse_y)

                if event.type == pygame.MOUSEBUTTONUP:
                    if dragging_control:
                        dragging_control = None

        if RUNNING:
            step_simulation()
        
        bg_rect = pygame.Rect(0, 0, WIDTH - PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(screen, COLORISTICS["BG_COLOR"], bg_rect)
    
        render_scene(screen, font, (WIDTH - PANEL_WIDTH - SIM_SIZE) // 2, (HEIGHT - SIM_SIZE) // 2)
        render_gui(screen, font, WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, HEIGHT)
        render_info(screen, font, 10, 10)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
