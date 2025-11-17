"""
"THE BEER-WARE LICENSE" (Revision 42):

<a3.ratajczak@gmail.com> wrote this file. As long as you retain this notice you can do whatever you want with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
Adam Ratajczak
"""

import pygame
import sys
import math as m
import numpy as np
import scipy.ndimage as sim
from scipy.sparse import lil_matrix
import pyamg

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
        "Value" : 0,
        "Items" : [
            "stagnant",
            "swirl",
            "turbulence",
            "vortex-ring",
            "double-vortex",
            "explosion",
            "implosion",
            "corner-blast",
            "shear-wall",
            "rotating-box",
            "swirl-band",
            "horizontal-sine",
            "diagonal-shear",
            "random-bursts",
            "four-quad-vortex",
            "taylor-green",
            "kelvin-helmholtz",
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
        "Title" : "Rendering Method",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Combobox",
        "Value" : 1,
        "Items" : [
            "Pressure",
            "Gas density",
            "Divergence",
        ]
    },
    {
        "Title" : "Drawing Method",
        "Enabled" : True,
        "Visible" : True,
        "Type" : "Combobox",
        "Value" : 0,
        "Items" : [
            "Gas",
            "Velocity",
            "Gas and velocity",
        ]
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

M_DiffusionU = None
M_DiffusionV = None
M_Pressure = None
def on_change_params(val):
    mu = get_control("Viscosity")["Value"] / 1000.0
    rho = get_control("Density")["Value"]
    dt = get_control("Time step")["Value"]
    
    global M_DiffusionU, M_DiffusionV, M_Pressure
    nu = mu / rho
    h = CELL_SIZE
    a = dt * nu / (h*h)

    AU = build_diffusion_matrix(FIELD_WIDTH + 1, FIELD_HEIGHT, a)
    M_DiffusionU = pyamg.ruge_stuben_solver(AU)

    AV = build_diffusion_matrix(FIELD_WIDTH, FIELD_HEIGHT + 1, a)
    M_DiffusionV = pyamg.ruge_stuben_solver(AV)
    
    AP = build_pressure_matrix(FIELD_WIDTH, FIELD_HEIGHT)
    M_Pressure = pyamg.ruge_stuben_solver(AP)

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

RUNNING = False
CONVERGED = False
SMOKE_MODE = 0
VELOCITY_MUL = 10
def on_start(control, mouse_x, mouse_y):
    get_control("Model preset")["Enabled"] = False
    get_control("State preset")["Enabled"] = False
    get_control("Fluid preset")["Enabled"] = False
    
    get_control("Start")["Enabled"] = False
    get_control("Pause")["Enabled"] = True
    get_control("Reset")["Enabled"] = True
    
    collapse_all_comboboxes()
    
    global RUNNING, CONVERGED
    RUNNING = True
    CONVERGED = False

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
    global RUNNING, T, CONVERGED
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
    
    CONVERGED = False
    

U = np.array([])
V = np.array([])
P = np.array([])
D = np.array([])
SOLID = np.array([])
FIELD_WIDTH = 0
FIELD_HEIGHT = 0
CELL_SIZE = 0
def initialize_state(state_type):
    global U, V, P, D, SOLID

    U = np.zeros((FIELD_WIDTH + 1, FIELD_HEIGHT), dtype=float)
    V = np.zeros((FIELD_WIDTH, FIELD_HEIGHT + 1), dtype=float)
    P = np.zeros((FIELD_WIDTH, FIELD_HEIGHT), dtype=float)
    SOLID = np.zeros((FIELD_WIDTH, FIELD_HEIGHT), dtype=bool)
    D = np.zeros((FIELD_WIDTH, FIELD_HEIGHT, 3), dtype=float)

    cx = (FIELD_WIDTH) * 0.5
    cy = (FIELD_HEIGHT) * 0.5

    if state_type == "swirl":
        for y in range(FIELD_HEIGHT):
            for x in range(FIELD_WIDTH):
                dx = x - cx
                dy = y - cy
                r = max(np.sqrt(dx * dx + dy * dy), 1.0)
                U[x, y]   = -dy / r * 5.0
                U[x+1, y] = -dy / r * 5.0
                V[x, y]   =  dx / r * 5.0
                V[x, y+1] =  dx / r * 5.0

    elif state_type == "kelvin-helmholtz":
        for x in range(FIELD_WIDTH + 1):
            for y in range(FIELD_HEIGHT):
                if y < FIELD_HEIGHT // 2:
                    U[x, y] = +5.0
                else:
                    U[x, y] = -5.0
                U[x, y] += (np.random.rand() - 0.5) * 0.5

    elif state_type == "taylor-green":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                X = x / FIELD_WIDTH * 2 * np.pi
                Y = y / FIELD_HEIGHT * 2 * np.pi
                U[x, y]   =  np.cos(X) * np.sin(Y) * 5.0
                U[x+1, y] =  np.cos(X) * np.sin(Y) * 5.0
                V[x, y]   = -np.sin(X) * np.cos(Y) * 5.0
                V[x, y+1] = -np.sin(X) * np.cos(Y) * 5.0

    elif state_type == "diagonal-jet":
        for i in range(min(FIELD_WIDTH, FIELD_HEIGHT)):
            U[i, i] += 8.0
            V[i, i] += 8.0

    elif state_type == "turbulence":
        U = np.random.randn(FIELD_WIDTH + 1, FIELD_HEIGHT) * 1.5
        V = np.random.randn(FIELD_WIDTH, FIELD_HEIGHT + 1) * 1.5
        
    elif state_type == "vortex-ring":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                dx = x - cx
                dy = y - cy
                r = np.sqrt(dx*dx + dy*dy)
                if 10 < r < 25:
                    V[x, y] += dx * 0.4
                    U[x, y] -= dy * 0.4
                    V[x, y+1] += dx * 0.4
                    U[x+1, y] -= dy * 0.4

    elif state_type == "double-vortex":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                dx = x - cx
                dy = y - cy
                U[x, y] = -dy * 0.1
                V[x, y] = dx * 0.1
                dx2 = x - cx*0.5
                dy2 = y - cy*0.5
                U[x, y] += dy2 * 0.1
                V[x, y] -= dx2 * 0.1

    elif state_type == "explosion":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                dx = x - cx
                dy = y - cy
                U[x, y] = dx * 0.3
                V[x, y] = dy * 0.3
                U[x+1, y] = dx * 0.3
                V[x, y+1] = dy * 0.3

    elif state_type == "implosion":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                dx = x - cx
                dy = y - cy
                U[x, y] = -dx * 0.3
                V[x, y] = -dy * 0.3
                U[x+1, y] = -dx * 0.3
                V[x, y+1] = -dy * 0.3

    elif state_type == "corner-blast":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                dx = x
                dy = y
                U[x, y] = dx * 0.2
                V[x, y] = dy * 0.2
                U[x+1, y] = dx * 0.2
                V[x, y+1] = dy * 0.2

    elif state_type == "shear-wall":
        for x in range(FIELD_WIDTH+1):
            for y in range(FIELD_HEIGHT):
                U[x, y] = (y / FIELD_HEIGHT - 0.5) * 10.0

    elif state_type == "rotating-box":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                U[x, y] = -(y - cy) * 0.2
                V[x, y] =  (x - cx) * 0.2

    elif state_type == "swirl-band":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                if cy-10 < y < cy+10:
                    U[x, y] = 6.0 * np.sin(x * 0.2)
                    V[x, y] = 6.0 * np.cos(x * 0.2)

    elif state_type == "horizontal-sine":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                U[x, y] = np.sin(y / FIELD_HEIGHT * np.pi * 4) * 4.0

    elif state_type == "diagonal-shear":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                U[x, y] = (x + y) * 0.01
                V[x, y] = (x - y) * 0.01

    elif state_type == "random-bursts":
        for _ in range(25):
            rx = np.random.randint(0, FIELD_WIDTH)
            ry = np.random.randint(0, FIELD_HEIGHT)
            for x in range(max(0,rx-3), min(FIELD_WIDTH,rx+3)):
                for y in range(max(0,ry-3), min(FIELD_HEIGHT,ry+3)):
                    U[x,y] += np.random.uniform(-5,5)
                    V[x,y] += np.random.uniform(-5,5)

    elif state_type == "four-quad-vortex":
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                dx = (x - cx)
                dy = (y - cy)
                U[x,y] = np.sign(dx) * (-dy) * 0.15
                V[x,y] = np.sign(dy) * (dx) * 0.15
    P.fill(0.0)
    D.fill(0.0)
    
    tweak_values = 10
    for x in range(FIELD_WIDTH):
        for y in range(FIELD_HEIGHT):
            U[x, y] *= tweak_values
            V[x, y] *= tweak_values
    
    SOLID[:, :] = False
    SOLID[0, :] = True
    SOLID[-1, :] = True
    SOLID[:, 0] = True
    SOLID[:, -1] = True

    on_change_params(None)

def on_change_state_presets(index):
    initialize_state(get_control("State preset")["Items"][index])

def map_color(val, colormap):
    val = clamp(val, 0.0, 1.0)

    keys = sorted(colormap.keys())

    for i in range(len(keys) - 1):
        k1 = keys[i]
        k2 = keys[i + 1]

        if k1 <= val <= k2:
            t = (val - k1) / (k2 - k1)

            c1 = colormap[k1]
            c2 = colormap[k2]

            r = int(lerp(c1[0], c2[0], t))
            g = int(lerp(c1[1], c2[1], t))
            b = int(lerp(c1[2], c2[2], t))

            return (r, g, b)
    return (0, 0, 0)
    

PRESSURE_COLOR_STOPS = {
    0.00: (0,   0,   139),   # dark blue
    0.25: (0,  255,   0),    # green
    0.50: (255, 255, 0),     # yellow
    0.75: (255, 165, 0),     # orange
    1.00: (255,   0, 0),     # red
}

def get_pressure_color(val):
    return map_color(val, PRESSURE_COLOR_STOPS)

DIVERGENCE_COLOR_STOPS = {
    0.00: (0,   0,   255),   # blue
    1.00: (255,   0, 0),     # red
}
def get_divergence_color(val):
    return map_color(val, DIVERGENCE_COLOR_STOPS)

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

def render_arrow_field(screen, xpos, ypos, arrow_color):
    arrow_max_len = 5
    arrow_head = 5
    arrow_shaft = 3
    arrow_freq = 5

    velocity_max = get_max_vel()
    for x in range(arrow_freq // 2, FIELD_WIDTH, arrow_freq):
        for y in range(arrow_freq // 2, FIELD_HEIGHT, arrow_freq):
                velocity = (U[x, y], V[x, y])
                velocity_mag = m.sqrt((velocity[0] * CELL_SIZE) ** 2 + (velocity[1] * CELL_SIZE) ** 2)
                if velocity_mag >= 1e-12 and velocity_max >= 1e-12:
                    t = clamp(velocity_mag / velocity_max, 0, 1)
                    draw_arrow(screen, (xpos + x * CELL_SIZE, ypos + y * CELL_SIZE), velocity, arrow_color, arrow_max_len * t * CELL_SIZE, arrow_head, arrow_shaft)

def render_scale(screen, font, xpos, ypos, width, height, mapping_func, val_min, val_max, ticks):
    vals = np.linspace(0.0, 1.0, height)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i, v in enumerate(vals):
        color = mapping_func(v)

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

        pval = val_min + t * (val_max - val_min)
        text = font.render(f"{pval:.2E}", True, (0, 0, 0))
        screen.blit(text, (xpos + width + 10, py - text.get_height()//2))

def get_max_vel():
    velocity_max = 0.0
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT):
            velocity = (U[x, y], V[x, y])
            velocity_mag = m.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
            velocity_max = max(velocity_max, velocity_mag)
    
    return velocity_max

def render_pressure(screen, font, xpos, ypos):
    pressure_min = +1e9
    pressure_max = -1e9
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT):
            p = get_pressure(x, y)
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
            if is_solid(x, y):
                pixel_rect = pygame.Rect(xpos + x * CELL_SIZE, ypos + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), pixel_rect)
                    
    render_arrow_field(screen, xpos, ypos, (0, 0, 0))
    render_scale(screen, font, xpos + FIELD_WIDTH * CELL_SIZE + 50, ypos, 30, FIELD_HEIGHT * CELL_SIZE, get_pressure_color, pressure_min, pressure_max, 7)

def render_smoke(screen, xpos, ypos):
    smooth = np.zeros((FIELD_WIDTH * CELL_SIZE,
                       FIELD_HEIGHT * CELL_SIZE,
                       3), dtype=np.uint8)

    for c in range(3):
        smooth[:, :, c] = sim.zoom(D[:, :, c],
                                   zoom=CELL_SIZE,
                                   order=1) * 255.0
    
    surf = pygame.surfarray.make_surface(smooth)
    screen.blit(surf, (xpos, ypos))

    for x in range(FIELD_WIDTH):
        for y in range(FIELD_HEIGHT):
            if is_solid(x, y):
                pygame.draw.rect(screen,
                                 (0, 0, 0),
                                 (xpos + x*CELL_SIZE,
                                  ypos + y*CELL_SIZE,
                                  CELL_SIZE,
                                  CELL_SIZE))

    render_arrow_field(screen, xpos, ypos, (220, 220, 220))

def render_divergence(screen, font, xpos, ypos):
    divergence = get_divergence_matrix()
    divergence_min = +1e9
    divergence_max = -1e9
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT):
            d = divergence[x, y]
            divergence_min = min(divergence_min, d)
            divergence_max = max(divergence_max, d)
    
    smooth_D = sim.zoom(divergence, zoom=CELL_SIZE, order=1)    
    for px in range(smooth_D.shape[0]):
        for py in range(smooth_D.shape[1]):
            if divergence_max == divergence_min:
                val = 0
            else:
                p = smooth_D[px, py]
                val = (p - divergence_min) / (divergence_max - divergence_min)
            
            color = get_divergence_color(val)
            screen.set_at((px + xpos, py + ypos), color)
            
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT):
            if is_solid(x, y):
                pixel_rect = pygame.Rect(xpos + x * CELL_SIZE, ypos + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), pixel_rect)
                    
    render_arrow_field(screen, xpos, ypos, (0, 0, 0))
    render_scale(screen, font, xpos + FIELD_WIDTH * CELL_SIZE + 50, ypos, 30, FIELD_HEIGHT * CELL_SIZE, get_divergence_color, divergence_min, divergence_max, 7)

def render_scene(screen, font, xpos, ypos):
    if RENDERING_METHOD == 0:
        render_pressure(screen, font, xpos, ypos)
    elif RENDERING_METHOD == 1:
        render_smoke(screen, xpos, ypos)
    elif RENDERING_METHOD == 2:
        render_divergence(screen, font, xpos, ypos)
                
def render_info(screen, font, xpos, ypos):
    text = font.render(f"Time elapsed: {T:.2f} s", True, (0, 0, 0))
    screen.blit(text, (xpos, ypos))
    if SMOKE_MODE == 0:
        text = font.render("Smoke: red", True, (0, 0, 0))
        screen.blit(text, (xpos, ypos + 30))
    elif SMOKE_MODE == 1:
        text = font.render("Smoke: green", True, (0, 0, 0))
        screen.blit(text, (xpos, ypos + 30))
    elif SMOKE_MODE == 2:
        text = font.render("Smoke: blue", True, (0, 0, 0))
        screen.blit(text, (xpos, ypos + 30))
        
    text = font.render(f"Velocity: {VELOCITY_MUL:.2f}m/s", True, (0, 0, 0))
    screen.blit(text, (xpos, ypos + 60))
    
    if CONVERGED:
        text = font.render("Converged!", True, (0, 0, 0))
        screen.blit(text, (xpos, ypos + 90))

DEFAULT_METHOD = 0
def on_change_method_presets(index):
    global DEFAULT_METHOD
    DEFAULT_METHOD = index
    
RENDERING_METHOD = 1
def on_change_rendering_method_presets(index):
    global RENDERING_METHOD
    RENDERING_METHOD = index

DRAWING_METHOD = 0
def on_change_drawing_method_presets(index):
    global DRAWING_METHOD
    DRAWING_METHOD = index

def calculate_velocity_divergence_at_cell(x, y):
    uR = U[x+1, y]
    uL = U[x,   y]
    vT = V[x,   y+1]
    vB = V[x,   y]

    div = ((uR - uL) + (vT - vB)) / CELL_SIZE
    return div

def get_pressure(x, y):
    oob = (x < 0 or x >= FIELD_WIDTH or y < 0 or y >= FIELD_HEIGHT)
    return 0 if oob else P[x, y]

def is_solid(x, y):
    oob = (x < 0 or x >= FIELD_WIDTH or y < 0 or y >= FIELD_HEIGHT)
    return True if oob else SOLID[x, y]


def apply_boundary(F):
    W, H = F.shape
    for x in range(W):
        for y in range(H):
            if is_solid(x, y):
                F[x, y] = 0.0
    return F
    
def build_diffusion_matrix(width, height, a):
    N = width * height
    A = lil_matrix((N, N))

    def idx(x, y):
        return y + x * height

    for x in range(width):
        for y in range(height):
            i = idx(x, y)

            if is_solid(x, y):
                A[i, i] = 1.0
                continue

            A[i, i] = 1.0 + 4.0 * a

            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0 <= nx < width and 0 <= ny < height and not is_solid(nx, ny):
                    j = idx(nx, ny)
                    A[i, j] = -a

    return A.tocsr()
    
def diffuse_velocity(U, V):
    def solve_component(F, solver):
        b = F.flatten()
        x = solver.solve(b, tol=1e-8)
        return x.reshape(F.shape)

    U = solve_component(U, M_DiffusionU)
    V = solve_component(V, M_DiffusionV)

    return U, V

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def lerp(a, b, t):
    return a + (b - a) * t

def sample_bilinear(eV, eCX, eCY, worldPos):
    width = (eCX - 1) * CELL_SIZE
    height = (eCY - 1) * CELL_SIZE

    px = (worldPos[0] + width  / 2.0) / CELL_SIZE
    py = (worldPos[1] + height / 2.0) / CELL_SIZE

    left   = int(m.floor(clamp(px, 0.0, eCX - 2.0)))
    bottom = int(m.floor(clamp(py, 0.0, eCY - 2.0)))
    right  = left + 1
    top    = bottom + 1

    xFrac = clamp(px - left,   0.0, 1.0)
    yFrac = clamp(py - bottom, 0.0, 1.0)

    vT = lerp(eV[left,  top],    eV[right, top],    xFrac)
    vB = lerp(eV[left,  bottom], eV[right, bottom], xFrac)

    return lerp(vB, vT, yFrac)

def get_vel_at_world_pos(worldPos):
    velX = sample_bilinear(U, FIELD_WIDTH + 1, FIELD_HEIGHT, worldPos)
    velY = sample_bilinear(V, FIELD_WIDTH, FIELD_HEIGHT + 1, worldPos)
    
    return (velX, velY)

def advect(U, V, D, dt):
    newU = U.copy()
    newV = V.copy()
    newD = D.copy()

    for x in range(FIELD_WIDTH + 1):
        for y in range(FIELD_HEIGHT):

            if is_solid(x, y) or is_solid(x-1, y):
                newU[x][y] = 0
                continue

            wx = (x - FIELD_WIDTH/2) * CELL_SIZE
            wy = (y + 0.5 - FIELD_HEIGHT/2) * CELL_SIZE

            vel = get_vel_at_world_pos((wx, wy))

            back = (wx - vel[0] * dt,
                    wy - vel[1] * dt)

            newU[x][y] = sample_bilinear(U, FIELD_WIDTH+1, FIELD_HEIGHT, back)

    for x in range(FIELD_WIDTH):
        for y in range(FIELD_HEIGHT + 1):

            if is_solid(x, y) or is_solid(x, y-1):
                newV[x][y] = 0
                continue

            wx = (x + 0.5 - FIELD_WIDTH/2) * CELL_SIZE
            wy = (y - FIELD_HEIGHT/2) * CELL_SIZE

            vel = get_vel_at_world_pos((wx, wy))

            back = (wx - vel[0] * dt,
                    wy - vel[1] * dt)

            newV[x][y] = sample_bilinear(V, FIELD_WIDTH, FIELD_HEIGHT+1, back)

    for x in range(FIELD_WIDTH):
        for y in range(FIELD_HEIGHT):

            if is_solid(x, y):
                newD[x][y][:] = 0.0
                continue

            wx = (x + 0.5 - FIELD_WIDTH  / 2) * CELL_SIZE
            wy = (y + 0.5 - FIELD_HEIGHT / 2) * CELL_SIZE

            vel = get_vel_at_world_pos((wx, wy))

            back = (wx - vel[0] * dt,
                    wy - vel[1] * dt)
            
            for c in range(3):
                newD[x, y, c] = sample_bilinear(D[..., c], FIELD_WIDTH, FIELD_HEIGHT, back) * 0.995

    return newU, newV, newD

def build_pressure_matrix(width, height):
    N = width * height
    A = lil_matrix((N, N))

    def idx(x, y):
        return y + x * height

    for x in range(width):
        for y in range(height):
            i = idx(x, y)

            if is_solid(x, y):
                A[i, i] = 1.0
                continue

            neighbors = 0

            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0 <= nx < width and 0 <= ny < height and not is_solid(nx, ny):
                    neighbors += 1

            A[i, i] = neighbors

            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0 <= nx < width and 0 <= ny < height and not is_solid(nx, ny):
                    j = idx(nx, ny)
                    A[i, j] = -1.0

    return A.tocsr()

def get_divergence_matrix():
    divergence = np.zeros([FIELD_WIDTH, FIELD_HEIGHT])
    for x in range(FIELD_WIDTH - 1):
        for y in range(FIELD_HEIGHT - 1):
            divergence[x, y] = calculate_velocity_divergence_at_cell(x, y)
    return divergence

def pressure_poisson_solve(P):
    divergence = get_divergence_matrix()
    b = divergence.flatten()
    
    new_p = M_Pressure.solve(b, tol=1e-8)

    return new_p.reshape(P.shape)
        
def converged(eps):
    for x in range(FIELD_WIDTH):
        for y in range(FIELD_HEIGHT):
            divergence = calculate_velocity_divergence_at_cell(x, y)
            if divergence > eps:
                return False
    return True

def project_velocities(U, V, rho, dt):
    k = dt / (rho * CELL_SIZE)
    for x in range(0, FIELD_WIDTH + 1):
        for y in range(0, FIELD_HEIGHT):
            if is_solid(x - 1, y) or is_solid(x, y):
                U[x, y] = 0.0
                continue
            
            pL = get_pressure(x - 1, y + 0)
            pR = get_pressure(x + 0, y + 0)
            U[x, y] -= k * (pR - pL)
            
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT + 1):
            if is_solid(x, y - 1) or is_solid(x, y):
                V[x, y] = 0.0
                continue
            
            pT = get_pressure(x + 0, y + 0)
            pB = get_pressure(x + 0, y - 1)
            V[x, y] -= k * (pT - pB)
    
    return U, V
        
def step_implicit_diffusion_method():
    global U, V, D, P, T, CONVERGED
    
    rho = get_control("Density")["Value"]
    dt = get_control("Time step")["Value"]
    
    U, V = diffuse_velocity(U, V)
    
    U, V, D = advect(U, V, D, dt)
    
    P = pressure_poisson_solve(P)
            
    U, V = project_velocities(U, V, rho, dt)
    
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
    get_control("Density")["OnChange"] = on_change_params
    get_control("Viscosity")["OnClick"] = on_click_slider
    get_control("Viscosity")["OnChange"] = on_change_params
    get_control("Time step")["OnClick"] = on_click_slider
    get_control("Time step")["OnChange"] = on_change_params
    get_control("Rendering Method")["OnClick"] = on_click_combobox
    get_control("Rendering Method")["OnChange"] = on_change_rendering_method_presets
    get_control("Drawing Method")["OnClick"] = on_click_combobox
    get_control("Drawing Method")["OnChange"] = on_change_drawing_method_presets
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
    on_change_state_presets(get_control("State preset")["Value"])

def main():
    global U, V, D, SMOKE_MODE, VELOCITY_MUL
    
    WIDTH, HEIGHT = 800, 600
    PANEL_WIDTH = 200
    FONT_SIZE = 16
    SIM_SIZE = 300
    CELL_SIZE = 5
    DRAWING_RADIUS = 3
    SMOKE_RADIUS = 5
    SCENE_X = (WIDTH - PANEL_WIDTH - SIM_SIZE) // 2
    SCENE_Y = (HEIGHT - SIM_SIZE) // 2
    
    pygame.init()
    init(SIM_SIZE, SIM_SIZE, CELL_SIZE)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Navier Stokes")

    font = pygame.font.SysFont(None, FONT_SIZE)

    clock = pygame.time.Clock()
    previous_time = pygame.time.get_ticks()

    dragging_control = None
    dragging_scene = False
    prev_mouse = None

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
                    prev_mouse = (mouse_x, mouse_y)

                    if SCENE_X <= mouse_x <= SCENE_X + SIM_SIZE and \
                    SCENE_Y <= mouse_y <= SCENE_Y + SIM_SIZE:

                        dragging_scene = True
                        continue

                    control = get_control_from_pt(mouse_x, mouse_y)
                    if control and control.get("Enabled", False):
                        if "OnClick" in control:
                            control["OnClick"](control, mouse_x, mouse_y)
                            dragging_control = control
                    continue
                if event.type == pygame.MOUSEWHEEL:
                    if DRAWING_METHOD == 0 or DRAWING_METHOD == 2:
                        if event.y > 0.0:
                            SMOKE_MODE = 0 if SMOKE_MODE == 2 else SMOKE_MODE + 1
                        else:
                            SMOKE_MODE = 2 if SMOKE_MODE == 0 else SMOKE_MODE - 1
                    elif DRAWING_METHOD == 1:
                        VELOCITY_MUL += event.y

                if event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.pos

                    if dragging_scene and prev_mouse:
                        dx = mouse_x - prev_mouse[0]
                        dy = mouse_y - prev_mouse[1]
                        prev_mouse = (mouse_x, mouse_y)

                        cx = (mouse_x - SCENE_X) // CELL_SIZE
                        cy = (mouse_y - SCENE_Y) // CELL_SIZE
                        
                        if DRAWING_METHOD == 0 or DRAWING_METHOD == 2:
                            for x in range(clamp(cx - SMOKE_RADIUS, 0, FIELD_WIDTH),
                                        clamp(cx + SMOKE_RADIUS, 0, FIELD_WIDTH)):
                                for y in range(clamp(cy - SMOKE_RADIUS, 0, FIELD_HEIGHT),
                                            clamp(cy + SMOKE_RADIUS, 0, FIELD_HEIGHT)):
                                    if (x - cx) ** 2 + (cy - y) ** 2 <= SMOKE_RADIUS ** 2:
                                        D[x, y, SMOKE_MODE] = clamp(D[x, y, SMOKE_MODE] + 0.25, 0, 1)
                        
                        if DRAWING_METHOD == 1 or DRAWING_METHOD == 2:
                            velx = dx * VELOCITY_MUL
                            vely = dy * VELOCITY_MUL
                            for x in range(clamp(cx - DRAWING_RADIUS, 0, FIELD_WIDTH),
                                        clamp(cx + DRAWING_RADIUS, 0, FIELD_WIDTH)):
                                for y in range(clamp(cy - DRAWING_RADIUS, 0, FIELD_HEIGHT),
                                            clamp(cy + DRAWING_RADIUS, 0, FIELD_HEIGHT)):
                                    if (x - cx) ** 2 + (cy - y) ** 2 <= DRAWING_RADIUS ** 2:
                                        if 0 <= x < FIELD_WIDTH + 1 and 0 <= y < FIELD_HEIGHT:
                                            U[x, y] += velx

                                        if 0 <= x < FIELD_WIDTH and 0 <= y < FIELD_HEIGHT + 1:
                                            V[x, y] += vely

                        continue

                    if dragging_control:
                        if "OnClick" in dragging_control:
                            dragging_control["OnClick"](dragging_control, mouse_x, mouse_y)
                        continue

                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_scene = False
                    dragging_control = None
                    prev_mouse = None


        if RUNNING:
            step_simulation()
        
        bg_rect = pygame.Rect(0, 0, WIDTH - PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(screen, COLORISTICS["BG_COLOR"], bg_rect)
    
        render_scene(screen, font, SCENE_X, SCENE_Y)
        render_gui(screen, font, WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, HEIGHT)
        render_info(screen, font, 10, 10)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
