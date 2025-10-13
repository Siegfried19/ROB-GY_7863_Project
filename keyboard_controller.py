from pynput import keyboard

import math
import threading
import numpy as np

KEY_TO_CTRL_INDEX = {
    keyboard.KeyCode.from_char('w'): (0, +1),
    keyboard.KeyCode.from_char('s'): (0, -1),
    keyboard.KeyCode.from_char('a'): (1, +1),
    keyboard.KeyCode.from_char('d'): (1, -1),
    keyboard.KeyCode.from_char('q'): (2, +1),
    keyboard.KeyCode.from_char('e'): (2, -1),
    keyboard.KeyCode.from_char('c'): (3, -1),
    keyboard.Key.space: (3, -1)
}

SPECIAL_KEYS = {
    'reset': keyboard.Key.backspace,
    'quit': keyboard.Key.esc,
}

flags = {'reset': False, 'quit': False}

ctrl_lock = threading.Lock()
control_target = np.zeros(4, dtype=float)
listener = None


def start_listener(suppress=True):
    global listener, mouse_listener
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, suppress=suppress)
    listener.daemon = True
    listener.start()
    print("[manual_controller] Keyboard listener started.")

def _on_press(key):
    if key == SPECIAL_KEYS['reset']:
        with ctrl_lock:
            flags['reset'] = True
        return
    if key == SPECIAL_KEYS['quit']:
        with ctrl_lock:
            flags['quit'] = True
        return
    if key in KEY_TO_CTRL_INDEX:
        idx, direction = KEY_TO_CTRL_INDEX[key]
        if idx < 4:
            with ctrl_lock:
                control_target[idx] = direction * 2.0
        else:
            with ctrl_lock:
                control_target[idx] = direction
                
def _on_release(key):
    if key in KEY_TO_CTRL_INDEX:
        idx, _ = KEY_TO_CTRL_INDEX[key]
        with ctrl_lock:
            control_target[idx] = 0.0


if __name__ == "__main__":
    import time

    start_listener()
    print("Press Esc to quit, Backspace to reset. control_target will keep printing.")

    try:
        while True:
            with ctrl_lock:
                current_target = control_target.copy()
                should_quit = flags['quit']
                if flags['reset']:
                    print("Reset flag detected.")
                    flags['reset'] = False
            print(f"control_target: {current_target}")
            if should_quit:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    if listener is not None:
        listener.stop()
        listener = None












# flag = False
# listener = None

# def on_press(key):
#     global flag
#     if key == keyboard.Key.space:
#         flag = True
#         print("Signal detected!")  # 输出信号

# def on_release(key):
#     global flag
#     if key == keyboard.Key.space:
#         flag = False

# def start_listener():
#     global listener
#     listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#     listener.start()

# def stop_listener():
#     global listener
#     listener.stop()
#     listener = None

# def is_active():
#     return flag

# if __name__ == "__main__":
#     start_listener()
                       
