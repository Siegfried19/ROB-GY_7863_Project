from pynput import keyboard

flag = False
listener = None

def on_press(key):
    global flag
    if key == keyboard.Key.space:
        flag = True
        print("Signal detected!")  # 输出信号

def on_release(key):
    global flag
    if key == keyboard.Key.space:
        flag = False

def start_listener():
    global listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

def stop_listener():
    global listener
    listener.stop()
    listener = None

def is_active():
    return flag

if __name__ == "__main__":
    start_listener()
                       