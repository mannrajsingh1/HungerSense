
from gpiozero import Button
from signal import pause

touch = Button(5, pull_up=False)

touch.when_pressed = lambda: print(
    "Touch detected! Meal interaction recorded."
)

print("Waiting for touch...")
pause()
