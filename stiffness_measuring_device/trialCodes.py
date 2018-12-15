import keyboard #Using module keyboard
while True:#making a loop#used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('q'):#if key 'q' is pressed 
            print('You Pressed A Key!')
            break#finishing the loop
        else:
            pass
    
         #if user pressed a key other than the given key the loop will break