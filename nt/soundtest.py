from ntcore import NetworkTableInstance
from playsound import playsound
import pygame
import time

def getShift(seconds):
    if (seconds < 10):
        return "Transition"
    elif (seconds < 10 + 25):
        return "Shift 1"
    elif (seconds < 10 + 25 + 25):
        return "Shift 2"
    elif (seconds < 10 + 25 + 25 + 25):
        return "Shift 3"
    elif (seconds < 10 + 25 + 25 + 25 + 25):
        return "Shift 4"
    elif (seconds < 140):
        return "Endgame"
    else:
        return "Disabled"

pygame.mixer.init()

shift_change_sound = pygame.mixer.Sound('media/jason.mp3')
time_count_sound = pygame.mixer.Sound('media/balls2.mp3')

seconds_sounds = []
shift_scounds = {}

print("Initializing sounds")

for i in range(30):
    seconds_sounds.append(pygame.mixer.Sound("media/seconds/s" + str(i) + ".mp3"))

shift_scounds["Transition"] = pygame.mixer.Sound("media/shifts/auto_lost.mp3")
shift_scounds["Shift 1"] = pygame.mixer.Sound("media/shifts/lob.mp3")
shift_scounds["Shift 2"] = pygame.mixer.Sound("media/shifts/scoring.mp3")
shift_scounds["Shift 3"] = pygame.mixer.Sound("media/shifts/lob.mp3")
shift_scounds["Shift 4"] = pygame.mixer.Sound("media/shifts/scoring.mp3")
shift_scounds["Endgame"] = pygame.mixer.Sound("media/shifts/endgame.mp3")

print("Sounds initialized")

ntInst = NetworkTableInstance.getDefault()
ntInst.startClient4("my-dashboard")
# ntInst.setServerTeam(3006)
ntInst.setServer("127.0.0.1")
print("Connecting to NetworkTables...")
    
time.sleep(1)

smartdashboard = ntInst.getTable("SmartDashboard")

subscriber = smartdashboard.getDoubleTopic("shift-time-int").subscribe(0.0)

shift = smartdashboard.getStringTopic("Match state").subscribe("")

disabled = smartdashboard.getBooleanTopic("disabled").subscribe(False)

print("Smartdashboard connected")

last_floor_val = 0
last_shift = ""
second_counter = 0
switch_shift = False
match_counter = 140
while True:
    if disabled.get():
        match_counter = 140
        continue
    shift_val = shift.get()
    # if (shift_val != last_shift):
    #     print("Received shift:", shift_val)
    #     print("\n")
    #     last_shift = shift_val
    #     playsound('media/jason.mp3', block=False)
    #     if (shift_val == "SHIFT_1" or shift_val == "SHIFT_2" or shift_val == "SHIFT_3" or shift_val == "SHIFT_4"):
    #         second_counter = 24
    #         # playsound('media/balls2.mp3', block=False)
    # else:
    #     if (second_counter >= 0):
    #         playsound("media/balls2.mp3", block=False)
    #         second_counter -= 1
    
    # time.sleep(1)
    
    # print("Received value:", subscriber.get())
    val = subscriber.get()
    floored_val = int(val)
    if (last_floor_val != floored_val):
        match_counter = match_counter - 1
        print(match_counter)
        # if (last_shift != shift_val):
        if (last_floor_val == 0 or match_counter == 139):
            predicted_shift = getShift(140 - match_counter)
            print("Received shift:", shift_val)
            print("Predicted shift:", predicted_shift)
            print("\n")
            last_shift = shift_val
            # shift_change_sound.play()
            shift_scounds[predicted_shift].play()
        else:
            print("Received floor:", floored_val)
            print("Received value:", val)
            print("\n")
            # time_count_sound.play()
            seconds_sounds[floored_val].play()
        last_floor_val = floored_val
    # if (last_shift != shift_val):
    #     print("Received shift:", shift_val)
    #     print("\n")
    #     last_shift = shift_val
    #     shift_change_sound.play()
    #     last_floor_val = floored_val
    #     switch_shift = True
    # elif switch_shift:
    #     last_floor_val = floored_val
    #     switch_shift = False
    # elif (last_floor_val != floored_val):
    #     print("Received floor:", floored_val)
    #     print("Received value:", val)
    #     print("\n")
    #     last_floor_val = floored_val
    #     time_count_sound.play()
    # playsound("media/balls2.mp3", block=False)
    # time.sleep(1)