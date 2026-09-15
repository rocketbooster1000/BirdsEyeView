from ntcore import NetworkTableInstance

ntinst = NetworkTableInstance.getDefault()

# Connect as a client to the robot
ntinst.startClient4("my-dashboard")
# ntinst.setServerTeam(XXXX)  # replace with your team number
# OR use IP directly:
# ntinst.setServer("10.XX.YY.2")
ntinst.setServer("127.0.0.1")

import time
time.sleep(1)  # wait for connection


table = ntinst.getTable("SmartDashboard")

while True:
    # Update values
    # table.getEntry("Speed").setDouble(3.14)
    # table.getEntry("ShooterReady").setBoolean(True)
    # table.getEntry("Status").setString("All systems go")
    
    print(table.getEntry("random").getDouble(0.0))

    time.sleep(1)  # update every second

# Get values
# speed = table.getEntry("Speed").getDouble(0.0)
# ready = table.getEntry("ShooterReady").getBoolean(False)
# status = table.getEntry("Status").getString("unknown")

# print(f"Speed: {speed}, Ready: {ready}, Status: {status}")