import cv2
import face_recognition 
import os
import math
import numpy
import csv
from datetime import datetime
import pandas as pd

# Get a list of all image paths in the imageData directory
imagePaths = [os.path.join("imageData", f) for f in os.listdir("imageData")]

# Create an empty list to store the registration number, name, department, and year of each student
students = []

# Loop through each image and extract the registration number, name, department, and year from the filename
for imagePath in imagePaths:
    filename = os.path.basename(imagePath)
    name, regNumber, department, year = filename.split(".")[0:4]
    students.append((regNumber, name, department, year))

# Write the student list to a CSV file
with open("students.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Registration Number", "Name", "Department", "Year"])
    writer.writerows(students)

videoCapture = cv2.VideoCapture(0)

# Get current date
currentDate = datetime.now().strftime('%Y-%m-%d')

# Create or open CSV file with current date
attendanceFile = f"Attendance_{currentDate}.csv"
try:
    if not os.path.isfile(attendanceFile):
        with open(attendanceFile, "w") as f:
            f.write("Registration Number,Name,Department,Year,Time\n")
except PermissionError:
    print("Error: Permission denied to access the attendance file.")

# Read attendance file and store registration numbers in a set
attendanceRecords = set()
try:
    with open(attendanceFile, "r") as f:
        for line in f.readlines()[1:]:
            regNumber = line.strip().split(",")[0]
            attendanceRecords.add(regNumber)
except PermissionError:
    print("Error: Permission denied to access the attendance file.")

# Set up absentees list
if f'Absentees-{currentDate}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Absentees-{currentDate}.csv','w') as f:
        f.write('Name,Roll,dept,year')
absenteesFile = f"Absentees_{currentDate}.csv"
absenteesList = []
try:
    if not os.path.isfile(absenteesFile):
        with open(absenteesFile, "w") as f:
            f.write("Registration Number,Name,Department,Year\n")
except PermissionError:
    print("Error: Permission denied to access the absentees file.")

# FOR MARKING ATTENDANCE
def markAttendance(name):
    try:
        # Check if the person has already been marked present
        if name not in attendanceRecords:
            # Look up the registration number, department and year for the given name
            for student in students:
                if student[1] == name:
                    regNumber = student[0]
                    department = student[2]
                    year = student[3]
                    break
            else:
                regNumber = "Unknown"
                department = "Unknown"
                year = "Unknown"
            with open(attendanceFile, "a") as f:
                time = datetime.now().strftime('%H:%M:%S')
                f.write(f'{regNumber},{name},{department},{year},{time}\n')
                attendanceRecords.add(name)
    except PermissionError:
        print("Error: Permission denied to access the attendance file.")

def computeAbsentees():
    try:
        StudentsList = pd.read_csv(attendanceFile)
        PresentList = pd.read_csv("students.csv")
        absentees= StudentsList[~StudentsList.isin(PresentList)].append(PresentList[~PresentList.isin(StudentsList)])
        absentees.to_csv(f'Attendance/Absentees-{currentDate}.csv', index=False)
    except PermissionError:
        print("Error: Permission denied to access the absentees file.")

# FOR ADDING TO ABSENTEES LIST
def markAbsentees(regNumber, name, department, year):
    try:
        with open(absenteesFile, "a") as f:
            f.write(f'{regNumber},{name},{department},{year}\n')
            absenteesList.append((regNumber, name, department, year))
    except PermissionError:
        print("Error: Permission denied to access the absentees file.")
        # FOR CHECKING THE ACCURACY
def getAccuracy(faceDistance, faceMatchThreshold = 0.6):
    if faceDistance > faceMatchThreshold:
        range = (1.0 - faceMatchThreshold)
        linearValue = (1.0 - faceDistance) / (range * 2.0)
        return linearValue
    else:
        range = faceMatchThreshold
        linearValue = 1.0 - (faceDistance / (range * 2.0))
        return linearValue + ((1.0 - linearValue) * math.pow((linearValue - 0.5) * 2, 0.2))

# FOR GETTING PATH, NAME, REGISTRATION NO. AND ENCODINGS OF EACH PERSON
allPaths = os.listdir("imageData")
allNames = []
allRegNumbers = []
allEncodings = []
for index in range(len(allPaths)):
    allNames.append(allPaths[index].split(".")[0])
    allRegNumbers.append(allPaths[index].split(".")[1])
    image = face_recognition.load_image_file("imageData/" + allPaths[index])
    temp = face_recognition.face_encodings(image)[0]
    allEncodings.append(temp)

while True:
    print("TCE attendance system")
    print("Menu")
    print("1) Mark attendance")
    print("2) Get absentees list")
    print("3) Exit")
    ch=int(input())
    if(ch==1):
        # READ FRAME FROM VIDEO CAPTURE
        ret, frame = videoCapture.read()

        if not ret:
            # Stop the loop if the frame cannot be read
            break

        frame = cv2.resize(frame, (0, 0), fx=2, fy=1.6)

        resizedFrame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        requiredFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)

        faceLocation = face_recognition.face_locations(requiredFrame)


        faceEncoding = face_recognition.face_encodings(requiredFrame, faceLocation)

        faceNames = []
        for encoding in faceEncoding:
            ismatched = face_recognition.compare_faces(allEncodings, encoding)
        matchedName = "Unknown"

        faceDistance = face_recognition.face_distance(allEncodings, encoding)

        bestMatchIndex = numpy.argmin(faceDistance)

        if ismatched[bestMatchIndex]:
            minimumFaceDistance = faceDistance[bestMatchIndex]
            accuracy = getAccuracy(minimumFaceDistance) * 100

            if accuracy > 80:
                matchedName = allNames[bestMatchIndex]
                markAttendance(matchedName)

            faceNames.append(matchedName)


        for (top, right, bottom, left), name in zip(faceLocation, faceNames):
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(frame, name, (left + 6, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            if(accuracy > 80):
                cv2.putText(frame, "%.2f"%accuracy + "%", (left + 6, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            

            #cv2.rectangle(frame, (faceCoordinates[3], faceCoordinates[0]), (faceCoordinates[1], faceCoordinates[2]), (0, 255, 0), 3)
            #cv2.putText(frame, matchedName, (faceCoordinates[3] + 6, faceCoordinates[2] - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
            
        cv2.imshow("Recording video", frame)
        cv2.waitKey(1)
    elif(ch==2):
        print("Please hold on till the absentees list is generated")
        computeAbsentees()
        print("The absentees file is successfully generated, check out the Attendance folder")
    else:
        print("thank you")
        break