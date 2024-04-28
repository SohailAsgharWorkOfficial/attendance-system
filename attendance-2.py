import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from tkinter import messagebox

# Create the main window
window = tk.Tk()
window.title("Attendance System")
window.configure(background='pink')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

x_cord = 75
y_cord = 20

# GUI elements
lbl_id = tk.Label(window, text="Enter Your College ID", width=20, height=2, fg="black", bg="Pink",
                  font=('Times New Roman', 25, ' bold '))
lbl_id.place(x=200 - x_cord, y=200 - y_cord)

txt_id = tk.Entry(window, width=30, bg="white", fg="blue", font=('Times New Roman', 15, ' bold '))
txt_id.place(x=250 - x_cord, y=300 - y_cord)

lbl_name = tk.Label(window, text="Enter Your Name", width=20, fg="black", bg="pink", height=2,
                    font=('Times New Roman', 25, ' bold '))
lbl_name.place(x=600 - x_cord, y=200 - y_cord)

txt_name = tk.Entry(window, width=30, bg="white", fg="blue", font=('Times New Roman', 15, ' bold '))
txt_name.place(x=650 - x_cord, y=300 - y_cord)

message = tk.Label(window, text="", bg="white", fg="blue", width=30, height=1, activebackground="white",
                   font=('Times New Roman', 15, ' bold '))
message.place(x=1075 - x_cord, y=300 - y_cord)

lbl_attendance = tk.Label(window, text="ATTENDANCE", width=20, fg="white", bg="lightgreen", height=2,
                          font=('Times New Roman', 30, ' bold '))
lbl_attendance.place(x=120, y=570 - y_cord)

message2 = tk.Label(window, text="", fg="red", bg="yellow", activeforeground="green", width=60, height=4,
                    font=('times', 15, ' bold '))
message2.place(x=700, y=570 - y_cord)

lbl_step1 = tk.Label(window, text="STEP 1", width=20, fg="green", bg="pink", height=2,
                     font=('Times New Roman', 20, ' bold '))
lbl_step1.place(x=240 - x_cord, y=375 - y_cord)

lbl_step2 = tk.Label(window, text="STEP 2", width=20, fg="green", bg="pink", height=2,
                     font=('Times New Roman', 20, ' bold '))
lbl_step2.place(x=645 - x_cord, y=375 - y_cord)

lbl_step3 = tk.Label(window, text="STEP 3", width=20, fg="green", bg="pink", height=2,
                     font=('Times New Roman', 20, ' bold '))
lbl_step3.place(x=1100 - x_cord, y=362 - y_cord)


def clear_entries():
    txt_id.delete(0, 'end')
    txt_name.delete(0, 'end')
    message.configure(text="")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def capture_images():
    Id = txt_id.get()
    name = txt_name.get()
    if not Id:
        message.configure(text="Please enter ID")
    elif not name:
        message.configure(text="Please enter Name")
    elif is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        sample_num = 0
        while True:
            ret, img = cam.read()
            cv2.imwrite(f"captures/User.{Id}.{name}.{sample_num}.jpg", img)  # Save in 'captures' folder
            sample_num += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif sample_num == 20:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = f"Images Saved for ID: {Id} Name: {name}"
        message.configure(text=res)
    else:
        if not is_number(Id):
            res = "Enter Numeric ID"
            message.configure(text=res)
        if not name.isalpha():
            res = "Enter Alphabetical Name"
            message.configure(text=res)


def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = get_images_and_labels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    clear_entries()
    message.configure(text=res)
    messagebox.showinfo('Completed', 'Your model has been trained successfully!!')


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        Id = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(image_np)
        Ids.append(Id)
    return faces, Ids


def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = f"{Id}-{df.loc[df['Id'] == Id]['Name'].values[0]}"
                attendance.loc[len(attendance)] = [Id, aa, date, timestamp]
            else:
                Id = 'Unknown'
                aa = str(Id)
            if conf > 75:
                no_of_file = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(f"ImagesUnknown/Image{no_of_file}.jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(aa), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if cv2.waitKey(1) == ord('q'):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    hour, minute, second = timestamp.split(":")
    file_name = f"Attendance/Attendance_{date}_{hour}-{minute}-{second}.csv"
    attendance.to_csv(file_name, index=False)
    cam.release()
    cv2.destroyAllWindows()
    res = "Attendance Taken"
    message2.configure(text=attendance)
    message.configure(text=res)
    messagebox.showinfo('Completed', 'Congratulations! Your attendance has been marked successfully for the day!!')


def quit_window():
    MsgBox = messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application', icon='warning')
    if MsgBox == 'yes':
        messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
        window.destroy()


# Buttons for different actions
btn_capture = tk.Button(window, text="Capture Images", command=capture_images, fg="white", bg="blue", width=25, height=2,
                        activebackground="pink", font=('Times New Roman', 15, ' bold '))
btn_capture.place(x=245 - x_cord, y=425 - y_cord)

btn_train = tk.Button(window, text="Train Model", command=train_images, fg="white", bg="blue", width=25, height=2,
                       activebackground="pink", font=('Times New Roman', 15, ' bold '))
btn_train.place(x=645 - x_cord, y=425 - y_cord)

btn_track = tk.Button(window, text="Mark Attendance", command=track_images, fg="white", bg="red", width=30, height=3,
                       activebackground="pink", font=('Times New Roman', 15, ' bold '))
btn_track.place(x=1075 - x_cord, y=412 - y_cord)

btn_quit = tk.Button(window, text="Quit", command=quit_window, fg="white", bg="red", width=10, height=2,
                      activebackground="pink", font=('Times New Roman', 15, ' bold '))
btn_quit.place(x=700, y=735 - y_cord)

# Start the GUI main loop
window.mainloop()
