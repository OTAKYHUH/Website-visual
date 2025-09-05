#import packages
import tkinter
from tkinter import ttk
from tkinter import messagebox
from tkcalendar import DateEntry
from datetime import date
import openpyxl
import os

def validate_ywt_input(new_value):
    if new_value == "":
        return True  # Allow empty input (for deletion)
    try:
        float(new_value)  # Allow integers and decimals
        return True
    except ValueError:
        return False
    
def enter_data():
    # Info
    Shift = Shift_dropdown.get()
    ywt = ywt_entry.get()
    dates = cal.get()
    Time = Time_dropdown.get()

    # Check for missing fields
    if not Shift or not ywt or not dates or not Time:
        messagebox.showerror("Input Error", "Please fill in all fields before submitting.")
        return
    
    print("Shift :", Shift, "\nYWT. :", ywt, "\nDate :", dates, "\nTime :", Time)

    # Reset ywt entry
    ywt_entry.delete(0, tkinter.END)

    # Advance Time dropdown to next value or clear if at the end
    time_values = Time_dropdown['values']
    if Time in time_values:
        current_index = time_values.index(Time)
        if current_index < len(time_values) - 1:
            Time_dropdown.set(time_values[current_index + 1])
        else:
            Time_dropdown.set('')  # Clear if at the end
    else:
        Time_dropdown.set(time_values[0])  # Or '' if you want to clear instead
    
    filepath = r"C:/Users/ahgua/Documents/Website visual/Data.xlsx"

    # checks if the file exists
    if not os.path.exists(filepath):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        heading = ["Shift","Date","Time","YWT"]
        sheet.append(heading) 
        workbook.save(filepath)

    workbook = openpyxl.load_workbook(filepath)
    sheet = workbook.active
    sheet.append([Shift, dates, Time, ywt])
    workbook.save(filepath)

#tkinter window
window = tkinter.Tk()
window.title("Data Entry Form")

#How the application will look like
frame = tkinter.Frame(window)
frame.pack()

#Saving User Info
user_info_frame = tkinter.LabelFrame(frame, text="Information")
user_info_frame.grid(row = 0, column=0)

#Shift Label
Shift_Label = tkinter.Label(user_info_frame, text ="Shift")
Shift_Label.grid(row = 0,column = 0)
Shift_dropdown = ttk.Combobox(user_info_frame, values = ["D","N"])
Shift_dropdown.grid(row=0,column = 1)

# YWT label
ywt_label = tkinter.Label(user_info_frame, text="ywt.")
vcmd = window.register(validate_ywt_input)
ywt_entry = tkinter.Entry(user_info_frame, validate="key", validatecommand=(vcmd, "%P"))
ywt_label.grid(row=0, column=2)
ywt_entry.grid(row=0, column=3)

# Date label
Date_label = tkinter.Label(user_info_frame, text ="Date")
Date_label.grid(row=1, column =0)

# DateEntry widget (Calendar picker)
cal = DateEntry(user_info_frame, width=12, maxdate=date.today(), background='darkblue',
                foreground='white', borderwidth=2, year=2025, date_pattern='y-mm-dd')
cal.grid(row=1, column=1)


day_shift_times = ["7:30", "8:30", "9:30", "10:30", "11:30", "12:30","13:30", "14:30", "15:30", "16:30", "17:30", "18:30"]
night_shift_times = ["19:30","20:30","21:30","22:30","23:30","00:30","01:30","02:30","03:30","04:30","05:30","06:30",]

# Time Label
Time_label = tkinter.Label(user_info_frame, text ="Time")
Time_label.grid(row=1, column =2)
# Time widget
Time_dropdown = ttk.Combobox(user_info_frame)
Time_dropdown.grid(row = 1, column = 3)

def update_time_options(event=None):
    shift = Shift_dropdown.get()
    if shift == "D":
        Time_dropdown['values'] = day_shift_times
    elif shift == "N":
        Time_dropdown['values'] = night_shift_times
    else:
        Time_dropdown['values'] = []

# Bind shift dropdown change to update_time_options
Shift_dropdown.bind("<<ComboboxSelected>>", update_time_options)

#Padding for everything
for widget in user_info_frame.winfo_children():
    widget.grid_configure(padx = 10, pady = 20)

# Submit Button
button = tkinter.Button(frame, text = "Submit Data", command = enter_data)
button.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "news")

# Bind Enter key to trigger the submit function
window.bind('<Return>', lambda event: enter_data())

#Make sure it stays open until manually closed
window.mainloop()