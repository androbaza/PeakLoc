import customtkinter as tk
import tkinter.filedialog as filedialog
from tktooltip import ToolTip

# Create the main window
root = tk.CTk()
root.title("PeakLoc")
root.geometry("800x400")

# Create a label and entry for the input file
input_file_label = tk.CTkLabel(root, text="Input File:")
input_file_label.grid(row=0, column=0, padx=5, pady=5)

input_file_entry = tk.CTkEntry(root, width=300)
input_file_entry.grid(row=0, column=1, padx=5, pady=5)

# Create a button to select the input file
def select_input_file():
    input_file_path = filedialog.askopenfilename()
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, input_file_path)

select_input_file_button = tk.CTkButton(root, text="Select File", command=select_input_file)
select_input_file_button.grid(row=0, column=2, padx=5, pady=5)

# Create a label and entry for parameter 1
param1_label = tk.CTkLabel(root, text="Parameter 1:")
param1_label.grid(row=1, column=0, padx=5, pady=5)

param1_entry = tk.CTkEntry(root, width=70)
param1_entry.insert(0, "preset_value")
param1_entry.grid(row=1, column=1, padx=5, pady=5)

# Create a tooltip for parameter 1
# param1_tooltip = tk.CTkTooltip(param1_entry, "Explanation of parameter 1")
ToolTip(param1_entry, msg="Explanation of parameter 1", delay=0.01, follow=True,
        parent_kwargs={"bg": "black", "padx": 3, "pady": 3},
        fg="white", bg="orange", padx=7, pady=7)

# Create a label and entry for parameter 2
param2_label = tk.CTkLabel(root, text="Parameter 2:")
param2_label.grid(row=2, column=0, padx=5, pady=5)

param2_entry = tk.CTkEntry(root, width=70)
param2_entry.insert(0, "preset_value")
param2_entry.grid(row=2, column=1, padx=5, pady=5)

# Create a tooltip for parameter 2
# param2_tooltip = tk.CTkTooltip(param2_entry, "Explanation of parameter 2")

# Create a button to run the main code
def run_main_code():
    input_file_path = input_file_entry.get()
    param1_value = param1_entry.get()
    param2_value = param2_entry.get()

    # Add your code here to process the input file with the given parameters

run_button = tk.CTkButton(root, text="Run", command=run_main_code)
run_button.grid(row=3, column=1, padx=5, pady=5)

# Start the main loop
root.mainloop()
