import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
# from main import process_file

class MyApp:
    def __init__(self, master):
        self.master = master
        self.master.title("My App")
        self.master.geometry("500x300")
        self.master.config(bg="#1c2237")

        self.input_file = None
        self.param1 = tk.StringVar()
        self.param2 = tk.StringVar()
        
        # Select input file button
        self.file_button = tk.Button(
            master, text="Select Input File", bg="#5181b8", fg="white", 
            font=("Arial", 12), command=self.select_file
        )
        self.file_button.pack(pady=20)

        # Parameter 1 input field
        tk.Label(
            master, text="Parameter 1", font=("Arial", 12), fg="white", bg="#1c2237"
        ).pack(pady=10)
        tk.Entry(
            master, textvariable=self.param1, font=("Arial", 12)
        ).pack(pady=5, padx=30, ipady=5)

        # Parameter 2 input field
        tk.Label(
            master, text="Parameter 2", font=("Arial", 12), fg="white", bg="#1c2237"
        ).pack(pady=10)
        tk.Entry(
            master, textvariable=self.param2, font=("Arial", 12)
        ).pack(pady=5, padx=30, ipady=5)

        # Run button
        self.run_button = tk.Button(
            master, text="Run", bg="#5181b8", fg="white", font=("Arial", 12), 
            command=self.run
        )
        self.run_button.pack(pady=20)

        # Set edge rounding and padding for all widgets
        # for widget in self.master.winfo_children():
        #     widget.grid(highlightbackground="#1c2237", highlightthickness=1, 
        #                   borderwidth=0, padx=10, pady=5)
        #     widget.configure(relief=tk.RIDGE, highlightcolor="#5181b8")

    def select_file(self):
        self.input_file = filedialog.askopenfilename()

    def run(self):
        if self.input_file is None:
            messagebox.showerror("Error", "Please select an input file.")
            return

        try:
            param1_val = float(self.param1.get())
            param2_val = float(self.param2.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid parameter values.")
            return

        # Call the main function to process the input file with the given parameters
        data = process_file(self.input_file, param1_val, param2_val)

        # Plot the resulting data
        plt.plot(np.arange(len(data)), data)
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.title("Plot Title")
        plt.show()

root = tk.Tk()
root.geometry("800x800")
app = MyApp(root)
root.mainloop()
