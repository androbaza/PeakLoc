import tkinter as tk
import tkinter.filedialog as fd
from customtkinter import HoverInfo

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Program Name")
        self.pack()

        # create label and button for selecting input file
        self.file_label = tk.Label(self, text="Select input file:")
        self.file_label.pack()
        self.file_button = tk.Button(self, text="Choose File", command=self.select_file)
        self.file_button.pack()

        # create label and entry for parameter 1
        self.param1_label = tk.Label(self, text="Parameter 1:")
        self.param1_label.pack()
        self.param1_entry = tk.Entry(self)
        self.param1_entry.insert(0, "Default Value")
        self.param1_entry.pack()
        # create hover info for parameter 1
        self.param1_hover = HoverInfo(self.param1_label, "Parameter 1 Description")

        # create label and entry for parameter 2
        self.param2_label = tk.Label(self, text="Parameter 2:")
        self.param2_label.pack()
        self.param2_entry = tk.Entry(self)
        self.param2_entry.insert(0, "Default Value")
        self.param2_entry.pack()
        # create hover info for parameter 2
        self.param2_hover = HoverInfo(self.param2_label, "Parameter 2 Description")

        # create button to run main code with selected file and parameters
        self.run_button = tk.Button(self, text="Run Program", command=self.run_program)
        self.run_button.pack()

    def select_file(self):
        # open file dialog to select input file
        self.filename = fd.askopenfilename()
        self.file_label.config(text="Input file: " + self.filename)

    def run_program(self):
        # get input file and parameters from GUI entries
        input_file = self.filename
        param1 = self.param1_entry.get()
        param2 = self.param2_entry.get()
        # call main code with selected file and parameters
        main_code(input_file, param1, param2)

def main_code(input_file, param1, param2):
    # main code here
    print("Input file:", input_file)
    print("Parameter 1:", param1)
    print("Parameter 2:", param2)

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
